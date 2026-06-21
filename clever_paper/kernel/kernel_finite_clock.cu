// kernel_finite_clock.cu
//
// Finite-clock CUDA projector (v1.9 Delta-range proposal + v1.8 delay-quadrature compatible paired complex dual witness) for the QuTiP/CUDA verification pipeline.
//
// Core contract:
//   theta_i is represented by an integer clock state k_i in {0, ..., q-1}
//   theta(k) = 2*pi*k/q
//
// Proposal process:
//   choose site i uniformly from N=L*L
//   proposal_id=0: choose new clock state k' uniformly from q-1 states excluding k_i
//   proposal_id=1: choose a nonzero integer clock offset uniformly from +/- window_radius
//   proposal_id=2: draw continuous delta theta uniformly from [-proposal_delta_rad, +proposal_delta_rad],
//                  then quantize to the nearest clock tick; zero/sub-tick proposals are allowed
//   compute DeltaE using the paper-derived local rule
//   accept with w = 0.5 * (1 - tanh(DeltaE/(2T)))
//
// Method IDs:
//   0 = constrained-Hamiltonian Glauber DeltaE
//   1 = selfish-energy control DeltaE
//
// Public kernels:
//   init_rng_states_kernel
//   init_clock_ordered_kernel
//   init_clock_random_kernel
//   run_finite_clock_projector_kernel
//   compute_clock_observables_kernel
//   compute_local_witness_kernel
//   apply_frozen_insertion_kernel
//   compute_clock_rate_parity_kernel
//
// v1.10 note: proposal_id=2 now samples the paper's continuous Delta range first,
// then quantizes onto the finite clock.  This differs from proposal_id=1 because
// it includes the zero/sub-tick mass that appears whenever |delta theta| rounds
// back to the current clock state.
// This file is NVRTC-friendly: no includes required.

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

typedef unsigned long long uint64_t;

static constexpr double PI_D = 3.141592653589793238462643383279502884;
static constexpr double TWO_PI_D = 6.283185307179586476925286766559005768;
static constexpr int MAX_CLOCK_LUT_Q = 64;

extern "C" {
__constant__ int CLOCK_LUT_Q[1];
__constant__ double CLOCK_COS_K[MAX_CLOCK_LUT_Q];
__constant__ double CLOCK_SIN_K[MAX_CLOCK_LUT_Q];
__constant__ double CLOCK_COS_DELTA[MAX_CLOCK_LUT_Q * MAX_CLOCK_LUT_Q];
__constant__ unsigned char CLOCK_VISION4[MAX_CLOCK_LUT_Q * 4];
}

// -----------------------------------------------------------------------------
// RNG: SplitMix64, deterministic and dependency-free.
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint64_t splitmix64_next(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ double rng_uniform01(uint64_t *state)
{
    const uint64_t x = splitmix64_next(state) >> 11;
    return (double)x * (1.0 / 9007199254740992.0); // 2^-53
}

__device__ __forceinline__ int rng_int(uint64_t *state, int n)
{
    if (n <= 1)
        return 0;
    int out = (int)(rng_uniform01(state) * (double)n);
    if (out >= n)
        out = n - 1;
    return out;
}

__device__ __forceinline__ int wrap_clock_k_device(int k, int q)
{
    k %= q;
    if (k < 0)
        k += q;
    return k;
}

__device__ __forceinline__ int nearest_clock_offset_from_delta_device(double delta_theta, int q)
{
    const double tick = TWO_PI_D / (double)q;
    const double x = delta_theta / tick;
    return (int)floor(x + 0.5);
}

__device__ __forceinline__ int sample_clock_proposal_device(
    uint64_t *rng,
    int old_k,
    int q,
    int proposal_id,
    int window_radius,
    double proposal_delta_rad)
{
    if (proposal_id == 0)
    {
        int r = rng_int(rng, q - 1);
        return (r < old_k) ? r : (r + 1);
    }

    if (proposal_id == 2)
    {
        const double delta_theta = (2.0 * rng_uniform01(rng) - 1.0) * proposal_delta_rad;
        const int delta_k = nearest_clock_offset_from_delta_device(delta_theta, q);
        return wrap_clock_k_device(old_k + delta_k, q);
    }

    int w = window_radius;
    if (w < 1)
        w = 1;
    const int max_w = (q - 1) / 2;
    if (w > max_w)
        w = max_w;
    int r = rng_int(rng, 2 * w);
    int delta_k = r - w;
    if (delta_k >= 0)
        delta_k += 1;
    return wrap_clock_k_device(old_k + delta_k, q);
}

extern "C" __global__ void init_rng_states_kernel(
    uint64_t *rng_states,
    int n_traj,
    uint64_t seed_root)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_traj)
        return;

    uint64_t s = seed_root ^ (0xD1B54A32D192ED03ULL * (uint64_t)(tid + 1)) ^ (0x94D049BB133111EBULL * (uint64_t)(tid + 0x9E37));

    splitmix64_next(&s);
    rng_states[tid] = s;
}

// -----------------------------------------------------------------------------
// Scalar math contract.
// -----------------------------------------------------------------------------

__device__ __forceinline__ double wrap_angle_device(double theta)
{
    double out = fmod(theta, TWO_PI_D);
    if (out < 0.0)
        out += TWO_PI_D;
    return out;
}

__device__ __forceinline__ double circ_dist_device(double a, double b)
{
    double d = fmod(a - b + PI_D, TWO_PI_D);
    if (d < 0.0)
        d += TWO_PI_D;
    d -= PI_D;
    return fabs(d);
}

__device__ __forceinline__ double clock_angle_device(int k, int q)
{
    return TWO_PI_D * (double)k / (double)q;
}

__device__ __forceinline__ bool clock_lut_enabled_device(int q)
{
    return q > 0 && q <= MAX_CLOCK_LUT_Q && CLOCK_LUT_Q[0] == q;
}

__device__ __forceinline__ double cos_k_device(int k, int q)
{
    if (clock_lut_enabled_device(q))
        return CLOCK_COS_K[k];
    return cos(clock_angle_device(k, q));
}

__device__ __forceinline__ double sin_k_device(int k, int q)
{
    if (clock_lut_enabled_device(q))
        return CLOCK_SIN_K[k];
    return sin(clock_angle_device(k, q));
}

__device__ __forceinline__ double cos_delta_k_device(int ka, int kb, int q)
{
    if (clock_lut_enabled_device(q))
        return CLOCK_COS_DELTA[ka * MAX_CLOCK_LUT_Q + kb];
    return cos(clock_angle_device(ka, q) - clock_angle_device(kb, q));
}

__device__ __forceinline__ double vision_J_clock_device(
    int k,
    int dir_id,
    int q,
    double fallback_psi_ij,
    double Psi,
    double J)
{
    if (clock_lut_enabled_device(q))
        return CLOCK_VISION4[k * 4 + dir_id] ? J : 0.0;
    return (circ_dist_device(clock_angle_device(k, q), fallback_psi_ij) <= Psi) ? J : 0.0;
}

__device__ __forceinline__ double vision_J_device(
    double theta_i,
    double psi_ij,
    double Psi,
    double J)
{
    return (circ_dist_device(theta_i, psi_ij) <= Psi) ? J : 0.0;
}

__device__ __forceinline__ double glauber_rate_device(double delta_E, double T)
{
    return 0.5 * (1.0 - tanh(delta_E / (2.0 * T)));
}

// -----------------------------------------------------------------------------
// Initialization kernels.
// -----------------------------------------------------------------------------

extern "C" __global__ void init_clock_ordered_kernel(
    int *state,
    int n_total,
    int k0)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_total)
        return;
    state[tid] = k0;
}

extern "C" __global__ void init_clock_random_kernel(
    int *state,
    uint64_t *rng_states,
    int L,
    int n_traj,
    int q)
{
    const int traj = blockIdx.x * blockDim.x + threadIdx.x;
    if (traj >= n_traj)
        return;

    const int N = L * L;
    const int base = traj * N;
    uint64_t rng = rng_states[traj];

    for (int i = 0; i < N; ++i)
    {
        state[base + i] = rng_int(&rng, q);
    }

    rng_states[traj] = rng;
}

extern "C" __global__ void compute_local_witness_kernel(
    const int *state_constrained,
    const int *state_selfish,
    const double *cphi,
    const double *sphi,
    int L,
    int q,
    int n_traj,
    double *a_i,
    double *psi_g_re_i,
    double *psi_g_im_i)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = L * L;
    const int total = n_traj * N;
    if (tid >= total)
        return;

    const int traj = tid / N;
    const int kc = state_constrained[tid];
    const int ks = state_selfish[tid];

    const double zc_re = cos_k_device(kc, q);
    const double zc_im = sin_k_device(kc, q);
    const double zs_re = cos_k_device(ks, q);
    const double zs_im = sin_k_device(ks, q);

    const double psi_re = zc_re - zs_re;
    const double psi_im = zc_im - zs_im;
    const double c = cphi[traj];
    const double s = sphi[traj];

    psi_g_re_i[tid] = psi_re * c + psi_im * s;
    psi_g_im_i[tid] = psi_im * c - psi_re * s;
    a_i[tid] = sqrt(psi_re * psi_re + psi_im * psi_im);
}

// -----------------------------------------------------------------------------
// DeltaE rules on integer finite-clock state.
// -----------------------------------------------------------------------------

__device__ __forceinline__ double delta_E_constrained_clock_device(
    const int *state,
    int base,
    int i,
    int new_k,
    int q,
    const int *neighbor_idx,
    const double *neighbor_psi,
    double Psi,
    double J)
{
    const int old_k = state[base + i];

    double acc = 0.0;

#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
    {
        const int offset = i * 4 + kk;
        const int j = neighbor_idx[offset];
        const double psi_ij = neighbor_psi[offset];
        const int kj = state[base + j];

        const double Jij_new = vision_J_clock_device(new_k, kk, q, psi_ij, Psi, J);
        const double Jij_old = vision_J_clock_device(old_k, kk, q, psi_ij, Psi, J);

        acc += (Jij_new + Jij_old) *
               (cos_delta_k_device(new_k, kj, q) - cos_delta_k_device(old_k, kj, q));
    }

    return -0.5 * acc;
}

__device__ __forceinline__ double delta_E_selfish_clock_device(
    const int *state,
    int base,
    int i,
    int new_k,
    int q,
    const int *neighbor_idx,
    const double *neighbor_psi,
    double Psi,
    double J)
{
    const int old_k = state[base + i];

    double acc = 0.0;

#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
    {
        const int offset = i * 4 + kk;
        const int j = neighbor_idx[offset];
        const double psi_ij = neighbor_psi[offset];
        const int kj = state[base + j];

        const double Jij_new = vision_J_clock_device(new_k, kk, q, psi_ij, Psi, J);
        const double Jij_old = vision_J_clock_device(old_k, kk, q, psi_ij, Psi, J);

        const double e_new = -Jij_new * cos_delta_k_device(new_k, kj, q);
        const double e_old = -Jij_old * cos_delta_k_device(old_k, kj, q);
        acc += e_new - e_old;
    }

    return acc;
}

__device__ __forceinline__ double delta_E_clock_device(
    const int *state,
    int base,
    int i,
    int new_k,
    int q,
    int method_id,
    const int *neighbor_idx,
    const double *neighbor_psi,
    double Psi,
    double J)
{
    if (method_id == 0)
    {
        return delta_E_constrained_clock_device(
            state, base, i, new_k, q, neighbor_idx, neighbor_psi, Psi, J);
    }
    return delta_E_selfish_clock_device(
        state, base, i, new_k, q, neighbor_idx, neighbor_psi, Psi, J);
}

// -----------------------------------------------------------------------------
// Observables.
// -----------------------------------------------------------------------------

__device__ double magnetization_clock_device(
    const int *state,
    int base,
    int L,
    int q)
{
    const int N = L * L;
    double mx = 0.0;
    double my = 0.0;

    for (int i = 0; i < N; ++i)
    {
        const int k = state[base + i];
        mx += cos_k_device(k, q);
        my += sin_k_device(k, q);
    }

    return sqrt(mx * mx + my * my) / (double)N;
}

__device__ void magnetization_components_clock_device(
    const int *state,
    int base,
    int L,
    int q,
    double *out_z_re,
    double *out_z_im,
    double *out_abs)
{
    const int N = L * L;
    double mx = 0.0;
    double my = 0.0;

    for (int i = 0; i < N; ++i)
    {
        const int k = state[base + i];
        mx += cos_k_device(k, q);
        my += sin_k_device(k, q);
    }

    const double z_re = mx / (double)N;
    const double z_im = my / (double)N;
    *out_z_re = z_re;
    *out_z_im = z_im;
    *out_abs = sqrt(z_re * z_re + z_im * z_im);
}

__device__ double HSS_clock_device(
    const int *state,
    int base,
    int n_bonds,
    int q,
    const int *bond_i,
    const int *bond_j,
    const double *bond_psi_ij,
    const double *bond_psi_ji,
    double Psi,
    double J)
{
    double H = 0.0;

    for (int b = 0; b < n_bonds; ++b)
    {
        const int i = bond_i[b];
        const int j = bond_j[b];
        const int ki = state[base + i];
        const int kj = state[base + j];

        const int dir_ij = (b & 1) ? 2 : 0;
        const int dir_ji = (b & 1) ? 3 : 1;
        const double Jij = vision_J_clock_device(ki, dir_ij, q, bond_psi_ij[b], Psi, J);
        const double Jji = vision_J_clock_device(kj, dir_ji, q, bond_psi_ji[b], Psi, J);

        H += -(Jij + Jji) * cos_delta_k_device(ki, kj, q);
    }

    return H;
}

extern "C" __global__ void compute_clock_observables_kernel(
    const int *state,
    int L,
    int n_traj,
    int q,
    double Psi,
    double J,
    const int *bond_i,
    const int *bond_j,
    const double *bond_psi_ij,
    const double *bond_psi_ji,
    int n_bonds,
    double *out_m,
    double *out_HSS)
{
    const int traj = blockIdx.x * blockDim.x + threadIdx.x;
    if (traj >= n_traj)
        return;

    const int N = L * L;
    const int base = traj * N;

    out_m[traj] = magnetization_clock_device(state, base, L, q);
    out_HSS[traj] = HSS_clock_device(
        state, base, n_bonds, q, bond_i, bond_j, bond_psi_ij, bond_psi_ji, Psi, J);
}

// -----------------------------------------------------------------------------
// Main finite-clock projector.
// -----------------------------------------------------------------------------

extern "C" __global__ void run_finite_clock_projector_kernel(
    int *state,
    uint64_t *rng_states,

    int L,
    int n_traj,
    int q,
    int n_steps,
    int sample_stride,
    int global_step_offset,

    double T,
    double Psi,
    double J,
    int method_id,
    int proposal_id,
    int window_radius,
    double proposal_delta_rad,

    const int *neighbor_idx,
    const double *neighbor_psi,

    const int *bond_i,
    const int *bond_j,
    const double *bond_psi_ij,
    const double *bond_psi_ji,
    int n_bonds,

    double *out_sum_m,
    double *out_sum_m2,
    double *out_sum_H,
    double *out_sum_H2,
    uint64_t *out_accept,
    uint64_t *out_propose,
    uint64_t *out_samples)
{
    const int traj = blockIdx.x * blockDim.x + threadIdx.x;
    if (traj >= n_traj)
        return;

    const int N = L * L;
    const int base = traj * N;
    uint64_t rng = rng_states[traj];

    uint64_t accept = 0ULL;
    uint64_t propose = 0ULL;
    uint64_t samples = 0ULL;

    double sum_m = 0.0;
    double sum_m2 = 0.0;
    double sum_H = 0.0;
    double sum_H2 = 0.0;

    for (int step = 0; step < n_steps; ++step)
    {
        const int i = rng_int(&rng, N);
        const int old_k = state[base + i];
        const int new_k = sample_clock_proposal_device(
            &rng, old_k, q, proposal_id, window_radius, proposal_delta_rad);

        const double dE = delta_E_clock_device(
            state, base, i, new_k, q, method_id, neighbor_idx, neighbor_psi, Psi, J);
        const double rate = glauber_rate_device(dE, T);
        const double u = rng_uniform01(&rng);

        propose += 1ULL;
        if (new_k != old_k && u < rate)
        {
            state[base + i] = new_k;
            accept += 1ULL;
        }

        const int global_step = global_step_offset + step;
        if (sample_stride > 0 && ((global_step % sample_stride) == 0))
        {
            const double m = magnetization_clock_device(state, base, L, q);
            const double H = HSS_clock_device(
                state, base, n_bonds, q,
                bond_i, bond_j, bond_psi_ij, bond_psi_ji, Psi, J);

            sum_m += m;
            sum_m2 += m * m;
            sum_H += H;
            sum_H2 += H * H;
            samples += 1ULL;
        }
    }

    rng_states[traj] = rng;

    out_sum_m[traj] = sum_m;
    out_sum_m2[traj] = sum_m2;
    out_sum_H[traj] = sum_H;
    out_sum_H2[traj] = sum_H2;
    out_accept[traj] = accept;
    out_propose[traj] = propose;
    out_samples[traj] = samples;
}

// -----------------------------------------------------------------------------
// Paired constrained/selfish dual-witness projector.
//
// This kernel evolves two branches from identical initial states using a shared
// proposal stream per trajectory.  Each step chooses one site, one proposed
// clock state, and one accept uniform u.  The constrained branch and selfish
// branch then apply their own DeltaE/rate to their own state.  This is the
// common-random-numbers dual witness: branch separation is caused by the rule,
// not by unrelated stochastic streams.
// -----------------------------------------------------------------------------

extern "C" __global__ void run_finite_clock_dual_witness_pair_kernel(
    int *state_constrained,
    int *state_selfish,
    uint64_t *rng_states,

    int L,
    int n_traj,
    int q,
    int n_steps,
    int sample_stride,
    int global_step_offset,

    double T,
    double Psi,
    double J,
    int proposal_id,
    int window_radius,
    double proposal_delta_rad,

    const int *neighbor_idx,
    const double *neighbor_psi,

    const int *bond_i,
    const int *bond_j,
    const double *bond_psi_ij,
    const double *bond_psi_ji,
    int n_bonds,

    double *out_c_sum_m,
    double *out_c_sum_m2,
    double *out_c_sum_H,
    double *out_c_sum_H2,
    uint64_t *out_c_accept,
    uint64_t *out_c_propose,

    double *out_s_sum_m,
    double *out_s_sum_m2,
    double *out_s_sum_H,
    double *out_s_sum_H2,
    uint64_t *out_s_accept,
    uint64_t *out_s_propose,

    uint64_t *out_samples,

    double *out_sum_zc_re,
    double *out_sum_zc_im,
    double *out_sum_zs_re,
    double *out_sum_zs_im,
    double *out_sum_psi_re,
    double *out_sum_psi_im,
    double *out_sum_psi_abs,
    double *out_sum_psi_abs2,
    double *out_sum_branch_dot,
    double *out_sum_branch_cross,
    double *out_sum_gauge_zs_re,
    double *out_sum_gauge_zs_im,
    double *out_sum_gauge_psi_re,
    double *out_sum_gauge_psi_im,
    double *out_sum_gauge_psi_abs,
    double *out_sum_gauge_psi_abs2,
    double *out_sum_phase_delta,
    double *out_sum_abs_phase_delta,
    double *out_sum_phase_delta2)
{
    const int traj = blockIdx.x * blockDim.x + threadIdx.x;
    if (traj >= n_traj)
        return;

    const int N = L * L;
    const int base = traj * N;
    uint64_t rng = rng_states[traj];

    uint64_t c_accept = 0ULL;
    uint64_t s_accept = 0ULL;
    uint64_t c_propose = 0ULL;
    uint64_t s_propose = 0ULL;
    uint64_t samples = 0ULL;

    double c_sum_m = 0.0, c_sum_m2 = 0.0, c_sum_H = 0.0, c_sum_H2 = 0.0;
    double s_sum_m = 0.0, s_sum_m2 = 0.0, s_sum_H = 0.0, s_sum_H2 = 0.0;

    double sum_zc_re = 0.0, sum_zc_im = 0.0;
    double sum_zs_re = 0.0, sum_zs_im = 0.0;
    double sum_psi_re = 0.0, sum_psi_im = 0.0;
    double sum_psi_abs = 0.0, sum_psi_abs2 = 0.0;
    double sum_branch_dot = 0.0, sum_branch_cross = 0.0;
    double sum_gauge_zs_re = 0.0, sum_gauge_zs_im = 0.0;
    double sum_gauge_psi_re = 0.0, sum_gauge_psi_im = 0.0;
    double sum_gauge_psi_abs = 0.0, sum_gauge_psi_abs2 = 0.0;
    double sum_phase_delta = 0.0, sum_abs_phase_delta = 0.0, sum_phase_delta2 = 0.0;

    for (int local_step = 0; local_step < n_steps; ++local_step)
    {
        const int i = rng_int(&rng, N);
        const int old_k_c = state_constrained[base + i];
        const int old_k_s = state_selfish[base + i];
        const int new_k_c = sample_clock_proposal_device(
            &rng, old_k_c, q, proposal_id, window_radius, proposal_delta_rad);

        // Apply the same signed offset to selfish branch when local-windowed.
        // For global proposals, use the same absolute new clock state.  This keeps
        // the proposal stream common while respecting each branch's current state.
        int new_k_s;
        if (proposal_id == 0)
        {
            new_k_s = new_k_c;
            if (new_k_s == old_k_s)
            {
                new_k_s = (new_k_s + 1) % q;
            }
        }
        else
        {
            int delta_k = new_k_c - old_k_c;
            if (delta_k > q / 2)
                delta_k -= q;
            if (delta_k < -q / 2)
                delta_k += q;
            new_k_s = state_selfish[base + i] + delta_k;
            new_k_s = wrap_clock_k_device(new_k_s, q);
        }

        const double u = rng_uniform01(&rng);

        const double dE_c = delta_E_clock_device(
            state_constrained, base, i, new_k_c, q, 0, neighbor_idx, neighbor_psi, Psi, J);
        const double rate_c = glauber_rate_device(dE_c, T);
        c_propose += 1ULL;
        if (new_k_c != old_k_c && u < rate_c)
        {
            state_constrained[base + i] = new_k_c;
            c_accept += 1ULL;
        }

        const double dE_s = delta_E_clock_device(
            state_selfish, base, i, new_k_s, q, 1, neighbor_idx, neighbor_psi, Psi, J);
        const double rate_s = glauber_rate_device(dE_s, T);
        s_propose += 1ULL;
        if (new_k_s != old_k_s && u < rate_s)
        {
            state_selfish[base + i] = new_k_s;
            s_accept += 1ULL;
        }

        const int global_step = global_step_offset + local_step;
        if (sample_stride > 0 && ((global_step % sample_stride) == 0))
        {
            double zc_re, zc_im, mc;
            double zs_re, zs_im, ms;
            magnetization_components_clock_device(state_constrained, base, L, q, &zc_re, &zc_im, &mc);
            magnetization_components_clock_device(state_selfish, base, L, q, &zs_re, &zs_im, &ms);

            const double Hc = HSS_clock_device(
                state_constrained, base, n_bonds, q,
                bond_i, bond_j, bond_psi_ij, bond_psi_ji, Psi, J);
            const double Hs = HSS_clock_device(
                state_selfish, base, n_bonds, q,
                bond_i, bond_j, bond_psi_ij, bond_psi_ji, Psi, J);

            const double psi_re = zc_re - zs_re;
            const double psi_im = zc_im - zs_im;
            const double psi_abs = sqrt(psi_re * psi_re + psi_im * psi_im);
            const double branch_dot = zc_re * zs_re + zc_im * zs_im;
            const double branch_cross = zc_re * zs_im - zc_im * zs_re;

            // Gauge-fixed witness: rotate each paired sample so constrained Z_c
            // lies on the positive real axis.  This removes arbitrary XY global
            // orientation before ensemble averaging.  In this frame:
            //   gauge_psi_re = radial/amplitude correction channel
            //   gauge_psi_im = transverse/phase-slip channel
            double gauge_zs_re = zs_re;
            double gauge_zs_im = zs_im;
            double gauge_psi_re = psi_re;
            double gauge_psi_im = psi_im;
            if (mc > 1.0e-30)
            {
                const double cphi = zc_re / mc;
                const double sphi = zc_im / mc;
                // Zs * exp(-i arg(Zc))
                gauge_zs_re = zs_re * cphi + zs_im * sphi;
                gauge_zs_im = zs_im * cphi - zs_re * sphi;
                // Zc_gauge = mc + 0i, so Psi_gauge = Zc_gauge - Zs_gauge.
                gauge_psi_re = mc - gauge_zs_re;
                gauge_psi_im = -gauge_zs_im;
            }
            const double gauge_psi_abs = sqrt(gauge_psi_re * gauge_psi_re + gauge_psi_im * gauge_psi_im);
            const double phase_delta = atan2(branch_cross, branch_dot);

            c_sum_m += mc;
            c_sum_m2 += mc * mc;
            c_sum_H += Hc;
            c_sum_H2 += Hc * Hc;
            s_sum_m += ms;
            s_sum_m2 += ms * ms;
            s_sum_H += Hs;
            s_sum_H2 += Hs * Hs;

            sum_zc_re += zc_re;
            sum_zc_im += zc_im;
            sum_zs_re += zs_re;
            sum_zs_im += zs_im;
            sum_psi_re += psi_re;
            sum_psi_im += psi_im;
            sum_psi_abs += psi_abs;
            sum_psi_abs2 += psi_abs * psi_abs;
            sum_branch_dot += branch_dot;
            sum_branch_cross += branch_cross;
            sum_gauge_zs_re += gauge_zs_re;
            sum_gauge_zs_im += gauge_zs_im;
            sum_gauge_psi_re += gauge_psi_re;
            sum_gauge_psi_im += gauge_psi_im;
            sum_gauge_psi_abs += gauge_psi_abs;
            sum_gauge_psi_abs2 += gauge_psi_abs * gauge_psi_abs;
            sum_phase_delta += phase_delta;
            sum_abs_phase_delta += fabs(phase_delta);
            sum_phase_delta2 += phase_delta * phase_delta;
            samples += 1ULL;
        }
    }

    rng_states[traj] = rng;

    out_c_sum_m[traj] = c_sum_m;
    out_c_sum_m2[traj] = c_sum_m2;
    out_c_sum_H[traj] = c_sum_H;
    out_c_sum_H2[traj] = c_sum_H2;
    out_c_accept[traj] = c_accept;
    out_c_propose[traj] = c_propose;

    out_s_sum_m[traj] = s_sum_m;
    out_s_sum_m2[traj] = s_sum_m2;
    out_s_sum_H[traj] = s_sum_H;
    out_s_sum_H2[traj] = s_sum_H2;
    out_s_accept[traj] = s_accept;
    out_s_propose[traj] = s_propose;

    out_samples[traj] = samples;

    out_sum_zc_re[traj] = sum_zc_re;
    out_sum_zc_im[traj] = sum_zc_im;
    out_sum_zs_re[traj] = sum_zs_re;
    out_sum_zs_im[traj] = sum_zs_im;
    out_sum_psi_re[traj] = sum_psi_re;
    out_sum_psi_im[traj] = sum_psi_im;
    out_sum_psi_abs[traj] = sum_psi_abs;
    out_sum_psi_abs2[traj] = sum_psi_abs2;
    out_sum_branch_dot[traj] = sum_branch_dot;
    out_sum_branch_cross[traj] = sum_branch_cross;
    out_sum_gauge_zs_re[traj] = sum_gauge_zs_re;
    out_sum_gauge_zs_im[traj] = sum_gauge_zs_im;
    out_sum_gauge_psi_re[traj] = sum_gauge_psi_re;
    out_sum_gauge_psi_im[traj] = sum_gauge_psi_im;
    out_sum_gauge_psi_abs[traj] = sum_gauge_psi_abs;
    out_sum_gauge_psi_abs2[traj] = sum_gauge_psi_abs2;
    out_sum_phase_delta[traj] = sum_phase_delta;
    out_sum_abs_phase_delta[traj] = sum_abs_phase_delta;
    out_sum_phase_delta2[traj] = sum_phase_delta2;
}

// -----------------------------------------------------------------------------
// Deterministic rate-parity kernel.
// -----------------------------------------------------------------------------

extern "C" __global__ void apply_frozen_insertion_kernel(
    int *state,
    const int *anchor_traj,
    const int *anchor_site,
    const double *kick_unit,
    int n_anchor_rows,
    int N,
    int q,
    double lambda0,
    int *out_delta_k,
    int *out_before_k,
    int *out_after_k,
    int *out_did_kick)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_anchor_rows)
        return;

    const int traj = anchor_traj[row];
    const int site = anchor_site[row];
    const int idx = traj * N + site;
    const int before = state[idx];
    const double kick_float = lambda0 * kick_unit[row];
    const int sign = (kick_float >= 0.0) ? 1 : -1;
    const int delta_k = sign * (int)floor(fabs(kick_float) + 0.5);
    const int after = wrap_clock_k_device(before + delta_k, q);
    const int did_kick = (delta_k != 0) ? 1 : 0;

    if (did_kick)
        state[idx] = after;

    out_delta_k[row] = delta_k;
    out_before_k[row] = before;
    out_after_k[row] = after;
    out_did_kick[row] = did_kick;
}

extern "C" __global__ void compute_clock_rate_parity_kernel(
    const int *state,
    const int *proposal_site,
    const int *proposal_new_k,
    int n_cases,
    int L,
    int q,
    double T,
    double Psi,
    double J,
    int method_id,
    const int *neighbor_idx,
    const double *neighbor_psi,
    double *out_delta_E,
    double *out_rate)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cases)
        return;

    const int N = L * L;
    const int base = tid * N;
    const int i = proposal_site[tid];
    const int new_k = proposal_new_k[tid];

    const double dE = delta_E_clock_device(
        state, base, i, new_k, q, method_id, neighbor_idx, neighbor_psi, Psi, J);

    out_delta_E[tid] = dE;
    out_rate[tid] = glauber_rate_device(dE, T);
}
