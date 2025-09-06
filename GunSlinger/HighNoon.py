#!/usr/bin/env python3
"""
gc3d_torch.py — 3D Gunslinger Continuum (GC) on GPU (RTX 4050 ready)
- Incompressible 3D periodic box
- Anisotropic viscosity via (a·∇)^2
- Readiness drag: -Rd * chi * P_perp(a) u
- Semi-implicit linear step in Fourier space + projection
- Mixed precision-friendly ops; uses cuFFT via torch.fft
CC0 — public domain.
"""
import argparse, csv, math
import torch

# ---------- utils ----------
def device_and_precision(device_arg):
    dev = torch.device(device_arg if torch.cuda.is_available() and device_arg.startswith("cuda") else "cpu")
    # Enable TF32 for speed on NVIDIA (safe for simulations)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Optional: slightly reduced precision but faster GEMMs
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass
    return dev

def make_grid(N, Lx, Ly, Lz, device):
    # Periodic coordinates without duplicating the endpoint:
    x = torch.arange(N, device=device, dtype=torch.float32) * (Lx / N)
    y = torch.arange(N, device=device, dtype=torch.float32) * (Ly / N)
    z = torch.arange(N, device=device, dtype=torch.float32) * (Lz / N)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Wave numbers (2π * k / L). fftfreq returns CPU → move to device.
    kx = 2*math.pi*torch.fft.fftfreq(N, d=Lx/N).to(device)
    ky = 2*math.pi*torch.fft.fftfreq(N, d=Ly/N).to(device)
    kz = 2*math.pi*torch.fft.fftfreq(N, d=Lz/N).to(device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX*KX + KY*KY + KZ*KZ
    K2[0,0,0] = 1.0  # avoid /0; we zero DC mode later
    return X, Y, Z, KX, KY, KZ, K2

def init_field(X, Y, Z):
    # 3D streamfunction-style seed → approximately divergence-free
    psi = 0.5*torch.cos(2*X)*torch.cos(3*Y)*torch.cos(2*Z) + 0.25*torch.cos(3*X+2*Y+Z)
    dx = (X[1,0,0]-X[0,0,0]).item()
    dy = (Y[0,1,0]-Y[0,0,0]).item()
    dz = (Z[0,0,1]-Z[0,0,0]).item()
    # torch.gradient returns tuple in same order as dims
    gX, gY, gZ = torch.gradient(psi, spacing=(dx,dy,dz), dim=(0,1,2))
    u =  gY
    v = -gX
    w =  0.1*gZ
    # small noise
    gen = torch.Generator(device=u.device).manual_seed(0)
    u = u + 0.01*torch.randn_like(u)
    v = v + 0.01*torch.randn_like(v)
    w = w + 0.01*torch.randn_like(w)
    return u, v, w

def project_incompressible(u_hat, v_hat, w_hat, KX, KY, KZ, K2):
    # P(k) = I - kk^T/|k|^2
    kdotu = KX*u_hat + KY*v_hat + KZ*w_hat
    u_hat = u_hat - (KX * kdotu) / K2
    v_hat = v_hat - (KY * kdotu) / K2
    w_hat = w_hat - (KZ * kdotu) / K2
    # kill DC
    u_hat[0,0,0] = 0
    v_hat[0,0,0] = 0
    w_hat[0,0,0] = 0
    return u_hat, v_hat, w_hat

@torch.no_grad()
def kinetic_energy(u, v, w):
    return 0.5*torch.mean(u*u + v*v + w*w).item()

@torch.no_grad()
def gunslinger_index_3d(u, v, w, a, eps=1e-12):
    # GI = < |P_perp(a) u| / |u| >
    adot = a[0]*u + a[1]*v + a[2]*w
    upx = u - a[0]*adot
    upy = v - a[1]*adot
    upz = w - a[2]*adot
    num = torch.sqrt(torch.clamp(upx*upx + upy*upy + upz*upz, min=0))
    den = torch.sqrt(torch.clamp(u*u+v*v+w*w, min=0)) + eps
    return torch.mean(num/den).item()

def finite_diff_grad(u, h, dim):
    return (torch.roll(u, -1, dims=dim) - torch.roll(u, 1, dims=dim)) / (2*h)

# ---------- main integrator ----------
def run_gc3d(N=64, L=2*math.pi, steps=200, dt=5e-3,
             Re=200.0, An=2.0, Rd=3.0, a_vec=(1,0,0),
             anneal_tau_An=None, anneal_tau_Rd=None,
             stripe=False, stripe_axis='z', stripe_strength=2.0, stripe_center=0.5, stripe_width=0.25,
             device='cuda', plot=False, out_csv='gc3d_timeseries.csv'):
    dev = device_and_precision(device)
    X, Y, Z, KX, KY, KZ, K2 = make_grid(N, L, L, L, dev)
    dx = dy = dz = (L/N)

    # constant alignment
    a = torch.tensor(a_vec, dtype=torch.float32, device=dev)
    a = a / torch.linalg.norm(a)
    ak = a[0]*KX + a[1]*KY + a[2]*KZ
    ak2 = ak*ak

    # initial condition
    u, v, w = init_field(X, Y, Z)

    # readiness field
    if stripe:
        axis = {'x':0, 'y':1, 'z':2}[stripe_axis]
        coord = [X, Y, Z][axis]
        Lax = L
        y0 = stripe_center*Lax
        width = stripe_width*Lax
        sigma = width/2.355
        band = torch.exp(-((coord - y0)**2)/(2*sigma*sigma))
        chi = 1.0 + stripe_strength*band
    else:
        chi = 1.0

    # FFT helpers (torch.fft uses cuFFT on CUDA automatically)
    def fftn3(x):  return torch.fft.fftn(x.to(torch.float32))
    def ifftn3(X): return torch.fft.ifftn(X).real

    # logging
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step","t","KE","GI","An","Rd"])
        writer.writeheader()

        for n in range(steps):
            t = n*dt
            # anneal
            An_t = float(An*math.exp(-t/anneal_tau_An)) if anneal_tau_An else An
            Rd_t = float(Rd*math.exp(-t/anneal_tau_Rd)) if anneal_tau_Rd else Rd

            nu_perp  = 1.0/Re
            nu_aniso = (1.0/Re)*An_t
            Lk = nu_perp*K2 + nu_aniso*ak2
            denom = (1.0 + dt*Lk).to(torch.complex64)

            # Nonlinear advection (real space, centered differences)
            ux = finite_diff_grad(u, dx, 0); uy = finite_diff_grad(u, dy, 1); uz = finite_diff_grad(u, dz, 2)
            vx = finite_diff_grad(v, dx, 0); vy = finite_diff_grad(v, dy, 1); vz = finite_diff_grad(v, dz, 2)
            wx = finite_diff_grad(w, dx, 0); wy = finite_diff_grad(w, dy, 1); wz = finite_diff_grad(w, dz, 2)
            Nu = -(u*ux + v*uy + w*uz)
            Nv = -(u*vx + v*vy + w*vz)
            Nw = -(u*wx + v*wy + w*wz)

            # Readiness drag: -Rd_t * chi * P_perp(a) u
            adot = a[0]*u + a[1]*v + a[2]*w
            upx = u - a[0]*adot
            upy = v - a[1]*adot
            upz = w - a[2]*adot
            if isinstance(chi, torch.Tensor):
                Ru, Rv, Rw = -Rd_t*chi*upx, -Rd_t*chi*upy, -Rd_t*chi*upz
            else:
                Ru, Rv, Rw = -Rd_t*upx, -Rd_t*upy, -Rd_t*upz

            # semi-implicit step in Fourier space
            rhs_u = u + dt*(Nu + Ru)
            rhs_v = v + dt*(Nv + Rv)
            rhs_w = w + dt*(Nw + Rw)

            u_hat = fftn3(rhs_u) / denom
            v_hat = fftn3(rhs_v) / denom
            w_hat = fftn3(rhs_w) / denom

            # projection
            u_hat, v_hat, w_hat = project_incompressible(u_hat, v_hat, w_hat, KX, KY, KZ, K2)

            # back to real
            u = ifftn3(u_hat)
            v = ifftn3(v_hat)
            w = ifftn3(w_hat)

            # diagnostics
            KE = kinetic_energy(u, v, w)
            GI = gunslinger_index_3d(u, v, w, a)
            if (n % max(1, steps//10) == 0) or (n == steps-1):
                print(f"[3D-GC] step {n:4d} t={t:6.3f} | KE={KE:8.5f} | GI={GI:6.3f} | An={An_t:5.3f} | Rd={Rd_t:5.3f}")
            writer.writerow({"step":n, "t":t, "KE":KE, "GI":GI, "An":An_t, "Rd":Rd_t})

    # optional quick slice plot
    if plot:
        try:
            import matplotlib.pyplot as plt
            zmid = N//2
            fig, axs = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
            im0 = axs[0].imshow(u[:,:,zmid].cpu(), origin='lower'); axs[0].set_title("u_x @ mid-z"); plt.colorbar(im0, ax=axs[0])
            im1 = axs[1].imshow(v[:,:,zmid].cpu(), origin='lower'); axs[1].set_title("u_y @ mid-z"); plt.colorbar(im1, ax=axs[1])
            im2 = axs[2].imshow(w[:,:,zmid].cpu(), origin='lower'); axs[2].set_title("u_z @ mid-z"); plt.colorbar(im2, ax=axs[2])
            fig.savefig("gc3d_slice.png", dpi=140); plt.close(fig)
            print("Saved gc3d_slice.png")
        except Exception as e:
            print("Plot skipped:", e)

def main():
    ap = argparse.ArgumentParser(description="3D Gunslinger Continuum (PyTorch/CUDA)")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--L", type=float, default=2*math.pi)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--Re", type=float, default=200.0)
    ap.add_argument("--An", type=float, default=2.0)
    ap.add_argument("--Rd", type=float, default=3.0)
    ap.add_argument("--angle", type=float, default=0.0, help="azimuth in degrees")
    ap.add_argument("--elev", type=float, default=0.0, help="elevation in degrees (0 in-plane, 90 up)")
    ap.add_argument("--anneal_tau_An", type=float, default=None)
    ap.add_argument("--anneal_tau_Rd", type=float, default=None)
    ap.add_argument("--stripe", action="store_true")
    ap.add_argument("--stripe_axis", choices=["x","y","z"], default="z")
    ap.add_argument("--stripe_strength", type=float, default=2.0)
    ap.add_argument("--stripe_center", type=float, default=0.5)
    ap.add_argument("--stripe_width", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out_csv", type=str, default="gc3d_timeseries.csv")
    args = ap.parse_args()

    # alignment vector from spherical angles
    az = math.radians(args.angle); el = math.radians(args.elev)
    a_vec = (math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el))

    run_gc3d(N=args.N, L=args.L, steps=args.steps, dt=args.dt,
             Re=args.Re, An=args.An, Rd=args.Rd, a_vec=a_vec,
             anneal_tau_An=args.anneal_tau_An, anneal_tau_Rd=args.anneal_tau_Rd,
             stripe=args.stripe, stripe_axis=args.stripe_axis,
             stripe_strength=args.stripe_strength, stripe_center=args.stripe_center, stripe_width=args.stripe_width,
             device=args.device, plot=args.plot, out_csv=args.out_csv)

if __name__ == "__main__":
    main()
