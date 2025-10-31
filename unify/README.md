# GIE Bridge: Bridging Quantum and Classical Gravity via Synchronization

A computational framework exploring compatibility between competing theories of gravitationally-induced entanglement (GIE).

## Overview

This repository implements a diagnostic framework that treats quantum gravity witnesses (Marletto-Vedral/Bose et al.) and classical-gravity-with-QFT-matter claims (Aziz-Howl) as potentially compatible mechanisms in different parameter regimes, rather than mutually exclusive theories.

The key innovation is an **adaptive synchronization bridge** that uses spectral signatures (quantum doublets) to interpolate between the two regimes, suggesting both effects can coexist in a unified parameter space.

## Motivation

Recent debates in quantum gravity have centered on whether entanglement between masses can definitively prove gravity is quantum:
- **Marletto-Vedral/Bose (2017)**: If gravity mediates entanglement, it must be quantum
- **Aziz-Howl (2025)**: Classical gravity with QFT matter can also generate entanglement via virtual matter propagators
- **Community response**: Heated debate about locality, nonlocality, and validity of claims

This framework explores a third option: **both mechanisms might be correct in different parameter regimes**.

## Repository Structure
```
├── mv_bose_qg_model.py      # Model A: Quantum mediator implementation
├── ah_cg_qft_model.py       # Model B: Classical gravity + QFT matter
├── unify_trace.py           # Unified framework with synchronization bridge
├── unify_trace.pdf          # Technical documentation and results
└── README.md               
```

## Key Features

### Model A: Quantum Gravity (Marletto-Vedral/Bose)
- Spin-dependent spatial superposition via Stern-Gerlach
- Gravitational phase accumulation: φ = GMm τ/(ℏd)
- Entanglement witness via negativity/concurrence
- Paper-faithful parameters: m = 10⁻¹⁴ kg, τ = 2.5 s, d = 450 μm

### Model B: Classical Gravity + QFT (Aziz-Howl)
- Fourth-order virtual matter propagators
- Branch-dependent amplitudes: ϑ ∝ Gm M² R² t/(ℏ d³)
- Paper-faithful preset: M = 15 kg, t = 1 s
- **Note**: Used as parametric surrogate; nonlocality concerns acknowledged

### Model C: Synchronization Bridge
- Quantum dimer: coupled driven-dissipative oscillators
- Spectral doublet from dressed normal modes
- Doublet confidence Dc ∈ [0,1] drives adaptive coupling
- Interpolation: λ = (1-Dc)·Na/(Na+Nb) + Dc·0.5

## Usage

### Quick Start
```bash
# Run unified framework with paper defaults
python unify_trace.py --mode paper

# Custom parameters
python unify_trace.py --mode quick \
  --mv_m 1e-14 --mv_tau 2.5 \
  --ah_M 15.0 --ah_t 1.0 \
  --c_g 0.42 --c_Trun 180
```

### Individual Models
```python
# Model A only
python mv_bose_qg_model.py --mode paper

# Model B only  
python ah_cg_qft_model.py --mode paper
```

### Parameter Sweeps
```bash
# Time evolution trace
python unify_trace.py --mode trace \
  --trace_tmin 0.1 --trace_tmax 3.0 --trace_n 50

# Mass-time grid (Model B)
python ah_cg_qft_model.py --mode grid \
  --grid_Mmin 1e-12 --grid_Mmax 1e-6
```

## Key Results

From paper defaults:
- **Model A negativity**: Na = 0.0389 (quantum mediator dominant at small masses)
- **Model B negativity**: Nb = 0.0019 (virtual matter effects at large masses)  
- **Doublet confidence**: Dc = 0.599 (strong spectral signature)
- **Bridge weight**: λ = 0.682 (A-lean, dressed-state regime)
- **Spectral split**: 0.121 Hz (matches theory within 1.3%)

## Technical Details

### Critical Implementation Points
1. **QRT Spectrum**: Use time-evolved steady state, not eigenvector
2. **DC Removal**: Subtract mean from correlation function
3. **Two-port sum**: Aggregate both leakage channels
4. **Asymmetric drive**: Breaks degeneracy (Ω₁ ≠ Ω₂)
5. **Pure dephasing**: Keeps lines finite (γφ > 0)

### Regime Classification
- **Quantum-mediator dominant**: Na ≫ Nb, Dc ≈ 0
- **Classical+QFT dominant**: Nb ≫ Na, Dc ≈ 0
- **Bridge (dressed-state)**: Dc > 0.25, coherent hybridization

## Caveats and Limitations

1. **Model B controversy**: Nonlocality concerns remain debated; we use it as a parametric surrogate
2. **Not adjudicating quantum gravity**: This framework explores compatibility, not fundamental truth
3. **Finite truncation**: Fock space N=5 for computational efficiency
4. **Detection thresholds**: Peak finding parameters affect Dc values

## Citation

If you use this code, please cite:
```
EchoKey Team (2025). "Bridging Quantum and Classical Gravity via Synchronization: 
A Computational Diagnostic." GitHub: https://github.com/JGPTech/Fun/edit/main/unify
```

## Related Work

- Marletto & Vedral, PRL 119, 240402 (2017)
- Bose et al., PRL 119, 240401 (2017)
- Aziz & Howl, Nature 646, 813 (2025)
- Marletto & Vedral rebuttal, arXiv:2510.19969 (2025)

## License

cc0 This code is provided as-is for research purposes. Feel free to use, modify, and distribute with attribution.

