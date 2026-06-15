# Probe summary

## Paper-matched reduced protocol check

Using the trapped-ion parameters quoted around Fig. 2:

- initial thermal occupation nbar0 = 10
- cooling rate = 1
- motional heating rate = 1e-3
- spin dephasing rate = 1e-2

Best grid point for the simplified objective QFI/(cool_time + cat_time):

| cool_time | cat_time | QFI | QFI / total prep time | post-cooling nbar |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4.9 | 267.479 | 54.5875 | 10 |

Interpretation: the paper-level reduced model gives an optimum near zero cooling and finite cat-generation time for these parameters.

## Dual-witness ECD branch-formula check

The branch formula is

    F_Q ~= 16 alpha^2 + 4/(2 nbar + 1)

This probe now checks that formula with two witnesses:

1. exact mixed-state spectral QFI in a finite Fock cutoff;
2. finite-displacement Bures response, converted to a local QFI estimate by
   `8 * (1 - sqrt(fidelity(rho, rho_delta))) / delta^2`.

The control parameter is eta = alpha^2/(2 nbar + 1). Smaller eta means stronger branch overlap / weaker validity of the branch-pseudospin approximation.

| eta | dual pass fraction at 10pct tolerance | max spectral rel. error | max Bures-response rel. error | max internal witness gap |
| ---: | ---: | ---: | ---: | ---: |
| 0.125 | 0.75 | 0.179294 | 0.18212 | 0.00344254 |
| 0.25 | 1 | 0.0953623 | 0.0986857 | 0.00367373 |
| 0.5 | 1 | 0.0159877 | 0.0182003 | 0.00224848 |
| 1 | 1 | 0.0032611 | 0.00318379 | 0.00252983 |
| 2 | 1 | 0.00936315 | 0.00902648 | 0.00208774 |

The full row-by-row output is in `dual_witness_ecd_sweep.csv`.

## Minimal claim boundary

This is not a falsification test of hot-state sensing. It is a sufficiency check for the reduced ECD branch formula.

A strong result is when both witnesses agree with the branch formula and with each other. A boundary result is when the branch formula looks acceptable under one witness but not the other, or when both fail in the low-eta branch-overlap regime.
