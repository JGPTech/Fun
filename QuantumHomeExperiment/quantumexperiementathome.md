# Plasma Display Quantum Erasure Experiment
## A Blueprint for Physical Quantum Mechanics at Home

**Author:** JGPTech  
**License:** CC0  
**Status:** Skeleton / Working Draft — Placeholders marked with `[PLACEHOLDER]`

---

## Overview

This document is a blueprint for a genuinely physical quantum erasure experiment using a consumer plasma display as the experimental substrate. The goal is not simulation, not visualization, not a cartoon of quantum mechanics — but an actual physical system in which quantum phenomena emerge from the real behavior of plasma discharge cells.

The visual output on the screen is not rendered math. It is an emergent property of the physics happening inside the cells.

A motivated experimenter with physics background and electronics skills should be able to pick this up and run with it.

---

## 1. Conceptual Premise

### 1.1 Why Plasma — Not Optics, Not Simulation

Most home quantum experiments fall into one of two traps:

1. **Classical wave optics dressed as quantum mechanics** — laser double-slit experiments demonstrate wave interference, which is a classical phenomenon. The math looks the same, but no quantum superposition is being probed.

2. **Software visualization** — rendering interference patterns computationally produces a correct image of what the physics *would* look like, but nothing quantum is actually happening.

This experiment takes a third path: using the physical properties of plasma display cells as the actual quantum substrate. The discharge events in each cell are real atomic transitions. The UV photons emitted are real quantum events. The phosphor excitation is a real energy level transition. Nothing is simulated.

### 1.2 The Core Idea

A plasma display is a large, cheap, pixel-perfect physical matrix of individually addressable quantum emitters. Each cell is a gas discharge tube at millimeter scale. By controlling which cells fire, when they fire, and in what phase relationship to each other, we define the geometry and coherence properties of a real quantum optical system.

The which-path information in our quantum erasure experiment is not a software flag. It is physically encoded in the activation state of specific cells. Erasure is not a software reset. It is the physical deactivation of those cells, destroying the which-path information in the real world.

The interference pattern — if achieved — is not rendered. It emerges from the actual physics of coherent photon emission across the cell matrix.

### 1.3 What Makes This Novel

No prior experiment has used a plasma display as a controllable coherent photon source matrix for quantum optics. The combination of:

- Individual cell addressability at scale (921,600 pixels in 720p)
- Real atomic emission (not LEDs, not lasers — gas discharge)
- Hardware-level timing control potential
- Low cost ($50 used plasma TV)

...creates a unique experimental platform that has not been explored.

---

## 2. Background Physics

### 2.1 Quantum Erasure — The Core Phenomenon

Quantum erasure demonstrates the relationship between which-path information and quantum interference.

**Standard setup:**

A particle (typically a photon) passes through a double slit. With both slits open and no which-path information available anywhere in the system, the particle interferes with itself and produces an interference pattern on a detector screen.

**Step 1 — Introduce which-path information:**

Tag each slit so that in principle the path of the particle could be determined. The interference pattern disappears. Not because anyone looked — but because the which-path information now exists as a physical degree of freedom entangled with the particle state. The two path amplitudes are no longer coherent.

**Step 2 — Erase the which-path information:**

Destroy the which-path tag before detection. The interference pattern returns.

**The delayed-choice variant:**

The erasure decision can be made *after* the particle is already detected. The interference pattern only reconstructs when the erased subset of detections is post-selected. This appears to show the present outcome depending on a future choice — though the resolution lies in understanding that the pattern is only visible in coincidence counting, not in the raw detector data.

**The key point:**

This is not about consciousness, observation, or measurement in the everyday sense. It is about whether distinguishing information is physically available anywhere in the system. The math enforces it.

### 2.2 Path Integral Formulation

The probability amplitude for a photon to travel from source S to detector D is:

$$\Psi(D) = \sum_{\text{all paths}} A_i \, e^{i S_i / \hbar}$$

Where $S_i$ is the classical action along path $i$ and $A_i$ is the amplitude weight.

For a two-slit geometry with slits at positions $y_1$ and $y_2$, slit separation $d$, screen distance $L$, and wavelength $\lambda$:

$$\Psi(x) = A \left( e^{i k r_1} + e^{i k r_2} \right)$$

Where $r_1$, $r_2$ are the path lengths from each slit to detector position $x$, and $k = 2\pi/\lambda$.

The intensity (probability of detection) is:

$$I(x) = |\Psi(x)|^2 = 2A^2 \left(1 + \cos\left(\frac{2\pi d \, x}{\lambda L}\right)\right)$$

Bright fringes appear where the path length difference $\Delta r = r_1 - r_2$ satisfies:

$$\Delta r = n\lambda, \quad n = 0, \pm 1, \pm 2, \ldots$$

**When which-path information is present:**

The two path amplitudes become orthogonal in the extended Hilbert space (particle + which-path marker). The cross term in $|\Psi|^2$ vanishes:

$$I(x) = |A_1|^2 + |A_2|^2 = 2A^2 \quad \text{(flat, no fringes)}$$

**When which-path information is erased:**

Coherence is restored. The cross term returns. Fringes reappear.

### 2.3 Plasma Cell Physics

Each plasma display cell contains a xenon-neon gas mixture at low pressure. When a sufficient voltage is applied across the electrodes:

1. The gas ionizes — free electrons accelerate toward the anode
2. Electron collisions excite xenon atoms to higher energy levels
3. Xenon relaxes via UV photon emission at approximately **147 nm** (resonance line) and **173 nm** (excimer band)
4. UV photons strike phosphor coatings on the cell walls
5. Phosphors emit visible light (red, green, or blue depending on cell)

Each discharge event is a cascade of real atomic transitions. The UV emission wavelengths are fixed by the quantum energy levels of xenon — not by the display electronics.

**Greyscale via PWM:**

Plasma displays achieve greyscale through pulse-width modulation — rapid, discrete firing cycles within each frame period. Intensity is controlled by the number of sub-field pulses, not by varying the discharge voltage. This means the cell is either fully on or fully off at any given moment — a binary quantum emitter.

**Zero-energy state:**

A non-firing cell (black pixel) is in its ground state. No discharge, no UV, no photon emission. This is the experimental baseline — the "closed slit" state.

---

## 3. The Experimental Architecture

### 3.1 The Plasma Screen as Physical Matrix

A 720p plasma display provides:

- **1,280 × 720 = 921,600** individually addressable pixels
- **~2.76 million** individual subpixel cells (RGB)
- Physical cell pitch: approximately **0.5–0.8 mm** depending on screen size
- True individual cell addressability via crossed electrode matrix

The screen is simultaneously:
- The slit apparatus (cell activation pattern defines geometry)
- The coherent source array (cells are the emitters)
- The display (emergent pattern is visible directly)

### 3.2 Experiment States

**State 0 — Both slits open, no which-path marker:**

Two columns of cells activated, separated by distance $d$. No tagging. If coherence conditions are met, interference pattern emerges across the detector region.

**State 1 — Which-path information present:**

Marker cells activated adjacent to each slit column. Each slit is now tagged with distinguishable information (e.g., orthogonal polarization, different wavelength phosphor, timing offset). Interference pattern collapses.

**State 2 — Which-path information erased:**

Marker cells deactivated. Distinguishing information destroyed. Interference pattern returns.

The experimenter cycles between states and observes the screen directly.

### 3.3 The Coherence Problem — The Central Challenge

**This is the hardest unsolved part of the design.**

Standard plasma display operation drives cells incoherently — each discharge is independent, with no phase relationship to neighboring cells. Incoherent sources do not produce stable interference patterns (they produce speckle that time-averages to a flat intensity distribution).

To achieve genuine quantum interference, phase coherence must be established between the emitting cells. This requires one of the following approaches:

**[PLACEHOLDER 3.3.A — Coherence Strategy 1: Timing Control]**

Drive adjacent cells with precisely controlled timing offsets at the discharge level. If two cells fire with a fixed phase relationship, their emitted photons are coherent. This requires bypassing the display controller entirely and driving address/sustain electrodes directly with precision hardware.

*Requirements to be determined:*
- Minimum timing precision required (sub-nanosecond? picosecond?)
- Whether commercial plasma panel electrode access is feasible without destroying the panel
- FPGA or equivalent hardware specifications
- Whether discharge jitter is small enough to maintain coherence across multiple firings

**[PLACEHOLDER 3.3.B — Coherence Strategy 2: Spatial Filtering + Monochromator]**

Accept incoherent emission from cells but pass the output through a spatial filter and monochromator to select a coherent subset of the emitted light. The cells become a controllable incoherent source; coherence is imposed optically downstream.

*Requirements to be determined:*
- Optical path design (spatial filter pinhole size, monochromator specs)
- Whether sufficient photon flux survives filtering for detection
- What detector is appropriate (SPAD array, CCD, human-visible?)
- Whether this approach still counts as "emergent" or crosses into classical optics territory

**[PLACEHOLDER 3.3.C — Coherence Strategy 3: Stimulated Emission Seeding]**

[To be investigated] — whether a seed laser could be used to stimulate coherent emission from the plasma cells, effectively locking their phase to an external reference.

*Requirements to be determined:*
- Feasibility assessment
- Whether UV seed at 147/173 nm is practical at hobbyist scale
- Safety considerations

---

## 4. Hardware Requirements

### 4.1 The Display Panel

- **Target:** Used 720p plasma television, 42"–50" preferred
- **Why 720p not 1080p:** Lower pixel density = larger cell pitch = easier electrode access
- **Recommended brands for panel accessibility:** [PLACEHOLDER 4.1.A — survey of panel disassembly documentation, Pioneer/Panasonic/Samsung plasma internals]
- **Estimated cost:** $30–$100 used market

### 4.2 Display Controller Bypass

**[PLACEHOLDER 4.2 — Hardware Control Layer]**

To achieve coherent cell firing, the standard display controller (which accepts HDMI/VGA and drives the panel via proprietary timing) must either be:

a) Bypassed entirely — electrodes driven directly
b) Replaced with a custom controller
c) Intercepted at the panel interface connector level

*Requirements to be determined:*
- Panel interface connector type and pinout (varies by manufacturer)
- Address electrode vs sustain electrode vs scan electrode roles and timing
- Minimum hardware to drive a subset of cells directly
- FPGA board recommendations (Xilinx, Altera, iCE40 family)
- Safety considerations (plasma panels operate at 150–200V sustain voltages)

### 4.3 Detection System

**[PLACEHOLDER 4.3 — Detector Specification]**

Depending on which coherence strategy is pursued:

- **Direct visual:** If pattern is bright enough, human eye + camera
- **CCD/CMOS camera:** Standard approach, microsecond integration
- **SPAD array:** Single-photon avalanche detectors for genuine single-photon statistics
- **Coincidence counter:** Required for delayed-choice / entanglement variants

*Requirements to be determined:*
- Minimum detectable fringe contrast
- Required spatial resolution at detector plane
- Whether single-photon sensitivity is necessary for this experiment

### 4.4 Soft Control Layer (Python)

For the software-addressable aspects of the experiment (cell pattern definition, state switching, data logging):

```python
# Skeleton — to be developed
# Controls which cells are active via display output
# OR via hardware interface if controller bypass is implemented

import numpy as np

# Experiment parameters
SLIT_WIDTH_CELLS = 4        # width of each slit in pixels
SLIT_SEPARATION_CELLS = 40  # center-to-center separation
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# Experiment states
STATE_BOTH_SLITS     = 0  # open slits, no which-path marker
STATE_WHICH_PATH     = 1  # which-path markers active
STATE_ERASED         = 2  # markers destroyed

def build_pattern(state, slit_width, slit_sep):
    """
    Returns pixel activation matrix for given experiment state.
    [PLACEHOLDER — implement full pattern logic]
    """
    pattern = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))
    # ... slit geometry
    # ... marker cell logic per state
    return pattern

def expected_intensity(x, d, wavelength, L):
    """
    Theoretical interference pattern for validation.
    I(x) = 2A^2 * (1 + cos(2*pi*d*x / (wavelength * L)))
    """
    return 2 * (1 + np.cos(2 * np.pi * d * x / (wavelength * L)))
```

---

## 5. The Experiment Sequence

### 5.1 Baseline Characterization

**[PLACEHOLDER 5.1]**

Before running the quantum erasure sequence, characterize the panel:

- Single cell emission spectrum (spectrometer measurement)
- Cell-to-cell intensity uniformity
- Discharge timing jitter (oscilloscope measurement at electrode level)
- Coherence length of emission under standard drive conditions

### 5.2 Coherence Verification

**[PLACEHOLDER 5.2]**

Establish that the coherence strategy chosen in Section 3.3 is actually producing coherent emission before attempting the full erasure experiment. Proposed tests:

- Two-cell interference fringe visibility as a function of cell separation
- Fringe stability over time
- Dependence on drive timing precision

### 5.3 Main Experiment

1. Configure panel in **State 0** (both slits, no marker)
2. Record intensity distribution at detector
3. Verify interference fringe pattern — compare to theoretical $I(x)$
4. Switch to **State 1** (which-path markers active)
5. Record intensity distribution — verify fringe collapse
6. Switch to **State 2** (markers erased)
7. Record intensity distribution — verify fringe restoration
8. Repeat with varied slit parameters ($d$, slit width)

### 5.4 Delayed-Choice Variant

**[PLACEHOLDER 5.4]**

Extension to the delayed-choice quantum eraser:

- Requires entangled photon pairs — one photon detected at screen, partner photon routed to erasure apparatus
- Erasure decision made after screen detection
- Interference visible only in coincidence-selected subset

*Requirements to be determined:*
- Whether plasma emission can be adapted for entangled pair generation (likely requires BBO crystal downstream)
- Full optical path design
- Coincidence counting electronics

---

## 6. What Success Looks Like

A successful experiment produces:

1. **In State 0:** A spatially periodic intensity distribution matching $I(x) = 2A^2(1 + \cos(2\pi d x / \lambda L))$ with fringe visibility $V = (I_{max} - I_{min})/(I_{max} + I_{min}) > 0$ — ideally $V \approx 1$ for full coherence.

2. **In State 1:** Flat intensity distribution, $V \approx 0$.

3. **In State 2:** Fringe pattern restored, $V$ returning toward State 0 value.

The key distinguisher from a classical result: the collapse in State 1 must be caused by the *existence* of the which-path information, not by any physical disturbance to the photon path. If State 1 disturbs the geometry and State 2 restores it, that is a classical result. The which-path marker must be a non-disturbing tag.

**[PLACEHOLDER 6.1 — Statistical Analysis Protocol]**

Define required sample sizes, fringe contrast thresholds, and statistical tests to distinguish genuine quantum erasure from classical artifacts.

---

## 7. Safety Notes

- Plasma panels operate at **150–200V sustain voltages** on electrode lines. Direct electrode access requires appropriate isolation and safety precautions.
- UV emission at 147/173 nm is present inside the panel. Standard glass front panel blocks this — do not remove the front glass while operating.
- High voltage capacitors may retain charge after power-off. Follow standard discharge procedures before working on panel internals.

**[PLACEHOLDER 7.1 — Full safety protocol for electrode-level access]**

---

## 8. Open Questions and Future Directions

1. Can plasma cell discharge timing be controlled precisely enough to establish inter-cell coherence? What is the fundamental jitter floor?
2. Is spatial filtering + monochromator sufficient to recover coherence from incoherent plasma emission at usable flux levels?
3. Can this platform be extended to a genuine Bell inequality violation test?
4. What is the minimum viable version of this experiment — the simplest configuration that demonstrates something irreducibly quantum?
5. Can the pixel grid topology be used to engineer non-trivial path geometries (multi-slit, curved slits, variable slit arrays) that would be difficult to implement mechanically?

---

## References and Prior Art

**[PLACEHOLDER — Literature review]**

- Kim et al. (1999) — delayed-choice quantum eraser (canonical experiment)
- Aspect et al. (1982) — Bell inequality violation
- Feynman, Leighton, Sands — *The Feynman Lectures on Physics*, Vol. III (path integral introduction)
- [Plasma display panel technical documentation — manufacturer service manuals]
- [FPGA-based plasma panel controller prior work — to be surveyed]

---

*This blueprint is released under CC0. No rights reserved. Take it, run with it, break it, improve it.*

*— JGPTech, March 2026*