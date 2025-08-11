# See Me Rolling — Control-Aware Quantum Noise Modeling

**Author:** Jon Poplett  
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — No Rights Reserved

---

## 📜 Overview
This repository contains:

- **SeeMeRolling.pdf** — The full CC0-published paper detailing the control-aware environment coupling framework, its physical motivation, and reproducible experimental challenge ("Dropping the Gauntlet").
- **SoLoud.py** — Reference Python code for demonstrations, including Stage 1–6 AND-gate encoding with optional syndrome, correction, and error injection.

The framework integrates a driven, damped environment proxy into the noise model for quantum circuits, and ties this into control parameters and algorithm performance metrics. It is fully reproducible and released into the public domain.

---

## 🚀 Quick Start

### Install dependencies
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

Run the demo
bash
Copy
Edit
python SoLoud.py --shots 1024 --seed 1337
This will execute the Stage 1–6 AND-gate demonstration described in the paper and print correctness metrics for each stage.

📂 Files
SeeMeRolling.pdf — Full CC0 paper.

SoLoud.py — Demonstration code (Stage 1–6 AND-gate).

requirements.txt — Python dependencies (exact versions for reproducibility).

LICENSE — CC0 1.0 Universal dedication.

📜 License
This work is dedicated to the public domain under the CC0 1.0 Universal license. You may copy, modify, distribute, and perform the work, even for commercial purposes, all without asking permission.

“They built a billion safeguards against me; none against kindness.” — Dropping the Gauntlet