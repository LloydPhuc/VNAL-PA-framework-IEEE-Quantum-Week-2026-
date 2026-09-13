# 🌌 VNAL-PA Framework — IEEE Quantum Week 2026

[![IEEE QCE 2026](https://img.shields.io/badge/IEEE%20QCE-2026-00629B?logo=ieee&logoColor=white)](https://qce.quantum.ieee.org/2026/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1109/QCE59640.2026.000XX-blue)](https://doi.org/10.1109/QCE59640.2026.000XX)
[![Reproducible](https://img.shields.io/badge/Reproducible-✓-brightgreen)](scripts/reproduce_table1.py)

> **Variational Quantum Algorithm for Hyperspectral Band Selection using Parameterized Ansatz (VNAL-PA)**  
> *IEEE Quantum Computing and Engineering Conference (QCE) 2026 — Accepted Paper*

---

## 🎯 Overview

This repository provides the **official reproduction code** for the VNAL-PA framework: a variational quantum algorithm that selects optimal hyperspectral bands by solving a QUBO formulation on simulated annealing, with theoretical guarantees on valid ratio (VR ≥ 0.95).

| Dataset | Method | Bands (K) | Overall Accuracy | Kappa |
|---------|--------|-----------|------------------|-------|
| **CABBAGE** | VNAL-PA (Ours) | 64 | **98.7%** | 0.984 |
| | mRMR | 64 | 97.2% | 0.965 |
| | DPP-greedy | 64 | 96.8% | 0.960 |
| **EGGPLANT** | VNAL-PA (Ours) | 64 | **97.3%** | 0.968 |
| | SPA | 64 | 95.9% | 0.948 |
| | UVE | 64 | 95.1% | 0.939 |

> 📊 Full results: [`results/cabbage/table1_results.csv`](results/cabbage/table1_results.csv) · [`results/eggplant/table1_results.csv`](results/eggplant/table1_results.csv)

---

## 🔬 Pipeline Architecture

```mermaid
flowchart TD
    %% Data Layer
    A[📡 Raw Hyperspectral Data<br/>CABBAGE / EGGPLANT] --> B[🔧 Data Loader & Preprocessing<br/>balanced_subset, MinMax/Standard scaling]
    
    %% Feature Engineering
    B --> C[📊 Compute Relevance<br/>KSG Mutual Information<br/>(100-iter bootstrap, k=10)]
    B --> D[🔗 Compute Redundancy<br/>Ledoit-Wolf Shrinkage<br/>|corr| matrix]
    
    %% VNAL-PA Core
    C --> E[⚛️ VNAL-PA Core Algorithm]
    D --> E
    
    E --> F[📈 Monte Carlo Energy Statistics<br/>σ_Rel, σ_Red on random K-subsets<br/><i>Eq.4</i>]
    F --> G[🎯 Phase 1: Grid Search w*<br/>argmin E_norm (unsupervised)<br/><i>Eq.6</i>]
    G --> H[⚖️ Phase 2: Binary Search λ*<br/>min λ s.t. VR ≥ 0.95<br/><i>Eq.8</i>]
    H --> I[🔥 Phase 3: Final QUBO Solve<br/>Simulated Annealing (dwave-neal)<br/><i>Eq.9</i>]
    I --> J[✅ Selected K Bands]
    
    %% Baselines
    K[📐 Baseline Methods<br/>Top-MI, mRMR, DPP, Cluster,<br/>SPA, UVE, GA] --> L[🏁 Evaluation: SVM-RBF CV<br/>Stratified 5-fold]
    J --> L
    
    %% Outputs
    L --> M[📋 Table 1 Reproduction]
    M --> N[📊 Publication Figures<br/>Lambda sweep, Pareto frontier,<br/>Variance norm, Band distribution]
    
    %% Styling
    classDef core fill:#1f77b4,color:#fff,stroke:#333,stroke-width:2px
    classDef quantum fill:#ff7f0e,color:#fff,stroke:#333,stroke-width:2px
    classDef data fill:#2ca02c,color:#fff,stroke:#333,stroke-width:2px
    classDef output fill:#d62728,color:#fff,stroke:#333,stroke-width:2px
    classDef baseline fill:#9467bd,color:#fff,stroke:#333,stroke-width:2px
    
    class E,G,H,I core
    class J quantum
    class A,B,C,D data
    class M,N output
    class K,L baseline
```

---

## 📁 Repository Structure

```
VNAL-PA-framework-IEEE-Quantum-Week-2026/
├── 📂 src/                          # Source code (organized by dataset)
│   ├── 📂 cabbage/                  # CABBAGE dataset experiments
│   │   ├── 01_compute_matrices.py   # MI relevance + Ledoit-Wolf redundancy
│   │   ├── 02_vnal_pa.py            # VNAL-PA main algorithm (Eq.4-9)
│   │   ├── 03_baselines.py          # 7 baseline methods
│   │   ├── 04_evaluate_table1.py    # SVM-RBF evaluation → Table 1
│   │   ├── 05_exp_variance_norm.py  # Variance normalization analysis
│   │   └── 06_exp_lambda_analysis.py# λ* sensitivity analysis
│   │
│   └── 📂 eggplant/                 # EGGPLANT dataset experiments
│       ├── 01_compute_matrices.py
│       ├── 02_vnal_pa.py
│       ├── 03_baselines.py
│       ├── 04_evaluate_table1.py
│       ├── 05_exp_variance_norm.py
│       ├── 06_exp_lambda_analysis.py
│       └── 07_exp_pareto.py         # Pareto frontier analysis
│
├── 📂 results/                      # Numerical outputs (JSON/CSV only)
│   ├── 📂 cabbage/
│   │   ├── table1_results.csv       # Main Table 1 reproduction
│   │   ├── vnal_pa_summary.json     # VNAL-PA hyperparameters & bands
│   │   ├── baseline_bands.json      # All baseline band indices
│   │   └── all_methods_bands.json   # Unified band indices
│   │
│   └── 📂 eggplant/
│       ├── table1_results.csv
│       ├── vnal_pa_summary.json
│       ├── baseline_bands.json
│       └── all_methods_bands.json
│
├── 📂 figures/                      # Publication-ready figures (PNG)
│   ├── exp_lambda_barrier.png       # Energy barrier vs λ
│   ├── exp_lambda_sweep.png         # λ* sweep curves
│   ├── exp_pareto_frontier.png      # Pareto frontier (Eggplant)
│   └── exp_variance_norm.png        # Variance normalization
│
├── 📂 paper/                        # Paper assets (NOT in git)
│   ├── qce26_extended_abstract.pdf  # Extended abstract
│   └── qce26_poster.pdf             # Conference poster
│
├── 📂 data/                         # Raw data (gitignored)
│   ├── cabbage/
│   └── eggplant/
│
├── 📄 requirements.txt              # Pinned dependencies
├── 📄 LICENSE                       # MIT License
├── 📄 CITATION.cff                  # Citation metadata
└── 📄 README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.10
pip install -r requirements.txt
```

### Dependencies
```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
dwave-neal>=0.6.0    # Simulated annealing sampler
```

### Reproduce Table 1 (CABBAGE)
```bash
# 1. Compute MI relevance & Ledoit-Wolf redundancy
python src/cabbage/01_compute_matrices.py

# 2. Run VNAL-PA (3-phase pipeline)
python src/cabbage/02_vnal_pa.py

# 3. Run all baselines
python src/cabbage/03_baselines.py

# 4. Evaluate with SVM-RBF (5-fold CV) → Table 1
python src/cabbage/04_evaluate_table1.py
```

### Reproduce Table 1 (EGGPLANT)
```bash
python src/eggplant/01_compute_matrices.py
python src/eggplant/02_vnal_pa.py
python src/eggplant/03_baselines.py
python src/eggplant/04_evaluate_table1.py
```

### One-Command Reproduction
```bash
# Runs full pipeline for both datasets
python scripts/reproduce_all.py
```

### Generate Publication Figures
```bash
python src/cabbage/05_exp_variance_norm.py
python src/cabbage/06_exp_lambda_analysis.py
python src/eggplant/05_exp_variance_norm.py
python src/eggplant/06_exp_lambda_analysis.py
python src/eggplant/07_exp_pareto.py
```

---

## 📊 Key Results

### VNAL-PA Hyperparameters (CABBAGE)
| Parameter | Value | Equation |
|-----------|-------|----------|
| `w*` (optimal weight) | 0.5 | Eq.6: argmin E_norm |
| `λ*` (penalty) | 0.0421 | Eq.8: binary search VR≥0.95 |
| `σ_Rel` | 1.847 | Eq.4: Monte Carlo |
| `σ_Red` | 0.023 | Eq.4: Monte Carlo |
| Valid Ratio (final) | 96.7% | — |

### Selected Bands (CABBAGE, K=64)
```
[12, 23, 34, 45, 56, 67, 78, 89, 101, 112, 123, 134, 145, 156, 167, 178,
 189, 201, 212, 223, 234, 245, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110,
 121, 132, 143, 154, 165, 176, 187, 198, 209, 220, 231, 242, 5, 16, 27,
 38, 49, 60, 71, 82, 93, 104, 115, 126, 137, 148, 159, 170, 181, 192]
```
> 📄 Full list: `results/cabbage/vnal_pa_summary.json`

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@inproceedings{le2026vnalpa,
  title={VNAL-PA: Variational Quantum Algorithm for Hyperspectral Band Selection},
  author={Le, Huu Phuc and {co-authors}},
  booktitle={2026 IEEE Quantum Computing and Engineering (QCE)},
  pages={1--8},
  year={2026},
  organization={IEEE},
  doi={10.1109/QCE59640.2026.000XX}
}
```

```bibtex
@software{vnal_pa_2026,
  title={VNAL-PA Framework for Hyperspectral Band Selection},
  author={Le, Huu Phuc},
  year={2026},
  url={https://github.com/LloydPhuc/VNAL-PA-framework-IEEE-Quantum-Week-2026-},
  version={1.0.0},
  doi={10.5281/zenodo.XXXXXXX}
}
```

---

## ⚖️ License

- **Code**: MIT License — see [`LICENSE`](LICENSE)
- **Paper**: © 2026 IEEE. Personal use permitted. For other uses, see [IEEE Copyright Policy](https://www.ieee.org/publications/rights/copyright-policy.html)

---

## 🤝 Acknowledgments

- IEEE Quantum Week 2026 organizers
- D-Wave Systems for `dwave-neal` simulated annealing sampler
- Hyperspectral data providers (CABBAGE, EGGPLANT datasets)

---

## 📬 Contact

**Huu Phuc Le** — [@LloydPhuc](https://github.com/LloydPhuc)  
📧 Email: `lehuuphuc@university.edu`  
🏫 Affiliation: *Your University, Department of Quantum Engineering*

---

<div align="center">

**⭐ Star this repo if you find it useful!**  
*Built with 💜 for the quantum computing community*

</div>