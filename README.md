# VNAL-PA Framework — IEEE Quantum Week 2026

Variational Quantum Algorithm for Hyperspectral Band Selection using Parameterized Ansatz (VNAL-PA).

## Structure
```
src/
  cabbage/     # CABBAGE dataset experiments
  eggplant/    # EGGPLANT dataset experiments
results/       # Numerical results (JSON/CSV)
figures/       # Publication figures (PNG/PDF)
paper/         # Extended abstract & poster
data/          # Raw data (gitignored if large)
```

## Requirements
```bash
pip install -r requirements.txt
```

## Run
```bash
# CABBAGE dataset
python src/cabbage/01_compute_matrices.py
python src/cabbage/02_vnal_pa.py
python src/cabbage/03_baselines.py
python src/cabbage/04_evaluate_table1.py

# EGGPLANT dataset
python src/eggplant/01_compute_matrices.py
python src/eggplant/02_vnal_pa.py
python src/eggplant/03_baselines.py
python src/eggplant/04_evaluate_table1.py
```

## Results
Key results reproduced in `results/table1_results.csv` and `results/all_methods_bands.json`.

## Citation
```bibtex
@inproceedings{le2026vnalpa,
  title={VNAL-PA: Variational Quantum Algorithm for Hyperspectral Band Selection},
  author={Le, Huu Phuc and ...},
  booktitle={IEEE Quantum Computing and Engineering (QCE)},
  year={2026}
}
```