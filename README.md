# FL-DP-IDS-5GC

Code and results for "Privacy-Preserving Intrusion Detection for 5G Core: A Federated Learning and Differential Privacy Approach" (IEEE ICC 2027).
Federated learning IDS with differential privacy for the 5G core, evaluated on CVE5G and 5G-NIDD.

## Data
- CVE5G:
- 5G-NIDD: https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless (use `Combined.csv`)

Datasets are not included in this repository.

## Setup
```bash
pip install -r requirements.txt
```

## Reproducing the results
All scripts are in `src/` and must stay in the same folder.

| Paper item | Command | Output |
|---|---|---|
| Fig. 1–3, CVE5G (MLP) | `python 05_ids_flow_fl_dp_v3.py ...` | `results_05_mlp.csv` |
| Fig. 1–3, CVE5G (1D-CNN) | `python 05_ids_flow_fl_dp_v3.py --model cnn ...` | `results_05_novo_cnn.csv` |
| Fig. 1–3, 5G-NIDD | `python 07_ids_5gnidd_fl_dp.py --csv Combined.csv ...` | `results_07.csv` |
| Table I, CVE5G (MLP) | `python 08_mia_audit.py --dataset cve5g --dup-mode always-in --overfit ...` | `results_08_mia_cve5g_alwaysin.csv` |
| ... | ... | ... |

## Tables and figures
`notebooks/tables.ipynb` and `notebooks/figures.ipynb` read the CSVs in `results/` and generate the paper's tables (LaTeX) and figures (PDF).
