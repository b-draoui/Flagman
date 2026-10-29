# Multiclass intrusion detection on UNSW-NB15: benchmark and robustness study

CS395 Machine Learning final project, Mediterranean Institute of Technology, Fall 2026.

Ten-class classification of network flows (Normal and nine attack categories) on the official UNSW-NB15 partition. Five model families are compared under one protocol, then tested for robustness to feature-group ablation, noise, missing values and attack types unseen during training.

## Data

UNSW-NB15, official partition files `UNSW_NB15_training-set.csv` (175,341 records) and `UNSW_NB15_testing-set.csv` (82,332 records), from the Australian Centre for Cyber Security, UNSW Canberra: https://research.unsw.edu.au/projects/unsw-nb15-dataset (Kaggle mirror: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15). Free for academic research use; the dataset page lists the papers to cite.

The CSV files are not included in this repository. Place both files in `data/raw/`.

## Setup

Python 3.12 or later.

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (macOS / Linux)
pip install -r requirements.txt
```

## Reproducing the results

Run the notebooks in order, from the `notebooks/` folder. Each notebook writes its tables and figures to `results/`, prefixed with its number.

| Notebook | Content | Runtime |
|---|---|---|
| `01_data_exploration` | Class balance, leakage checks, duplicates, distributions, correlations, categorical features | about 15 s |
| `02_model_selection` | Grid search per model, preprocessing and imbalance-handling checks | about 30 min |
| `03_benchmark` | 10-fold cross-validation, final test evaluation, calibration, duplicate sensitivity | about 75 min |
| `04_statistical_tests` | Significance tests and the benchmark summary table | a few seconds |
| `05_robustness` | Feature-group ablation, noise, missing values, leave-one-attack-out | about 100 min |

Runtimes were measured on a 2-core machine. `02` must run before `03` and `05`; `03` must run before `04` and `05`.

## Protocol

- Inputs: 42 features (39 numeric, 3 categorical). `id` and `label` are excluded.
- Exact duplicate training records (same features and class) are removed; the official test set is used unchanged.
- Preprocessing is fitted on training data only: one-hot encoding for categorical features (categories with fewer than 10 records grouped); log(1 + x) and standardisation for logistic regression, k-NN and Naive Bayes.
- Class imbalance: balanced class or sample weights for every model except k-NN and the majority-class baseline.
- Hyperparameters: grid search on a stratified 80/20 split of the training set, selected by macro-F1.
- Benchmark: stratified 10-fold cross-validation with identical folds for every model, then a single evaluation of each model refitted on the whole training set on the official test set.
- Significance: Wilcoxon signed-rank test on paired fold scores with Holm correction, Cliff's delta, and the Nadeau-Bengio corrected resampled t-test.

## Seed policy

All randomness uses the seed defined in `src/data.py` (`SEED = 42`): cross-validation folds, the tuning split, model seeds, SMOTE, bootstrap resampling, and perturbations (`SEED + repetition`).

## Repository structure

```
src/
  data.py            loading, column definitions, deduplication
  preprocessing.py   preprocessing pipeline
  models.py          model definitions and hyperparameter grids
  evaluation.py      metrics, calibration, bootstrap
  stats.py           significance tests and effect sizes
  plotting.py        figure style
notebooks/           01 to 05
results/             tables (.csv) and figures (.png)
```
