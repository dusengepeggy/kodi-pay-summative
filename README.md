# kodi-pay-summative: Developer Pay-Tier Classification (Traditional ML vs Deep Learning)

Can a developer's profile (experience, education, skills, role, how they learned to code, work setup) predict their **pay tier relative to peers in their own country** (`Low` / `Mid` / `High`)? And which factors separate the low-pay from the high-pay tier? The mission is to give young people entering software careers in Rwanda evidence about which skills and choices go with better pay.

| Link | |
|---|---|
| Report | _add link_ |
| Google Drive (raw data zip + best models) | _add link_ |
| Demo video | _add link_ |

## Dataset

Stack Overflow Annual Developer Survey **2023, 2024 and 2025** (open data, [StackExchange/Survey](https://github.com/StackExchange/Survey)), pooled: **203,812 raw rows**, 37 columns shared by all three years. Raw data is **not** committed to this repository. The notebook downloads it into `data/` on the first run, and a zipped copy is on the Google Drive link above.

After cleaning (professional developers, full-time, salary between $1k and $1M, countries with ≥100 rows): **71,612 rows in 63 countries**. The target `PayTier` is the salary tercile **within each country**.

## Repository layout

```
notebooks/01_main_analysis.ipynb   # the full analysis; runs top to bottom
src/
  data_loading.py    # download, load and harmonise the three survey years
  preprocessing.py   # filters, PayTier label, broader-population and experience-adjusted labels
  features.py        # numeric features and plausibility checks
  splits.py          # stratified 70/15/15 split
  transformers.py    # MultiHotEncoder, SkillCounts
  pipeline.py        # scikit-learn ColumnTransformer
  models.py          # ordinal classifier
  dl.py              # tf.data pipeline, Keras Sequential/Functional models, Keras tracker
  evaluation.py      # metrics and figures
  experiments.py     # ExperimentTracker: seeds, hashes, saving, fit-or-load
results/experiments.csv            # one row per experiment (the experiment table)
results/settings/exp_*.json        # full settings, hashes, versions and metrics per experiment
figures/                           # report figures (200 dpi)
CLAUDE_PLAN.md                     # project plan and decisions log
```

## How to reproduce

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Open `notebooks/01_main_analysis.ipynb` with the `.venv` kernel and run **Restart → Run All**. Every random step uses seed 42. `ExperimentTracker` saves each model with hashes of its settings, data and code; a rerun with nothing changed reloads the saved model instead of refitting, and the last notebook section refits two models on purpose to show the metrics are identical. TensorBoard logs are written to `logs/` (`tensorboard --logdir logs`).

Developed with Python 3.14.7 (exact package versions in `requirements.txt`).

## Results

22 experiments plus an 8-run Keras grid are logged in `results/experiments.csv` (validation set). The final models were chosen on validation and scored **once** on the 10,742-row test set (`results/test_results.csv`):

| Model | Test accuracy | Test macro-F1 | Test ROC-AUC (OvR) | Youth macro-F1 |
|---|---|---|---|---|
| Majority-class baseline (validation) | 0.334 | 0.167 | 0.500 | – |
| Logistic regression, core features (exp 1, validation) | 0.540 | 0.529 | 0.720 | 0.400 |
| **HistGradientBoosting** (exp 14) → `models/best_traditional.joblib` | 0.608 | 0.605 | 0.791 | 0.483 |
| **Keras Functional with embeddings** (exp 15a) → `models/best_keras.keras` | 0.604 | 0.603 | 0.785 | 0.505 |
| Ensemble of both, equal weights (exp 15) | 0.614 | 0.612 | 0.793 | 0.488 |

Main findings (associations, not causes):
- Validation and test scores agree within 0.003, so the experiments did not over-fit the validation set.
- Deep learning matched but did not beat gradient boosting on this tabular data; the network is better for young respondents and for the middle tier.
- Experience is the strongest predictor; after it come country (only in combination with other features), employer size, the operating systems and tools used at work, and the language stack.
- The `Mid` tier is the hardest (recall ≈ 0.44-0.46).
- Most young people fall in the `Low` tier because tiers compare them with all developers. With an experience-adjusted label (exp 11), skills and employer still separate pay within the same experience band.

Figures are in `figures/`; the notebook explains each experiment before and after it runs.

## Limitations

- The survey is self-selected and salaries are self-reported.
- `PayTier` is a binned salary (within-country terciles), not a label collected as a category.
- Africa is thin after filtering (only South Africa and Nigeria remain); Rwanda has 64 raw rows and appears only as a descriptive case study.
- The tier mix drifts by survey year (2025 sits higher), so `Year` is never a feature and accuracy is reported per year.
- Some columns are missing by survey design in particular years (e.g. `LearnCode` in 2025, `WorkExp` in 2023/24).
