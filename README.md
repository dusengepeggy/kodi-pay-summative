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

RESULTS_PLACEHOLDER

## Limitations

- The survey is self-selected and salaries are self-reported.
- `PayTier` is a binned salary (within-country terciles), not a label collected as a category.
- Africa is thin after filtering (only South Africa and Nigeria remain); Rwanda has 64 raw rows and appears only as a descriptive case study.
- The tier mix drifts by survey year (2025 sits higher), so `Year` is never a feature and accuracy is reported per year.
- Some columns are missing by survey design in particular years (e.g. `LearnCode` in 2025, `WorkExp` in 2023/24).
