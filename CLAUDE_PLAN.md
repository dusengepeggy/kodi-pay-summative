# CLAUDE_PLAN.md — Developer Pay-Tier Classification (ML vs DL)

> It records every decision made so far, the current state of the project, and the remaining work.
> **Do not silently change a decision below. If one needs to change, say why and update this file.**

---

## 0. Working agreements (how Claude should work with me)

1. **Explain every block of code** in plain language: what it does, why, and what to look for in the output. Walk through line by line for anything non-trivial.
2. **One experiment per notebook cell**, each with a markdown cell above (hypothesis, what changed vs. the previous experiment) and a short observation below.
3. **The test set is untouched** until the final model is chosen. Tune on validation or CV on train only. **Claude pauses for my OK before touching the test set.**
4. **Never fabricate references, numbers or outputs.** If a number is not in the notebook output, say so and ask me to run the cell.
5. **Report prose is mine.** The assignment limits AI-generated text to <20%. Claude may help with outlines, structure, feedback on drafts I write, explaining concepts and code. Claude must not write the report body for me.
6. Keep code **modular** (logic in `src/`, notebook only orchestrates), seeded, and runnable top to bottom. Light comments in `src/`; the explanations live in notebook markdown.
7. When I paste an error, ask for the **full traceback** and fix the root cause.
8. Ask at most one clarifying question at a time.
9. Commit after every crucial step. **No Co-Authored-By trailer on commits.**

---

## 1. Project summary

| Item | Value |
|---|---|
| Repo / folder | `kodi-pay-summative` (public GitHub repo, no raw data committed) |
| Mission | Reduce youth unemployment and underpayment in Rwanda by giving young people evidence about which skills and choices go with better pay in software careers |
| Problem statement | *Given a developer's profile (experience, education, skills, role, how they learned to code, work setup), can we predict their compensation tier relative to peers in their country, and which factors separate low-pay from high-pay tiers, so that young people entering software careers know which skills and choices are associated with better pay?* |
| Task | 3-class classification: `Low` / `Mid` / `High` (`PayTier`) |
| Dataset | Stack Overflow Annual Developer Survey, 2023 + 2024 + 2025 pooled (open CSVs from the `StackExchange/Survey` GitHub repo) |
| Environment | Windows, VS Code, Python **3.14.7** `.venv` (see §3 Environment for why this changed from 3.12) |

### Why the mission fits, and honest limits
- The model learns from a **global, self-selected** developer sample. Rwanda has only 64 raw rows (26 with salary), so it **cannot be modelled directly**. Say this openly.
- Rwanda / East Africa appear only as a small **descriptive case study**, never with statistical claims.
- NISR Labour Force Survey microdata (2020–2024; ~72k–102k person records, ICT-skills and ISCO occupation variables, access via the NISR catalog login) is **not** the modelling dataset: ISCO groups are coarse ("Professionals"), skills are coarse, and no developer-level skills/pay data. Use NISR **only as citable evidence** for youth unemployment in the introduction/literature review/discussion. Verify access terms and the youth definition (commonly 16–30) from an official source before citing.

### Known risk to flag with the instructor
`PayTier` is **derived by binning** a numeric salary. The brief bans "regression-only datasets without class labels". Mitigation: ask the instructor to confirm; justify in the report. Fallback target in the same data: `Employment` ("Not employed, but looking for work", heavily imbalanced).

---

## 2. Assignment requirements checklist (60 pts)

Hard requirements: ≥20,000 raw rows (we have 203,812), open source, categorical target, **sklearn** (multiple algorithms) **and TensorFlow/Keras** (Sequential or Functional, with **`tf.data`**), hyperparameter tuning, feature engineering, learning curves, confusion matrices, ROC-AUC, precision/recall. Report 3,500–5,000 words, IEEE. Public GitHub repo (README, `requirements.txt`, notebook), **no raw data on GitHub**. Google Drive folder ("anyone with link can view") with zipped raw data, best sklearn model (`.joblib`) and best Keras model. 5–10 min video, face on camera. GitHub, Drive and video links **on the first page of the report**.

| Rubric criterion | Pts | Where we satisfy it |
|---|---|---|
| Problem, data, mission alignment | 6 | §1, intro, NISR evidence, dataset justification |
| Literature review (≥10 scholarly sources, critical, IEEE) | 7 | Report §3 (I find real sources) |
| Preprocessing and feature engineering | 5 | §4–5, ablations, funnel table |
| Model implementation and optimization (7+ experiments) | 7 | §7 roadmap |
| Experiment/result table (+ TensorBoard or W&B offline logs) | 9 | `results/experiments.csv`, settings JSONs, TensorBoard |
| Evaluation and error analysis | 6 | §8 |
| Code quality and documentation | 10 | §3, §9 |
| Report quality and originality (<20% AI) | 5 | §11 |
| Demo video | 5 | §12 |

---

## 3. Repository layout and hygiene

```
kodi-pay-summative/
├── CLAUDE_PLAN.md
├── README.md
├── requirements.txt          # pinned with `pip freeze` once stable
├── .gitignore
├── notebooks/01_main_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── splits.py
│   ├── transformers.py
│   ├── pipeline.py
│   ├── evaluation.py
│   ├── experiments.py        # ExperimentTracker (sklearn)
│   └── dl.py                 # Keras models, tf.data, Keras tracker
├── data/                     # gitignored (raw CSVs and zip)
├── models/                   # gitignored except best_*
├── results/
│   ├── experiments.csv       # committed
│   └── settings/exp_*.json   # committed
├── figures/                  # committed (report figures)
└── logs/                     # TensorBoard (gitignored)
```

`.gitignore` contains `models/*` with a `!models/best_*.joblib` exception (the wildcard is required for the `!` exception to work) and `!results/*.csv`. Commit best models only if each is under ~50 MB; otherwise Drive only. Raw data is **never** committed; the notebook downloads it.

### Environment
- **Decision change (2026-10-17):** the plan originally said Python 3.12. 3.12 is not installed on this machine, and the existing `.venv` (Python 3.14.7) runs TensorFlow 2.22.0rc0 and Keras correctly (verified by building and calling a model). The notebook also executes headlessly with nbconvert. So we keep 3.14 and pin the exact versions in `requirements.txt`. If the kernel crashes again, fall back to installing 3.12.
- Packages: pandas, numpy, scikit-learn, tensorflow, matplotlib, seaborn, requests, joblib, tensorboard, wandb, ipykernel.
- Notebook first cell: `sys.path.append("..")`, `%load_ext autoreload`, `%autoreload 2`.
- Notebook runs from `notebooks/`, so figure paths use `../figures`.

---

## 4. Data pipeline (implemented, keep as is)

### 4.1 `data_loading.py`
- `PROJECT_ROOT = Path(__file__).resolve().parents[1]`.
- `download_data(year)` fetches `https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive/{year}/results.csv` into `data/survey_{year}.csv` if absent.
- `load_years()` uses `pd.read_csv(..., low_memory=False)`.
- `harmonize()` keeps only columns shared by all three years (37 shared columns) and adds `Year`.
- `make_raw_archive()` zips `data/` for the Drive upload.

### 4.2 Raw facts (cite in report)
| Year | Raw rows | With salary | Rwanda rows |
|---|---|---|---|
| 2023 | 89,184 | 48,019 | 27 |
| 2024 | 65,437 | 23,435 | 22 |
| 2025 | 49,191 | 23,947 | 15 |
| Total | **203,812** | 95,401 | 64 (26 with salary) |

### 4.3 `preprocessing.clean_and_label(df, branches=(PRO_DEV,), full_time_only=True, min_country_rows=100)`
Funnel (current default): **203,812 → 95,401 (has salary) → 86,098 (professional developers) → 75,357 (employed full-time) → 74,014 (salary $1,000–$1,000,000) → 71,612 (countries with ≥100 rows, 63 countries)**.
- Full-time rule: `"Employed, full-time"` substring **or** exactly `"Employed"` (2025 dropped the full/part-time distinction).
- `PayTier` = within-country tercile of `ConvertedCompYearly` (rank pct, `method="first"`), **pooled across years** (per-year terciles are too noisy: ~30 rows/country/year).
- Returns `(data, funnel_table)`. The funnel table goes in the report.
- Observed drift: tier mix by year is ~34/34/32 (2023), ~36/33/32 (2024), ~28.5/32/39.5 (2025). Treat as a **limitation**, report accuracy by year, and **never use `Year` as a feature**.
- Limitations to state: tiers are rank-based so test salaries slightly influence cut-offs; `method="first"` can split identical salaries across tiers; nominal USD over three years.

### 4.3b Population caveats
Survey is self-selected; salaries are self-reported; Africa is thin after filtering (only **South Africa 533** and **Nigeria 142** survive; Kenya, Ghana, Ethiopia, Uganda and Rwanda are dropped).

### 4.4 `features.add_basic_features(df)`
- `Age_num` from midpoints (17, 21, 30, 40, 50, 60, 67); "Prefer not to say" becomes NaN.
- `YearsCode_num` ("Less than 1 year" = 0.5, "More than 50 years" = 51), `WorkExp_num`, `EdLevel_ord` (1–7 via keyword rules; "Something else"/"Other (please specify):"/blank = NaN).
- **Plausibility checks:** values above 51, or above (oldest age in bracket − earliest start age) are masked to NaN (coding start ≥5, work start ≥14). The raw data had maxima of 100. Count stored in `df.attrs["n_implausible_years"]` (113 values).
- `WorkExp_missing` flag is computed **after** masking. `IsYouth` = under 18 or 18–24 (used for evaluation, not as a model feature).
- Fixed rules only (no learning from data), so applying before the split does not leak.

### 4.5 `splits.make_splits(data, test_size=0.15, val_size=0.15, seed=42)`
- Stratified on `Year + "_" + PayTier`. Sizes: **train 50,128 / val 10,742 / test 10,742**, tier mix 33.4/33.3/33.3 in each.
- **Canonical order in the notebook: clean_and_label → add_basic_features → make_splits.**

### 4.6 `transformers.MultiHotEncoder(min_freq=0.01, sep=";")`
- Sklearn-compatible; **vocabulary learned on training rows only**; unknown tokens ignored at transform.
- Columns: `LanguageHaveWorkedWith` (36 kept), `DatabaseHaveWorkedWith` (26), `PlatformHaveWorkedWith` (43), `WebframeHaveWorkedWith` (29), `DevType` (16), `LearnCode` (20) = **170 features**.
- Caveat: `LearnCode` is 29% missing **only in 2025** (survey design); blank = all zeros. Keep it but test its effect.

### 4.7 `pipeline.build_preprocessor(use_job_context=False, use_country=False, min_freq=0.01)`
- Numeric (`Age_num`, `YearsCode_num`, `EdLevel_ord`, `WorkExp_num`, `WorkExp_missing`): median impute, then standard scale.
- Multi-hot as above.
- Optional categorical (impute `"Unknown"`, one-hot, `handle_unknown="ignore"`): `OrgSize`, `RemoteWork`, `ICorPM`, `Industry` (job context) and/or `Country`.
- `remainder="drop"` keeps `PayTier`, salary columns, `Year` and `ResponseId` out of the model.
- Core feature matrix has **175 columns** (5 + 170).
- Dropped on purpose: `ResponseId`, `CompTotal`, `Currency`, `ConvertedCompYearly` (leakage); all `...WantToWorkWith` columns (preferences, 24–46% missing); AI columns (`AIAcc` 57%/37% missing); SO engagement and OS columns.
- Caution on job-context features: `ICorPM` and `Industry` are mostly missing in 2023/24 but not 2025, so `"Unknown"` can proxy for survey year. If experiment 2 improves, check accuracy by year.
- Caution on `Country`: tiers are country-relative, so `Country` alone should not help; test it as an ablation (a null result supports the label design).

---

## 5. Reproducibility and persistence (`src/experiments.py`)

`ExperimentTracker(train, val, target, seed=42)`:
- `set_seeds` seeds Python `random`, NumPy, `PYTHONHASHSEED` and TensorFlow.
- `run(exp_id, name, model, settings, note, force=False)`: seeds, computes `config_hash`, `data_hash` and `code_hash`. If a saved JSON matches all three hashes and the model file exists, it **loads the saved model instead of refitting**. With `force=True` it refits and prints the **largest metric difference vs the saved run** as the reproducibility proof.
- Saves per experiment: `models/exp_{id}.joblib`, `results/settings/exp_{id}.json` and a row in `results/experiments.csv` (re-running replaces the row). Metrics are also written as TensorBoard scalars under `logs/`.
- If data/filters/seed/features change, **delete `results/experiments.csv`** so stale rows cannot sit beside new ones.
- `dl.py` holds the Keras counterpart (saves `.keras` + history JSON, `enable_op_determinism`).

---

## 6. Results so far (validation set, before the notebook rebuild)

| Exp | Model | Val acc | Val macro-F1 | Val AUC | Notes |
|---|---|---|---|---|---|
| 0 | Majority class | 0.334 | 0.167 | 0.500 | chance baseline, confirms split/target |
| 1 | Logistic Regression, core features, C=1 | **0.539** | 0.529 | 0.719 | train acc 0.546 (gap <1 pt: high bias, not overfitting) |

- Accuracy by year (exp 1): 2023 = 0.536, 2024 = 0.530, 2025 = 0.558.
- Per-tier recall (exp 1): Low 0.685, Mid **0.323**, High 0.611. Mid's errors split evenly (36% to Low, 32% to High): it sits between two neighbours, and a flat multi-class model ignores the ordering.
- **Youth (1,178 val rows, aged under 25):** accuracy 0.677 vs the always-"Low" baseline **0.711** (worse on accuracy, but macro-F1 0.399 vs about 0.277 for the baseline). Recall Low 0.894, Mid 0.098, High 0.289. Only 76 High rows, so recall carries roughly ±10 pt uncertainty. Training-set youth tier mix: **71.5% Low, 20.7% Mid, 7.7% High**.
- Interpretation: tiers are relative to all developers including seniors, so low early-career tiers are largely seniority, **not evidence of underpayment**. Skill-based prediction is weak within the youth group.
- After the rebuild, `results/experiments.csv` is the source of truth.

---

## 7. Experiment roadmap (7+ required, each building on the last)

Each experiment: own cell + markdown hypothesis + observation. Fit on train only; compare on validation. Use **stratified K-fold on train** for hyperparameter search (`RandomizedSearchCV`, `random_state=42`, `scoring="f1_macro"`).

| # | Experiment | What changes | Question |
|---|---|---|---|
| 0 | Majority class | – | chance reference |
| 1 | Logistic Regression, core | – | interpretable benchmark |
| 2 | LR + job context | feature group | does employer info add anything beyond the profile? (check by year) |
| 3 | LR tuned (C, penalty, class_weight) | regularization | is the limit regularization or features? |
| 4 | Random Forest | non-linear, depth/n_estimators | is the ceiling model capacity? |
| 5 | HistGradientBoosting (tuned) | learning rate, depth, leaf size | strongest classical model |
| 6 | Best tree + feature engineering ablations | `min_freq` (0.005/0.01/0.02), with/without `LearnCode`, `Country`, skill-count features | which features matter? |
| 7 | Keras Sequential MLP via `tf.data` | layers, units, dropout | first deep model |
| 8 | Keras Functional model | embeddings for categoricals + multi-hot branch | architectural change |
| 9 | Tuned DL | learning rate, batch size, L2, early stopping, ReduceLROnPlateau | optimization |
| 10 | Training-population experiment | train on professionals + "code sometimes" (+ part-time/freelance) | does broader training help young developers? **Evaluate on the same strict validation set.** |
| 11 | Experience-adjusted label | tiers within country × experience band | do skills matter *beyond seniority*? (new label, so report separately) |
| 12 (optional) | Ordinal approach | cumulative binary classifiers | exploit Low < Mid < High to fix Mid |

Notes:
- SVM does not scale to ~50k rows × 175 features: use `LinearSVC` or a subsample and **say why**.
- For `tf.data`: build `Dataset.from_tensor_slices` from the **already-preprocessed** train/val matrices (preprocessor fitted on train only), `.shuffle(seed=42)`, `.batch()`, `.prefetch(tf.data.AUTOTUNE)`.
- Keras models: `tf.keras.utils.set_random_seed(42)`, early stopping on validation loss, save history for learning curves.
- Log every run to TensorBoard and keep a run-ID column linking the table to logs.

---

## 8. Evaluation and error analysis

For the best sklearn and best Keras models (and key ablations):
1. **Learning curves**: sklearn `learning_curve` and Keras loss/accuracy per epoch. Diagnose under/overfitting; tie to bias–variance.
2. **Confusion matrices** (row-normalized): all rows and **youth-only**. Adjacent vs far errors; why Mid is hard.
3. **ROC (one-vs-rest) and precision–recall curves** per tier, with AUC.
4. **Per-group accuracy and macro-F1:** by `Year`, by `IsYouth`, by country group (and the small Africa set descriptively), always with the **subgroup majority-class baseline**.
5. **Feature importance / coefficients:** LR coefficients, permutation importance on the tree model. **Associations, not causes**.
6. **"Predicted higher than actual" analysis**: describe the group, caveat heavily. Do not call them "underpaid".
7. **Rwanda/East Africa case study:** apply the final model to the small excluded African rows (descriptive only; report counts).
8. Final **test-set evaluation once**, for the chosen sklearn and Keras models.
9. Dataset limitations: self-selection, thin Africa coverage, year drift, binned target, nominal USD, missing-by-design columns.
10. Future work: ordinal models, experience-adjusted tiers, regional data (NISR), larger African samples.

Save report figures to `figures/` at 200 dpi.

---

## 9. Notebook structure (`01_main_analysis.ipynb`)

Every section opens with a **markdown cell** (purpose, method, expected output) and ends with an **interpretation cell**. Order:

1. Title, problem statement, links (GitHub / Drive / video), how to run, seed and versions
2. Setup (imports, autoreload, `set_seeds`, version print)
3. Data loading and size check (raw rows 203,812)
4. Exploration (respondents, missingness by year, salary distribution)
5. Cleaning and target construction (funnel, tier balance, drift, Africa counts)
6. Feature engineering and plausibility checks (+ `EdLevel` × `Year` crosstab)
7. Splitting (sizes, tier mix)
8. Preprocessing pipeline check (175 columns; multi-hot vocabulary)
9. Experiments 0–6 (traditional ML), plus a comparison table
10. `tf.data` pipeline and experiments 7–9 (deep learning)
11. Population and label experiments (10–11)
12. Evaluation: curves, matrices, ROC/PR, subgroup and error analysis
13. Final test evaluation (once)
14. Save best models and zip raw data for Drive
15. Conclusions and limitations; reproducibility proof (`force=True` rerun)

---

## 10. Outputs and deliverables

- **GitHub:** code, notebook, README, `requirements.txt`, `results/`, `figures/`. No raw data.
- **Google Drive** ("Anyone with the link can view"): zipped raw dataset, best sklearn model (`.joblib`), best Keras model (`.keras`), plus a short README of how to load them.
- Verify all three links open in an incognito window before submitting.

---

## 11. Report plan (IEEE, 3,500–5,000 words, **written by me**)

Links (GitHub, Drive, video) on page 1. Budget: Introduction ~500, Literature Review ~700, Methodology ~1,000, Results ~900, Discussion ~800, Conclusion ~300.

- **≥10 real scholarly sources**, compared critically. Topics: developer salary prediction with ML; youth unemployment in Rwanda/Sub-Saharan Africa; ICT skills mismatch; gradient boosting vs deep learning on tabular data; survey-data biases. **Claude must not invent citations.**
- Include: funnel table, experiment table, learning curves, confusion matrices, ROC/PR curves, subgroup tables. Tables report what was actually run.
- AI use: keep generated text well under 20%; write in my own voice, add my Rwanda context.

---

## 12. Demo video plan (5–10 min, face visible)

| Time | Content |
|---|---|
| 0:00–1:00 | Who I am, mission, problem statement |
| 1:00–2:00 | Dataset: source, size, limits, why chosen; NISR as context |
| 2:00–3:30 | Preprocessing, leakage prevention, feature engineering, funnel |
| 3:30–5:30 | Models and experiments, reasons for hyperparameter choices |
| 5:30–8:00 | Learning curves, confusion matrix, ROC; error analysis (Mid tier, youth) |
| 8:00–9:00 | Limitations, future work, transparency (dataset source, AI-tool assistance) |

---

## 13. Definition of done

- [ ] Instructor confirmation (or documented justification) on the binned target
- [x] `n_implausible_years` cleanup; `low_memory=False`
- [ ] `src/` modules complete, including `dl.py`
- [ ] Notebook runs **Restart → Run All** with no errors; markdown before and after every section
- [ ] ≥7 documented experiments (target: 11+), table complete
- [ ] TensorBoard logs backing the table
- [ ] Reproducibility proof (`force=True` difference ≈ 0)
- [ ] Learning curves, confusion matrices (all and youth), ROC/PR, subgroup analysis, bias–variance discussion
- [ ] Test set evaluated once
- [ ] Best sklearn + Keras models saved; Drive folder populated and public-viewable
- [ ] `requirements.txt` pinned; README written; no raw data in git
- [ ] Report and video
