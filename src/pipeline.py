from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.transformers import MultiHotEncoder, SkillCounts

TARGET = "PayTier"

NUMERIC = ["Age_num", "YearsCode_num", "EdLevel_ord", "WorkExp_num", "WorkExp_missing"]
MULTI = ["LanguageHaveWorkedWith", "DatabaseHaveWorkedWith",
         "PlatformHaveWorkedWith", "WebframeHaveWorkedWith",
         "DevType", "LearnCode"]
JOB_CONTEXT = ["OrgSize", "RemoteWork", "ICorPM", "Industry"]
# Columns added in experiment 14 (same answer options in every survey year)
EXTRA_NUMERIC = ["PurchaseInfluence_ord"]
EXTRA_MULTI = ["OfficeStackAsyncHaveWorkedWith", "OpSysProfessional use"]


def build_preprocessor(use_job_context=False, use_country=False, min_freq=0.01,
                       multi_cols=MULTI, skill_counts=False, extra_features=False):
    """Return a ColumnTransformer for the chosen feature group."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    parts = [
        ("num", numeric_pipe, NUMERIC + (EXTRA_NUMERIC if extra_features else [])),
        ("multi", MultiHotEncoder(min_freq=min_freq),
         list(multi_cols) + (EXTRA_MULTI if extra_features else [])),
    ]
    if skill_counts:
        parts.append(("counts", Pipeline([("count", SkillCounts()),
                                          ("scale", StandardScaler())]), list(multi_cols)))
    cat_cols = (JOB_CONTEXT if use_job_context else []) + (["Country"] if use_country else [])
    if cat_cols:
        parts.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(parts, remainder="drop")


def build_category_indexer(cols):
    """Integer codes for categorical columns (0 = missing/unseen), for Keras embeddings."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("codes", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
