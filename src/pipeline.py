from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.transformers import MultiHotEncoder

TARGET = "PayTier"

NUMERIC = ["Age_num", "YearsCode_num", "EdLevel_ord", "WorkExp_num", "WorkExp_missing"]
MULTI = ["LanguageHaveWorkedWith", "DatabaseHaveWorkedWith",
         "PlatformHaveWorkedWith", "WebframeHaveWorkedWith",
         "DevType", "LearnCode"]
JOB_CONTEXT = ["OrgSize", "RemoteWork", "ICorPM", "Industry"]


def build_preprocessor(use_job_context=False, use_country=False, min_freq=0.01):
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
        ("num", numeric_pipe, NUMERIC),
        ("multi", MultiHotEncoder(min_freq=min_freq), MULTI),
    ]
    cat_cols = (JOB_CONTEXT if use_job_context else []) + (["Country"] if use_country else [])
    if cat_cols:
        parts.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(parts, remainder="drop")