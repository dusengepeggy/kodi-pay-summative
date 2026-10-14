import numpy as np
import pandas as pd

AGE_MIDPOINTS = {
    "Under 18 years old": 17,
    "18-24 years old": 21,
    "25-34 years old": 30,
    "35-44 years old": 40,
    "45-54 years old": 50,
    "55-64 years old": 60,
    "65 years or older": 67,
}

AGE_UPPER = {
    "Under 18 years old": 17,
    "18-24 years old": 24,
    "25-34 years old": 34,
    "35-44 years old": 44,
    "45-54 years old": 54,
    "55-64 years old": 64,
    "65 years or older": 80,
}

EDU_RULES = [
    ("primary", 1),
    ("secondary", 2),
    ("some college", 3),
    ("associate", 4),
    ("bachelor", 5),
    ("master", 6),
    ("professional degree", 7),
]

MAX_YEARS = 51
MIN_START_AGE_CODING = 5
MIN_START_AGE_WORK = 14


def _education_level(text):
    """Map an EdLevel answer to an ordinal 1-7; unknown/other -> NaN."""
    if pd.isna(text):
        return np.nan
    t = text.lower()
    for keyword, level in EDU_RULES:
        if keyword in t:
            return level
    return np.nan


def add_basic_features(df):
    """Add numeric Age, YearsCode, EdLevel, WorkExp (with plausibility
    checks), a missing-experience flag, and a youth flag."""
    out = df.copy()

    out["Age_num"] = out["Age"].map(AGE_MIDPOINTS)

    years = (out["YearsCode"].astype(str)
             .replace({"Less than 1 year": "0.5", "More than 50 years": "51"}))
    out["YearsCode_num"] = pd.to_numeric(years, errors="coerce")
    out["WorkExp_num"] = pd.to_numeric(out["WorkExp"], errors="coerce")

    age_hi = out["Age"].map(AGE_UPPER)
    bad_code = (out["YearsCode_num"] > MAX_YEARS) | \
               (out["YearsCode_num"] > age_hi - MIN_START_AGE_CODING)
    bad_work = (out["WorkExp_num"] > MAX_YEARS) | \
               (out["WorkExp_num"] > age_hi - MIN_START_AGE_WORK)

    out["n_implausible_years"] = int(bad_code.sum() + bad_work.sum())
    out["YearsCode_num"] = out["YearsCode_num"].mask(bad_code)
    out["WorkExp_num"] = out["WorkExp_num"].mask(bad_work)

    out["EdLevel_ord"] = out["EdLevel"].map(_education_level)
    out["WorkExp_missing"] = out["WorkExp_num"].isna().astype(int)
    out["IsYouth"] = out["Age"].isin(["Under 18 years old", "18-24 years old"])
    return out