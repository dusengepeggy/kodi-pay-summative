import pandas as pd

PRO_DEV = "I am a developer by profession"
CODE_SOMETIMES = ("I am not primarily a developer, but I write code "
                  "sometimes as part of my work/studies")

SALARY_MIN, SALARY_MAX = 1_000, 1_000_000
MIN_COUNTRY_ROWS = 100
TIERS = ["Low", "Mid", "High"]
EXP_BANDS = [0, 5, 10, 20, 100]
EXP_LABELS = ["0-4", "5-9", "10-19", "20+"]


def filter_respondents(df, branches=(PRO_DEV,), full_time_only=True):
    """Salary, MainBranch, employment and salary-range filters.
    Returns (filtered_df, funnel_steps)."""
    steps = [("Raw pooled rows", len(df))]

    d = df[df["ConvertedCompYearly"].notna()].copy()
    steps.append(("Respondents with salary", len(d)))

    d = d[d["MainBranch"].isin(branches)]
    steps.append(("Selected MainBranch", len(d)))

    if full_time_only:
        emp = d["Employment"].fillna("")
        # 2023/24 say "Employed, full-time"; 2025 says just "Employed"
        d = d[emp.str.contains("Employed, full-time") | (emp == "Employed")]
        steps.append(("Employed full-time", len(d)))

    d = d[d["ConvertedCompYearly"].between(SALARY_MIN, SALARY_MAX)]
    steps.append((f"Salary in [{SALARY_MIN:,}, {SALARY_MAX:,}]", len(d)))
    return d, steps


def tier_within(d, by):
    """Tercile of salary within each group (rank pct; method="first" avoids
    duplicate-edge errors)."""
    pct = d.groupby(by, observed=True)["ConvertedCompYearly"].rank(pct=True, method="first")
    return pd.cut(pct, [0, 1/3, 2/3, 1.0], labels=TIERS, include_lowest=True)


def clean_and_label(df, branches=(PRO_DEV,), full_time_only=True,
                    min_country_rows=MIN_COUNTRY_ROWS):
    """Filter respondents, keep countries with enough rows, add PayTier.
    Returns (cleaned_df, funnel_table)."""
    d, steps = filter_respondents(df, branches, full_time_only)

    counts = d["Country"].value_counts()
    d = d[d["Country"].isin(counts[counts >= min_country_rows].index)]
    steps.append((f"Countries with >={min_country_rows} rows", len(d)))

    d = d.assign(PayTier=tier_within(d, "Country"))
    return d.reset_index(drop=True), pd.DataFrame(steps, columns=["Step", "Rows"])


def label_with_cutoffs(rows, reference):
    """Label new rows with the per-country salary cut-offs of `reference`
    (used so extra training rows share the strict training label scale)."""
    cuts = (reference.groupby("Country")["ConvertedCompYearly"]
            .quantile([1/3, 2/3]).unstack())
    rows = rows[rows["Country"].isin(cuts.index)].copy()
    lo = rows["Country"].map(cuts[1/3])
    hi = rows["Country"].map(cuts[2/3])
    sal = rows["ConvertedCompYearly"]
    tier = pd.Series("Mid", index=rows.index)
    tier[sal <= lo] = "Low"
    tier[sal > hi] = "High"
    return rows.assign(PayTier=pd.Categorical(tier, categories=TIERS))


def add_experience_label(d, min_group=30):
    """Experience-adjusted label: tercile within Country x years-coding band.
    Rows with unknown experience or in groups smaller than `min_group` get NaN."""
    band = pd.cut(d["YearsCode_num"], EXP_BANDS, labels=EXP_LABELS, right=False)
    d = d.assign(ExpBand=band)
    size = d.groupby(["Country", "ExpBand"], observed=True)["ResponseId"].transform("size")
    ok = d["ExpBand"].notna() & (size >= min_group)
    tier = pd.Series(pd.NA, index=d.index, dtype="object")
    tier[ok] = tier_within(d[ok], ["Country", "ExpBand"]).astype(str)
    return d.assign(PayTierExp=tier)
