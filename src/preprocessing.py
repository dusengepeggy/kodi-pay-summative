import pandas as pd

PRO_DEV = "I am a developer by profession"
CODE_SOMETIMES = ("I am not primarily a developer, but I write code "
                  "sometimes as part of my work/studies") 

SALARY_MIN, SALARY_MAX = 1_000, 1_000_000
MIN_COUNTRY_ROWS = 100

def clean_and_label(df, branches=(PRO_DEV,), full_time_only=True,
                    min_country_rows=MIN_COUNTRY_ROWS):
    """Filter respondents, keep countries with enough rows, add PayTier.
    Returns (cleaned_df, funnel_table)."""

    steps = [("Raw pooled rows", len(df))]

    def log(name,d):
        steps.append((name, len(d)))

    d =  df[df["ConvertedCompYearly"].notna()].copy()
    log("Respondents with salary", d)

    d = d[d["MainBranch"].isin(branches)]
    log("Selected MainBranch", d)

    if full_time_only:
        emp = d["Employment"].fillna("")
        # 2023/24 say "Employed, full-time"; 2025 says just "Employed"
        d = d[emp.str.contains("Employed, full-time") | (emp == "Employed")]
        log("Employed full-time", d)

    d = d[d["ConvertedCompYearly"].between(SALARY_MIN, SALARY_MAX)]
    log(f"Salary in [{SALARY_MIN:,}, {SALARY_MAX:,}]", d)

    counts = d["Country"].value_counts()
    keep = counts[counts >= min_country_rows].index
    d = d[d["Country"].isin(keep)]
    log(f"Countries with >={min_country_rows} rows", d)

    # Within-country terciles; method="first" avoids duplicate-edge errors
    pct = d.groupby("Country")["ConvertedCompYearly"].rank(pct=True, method="first")
    d = d.assign(PayTier=pd.cut(pct, [0, 1/3, 2/3, 1.0],
                                labels=["Low", "Mid", "High"],
                                include_lowest=True))
    return d.reset_index(drop=True), pd.DataFrame(steps, columns=["Step", "Rows"])