"""Step 2 (Transform + Load): select, validate, clean and feature-engineer the loans.

What this does, in order:
1. Load only the columns known at approval time (see config.RAW_COLUMNS).
2. Keep loans with a final outcome: 36-month loans issued 2012-2015 that ended
   as Fully Paid or Charged Off. Label: 1 = Charged Off (default), 0 = Fully Paid.
3. Parse messy text fields: " 36 months" -> 36, "10+ years" -> 10,
   "Aug-2003" -> a date, "13.5%" -> 13.5.
4. Fix impossible values: negative or 999 debt-to-income, incomes of 0,
   revolving utilisation far above 100%.
5. Engineer features a credit analyst would use: average FICO, years of credit
   history, loan-to-income and installment-to-income.
6. Save the clean table to data/processed/ and a data-quality report to reports/.

Missing values are *not* filled here. They are filled inside the model pipeline
using training data only, so the test set never influences the training step.

Run from the project folder:  python src/etl.py
"""
import json

import numpy as np
import pandas as pd

from config import (CATEGORICAL, CLEAN_FILE, FIRST_YEAR, LAST_YEAR, NUMERIC, PROCESSED_DIR,
                    RAW_COLUMNS, RAW_DIR, RAW_PATTERN, REPORTS_DIR, TARGET, TERM_MONTHS)

FINAL_STATUS = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.rglob(RAW_PATTERN))
    if not files:
        raise FileNotFoundError("No raw file found. Run `python src/download_data.py` first.")
    print(f"Reading {files[0].name} (about a minute for 2.2M rows)...")
    return pd.read_csv(files[0], usecols=RAW_COLUMNS, low_memory=False)


def to_number(series: pd.Series) -> pd.Series:
    """Turn values like '13.5%', ' 36 months' or 13.5 into floats."""
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype(str).str.extract(r"(-?\d+\.?\d*)")[0]
    return pd.to_numeric(series, errors="coerce")


def parse_emp_length(series: pd.Series) -> pd.Series:
    # "< 1 year" -> 0, "1 year" -> 1, ..., "10+ years" -> 10, "n/a" -> missing
    text = series.astype(str)
    years = to_number(text)
    return years.where(~text.str.startswith("<"), 0.0)


def select_loans(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    report["raw_rows"] = int(len(df))
    report["loan_status_counts_raw"] = df["loan_status"].value_counts(dropna=False).to_dict()
    df = df.dropna(subset=["loan_status", "issue_d"])  # the file ends with summary rows
    df = df.assign(issue_d=pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce"),
                   term=to_number(df["term"]))
    df = df[df["term"].eq(TERM_MONTHS) & df["issue_d"].dt.year.between(FIRST_YEAR, LAST_YEAR)]
    report["rows_36m_2012_2015"] = int(len(df))
    df = df[df["loan_status"].isin(FINAL_STATUS)].copy()
    report["rows_with_final_outcome"] = int(len(df))
    if df.empty:
        raise ValueError("No loans left after filtering. Check loan_status, term and issue_d formats.")
    df[TARGET] = df["loan_status"].map(FINAL_STATUS).astype(int)
    return df


def clean(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    for col in ["int_rate", "revol_util", "loan_amnt", "installment", "annual_inc", "dti"]:
        df[col] = to_number(df[col])
    df["emp_years"] = parse_emp_length(df["emp_length"])
    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_years"] = (df["issue_d"] - earliest).dt.days / 365.25

    fixes = {
        "dti_invalid": int((~df["dti"].between(0, 100)).sum() - df["dti"].isna().sum()),
        "income_zero_or_negative": int((df["annual_inc"] <= 0).sum()),
        "revol_util_above_150": int((df["revol_util"] > 150).sum()),
    }
    df["dti"] = df["dti"].where(df["dti"].between(0, 100))
    df["annual_inc"] = df["annual_inc"].where(df["annual_inc"] > 0)
    df["revol_util"] = df["revol_util"].clip(upper=150)
    # Clip the top 0.1% of incomes: a few self-reported values are in the tens of millions.
    df["annual_inc"] = df["annual_inc"].clip(upper=df["annual_inc"].quantile(0.999))
    df["home_ownership"] = df["home_ownership"].replace({"ANY": "OTHER", "NONE": "OTHER"})
    report["value_fixes"] = fixes
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    df["installment_to_income"] = df["installment"] * 12 / df["annual_inc"]
    df["issue_year"] = df["issue_d"].dt.year
    return df[["issue_year"] + CATEGORICAL + NUMERIC + [TARGET]]


def main() -> None:
    report = {}
    df = engineer(clean(select_loans(load_raw(), report), report))
    report["final_rows"] = int(len(df))
    report["final_columns"] = int(df.shape[1])
    report["default_rate_overall"] = round(float(df[TARGET].mean()), 4)
    report["default_rate_by_year"] = df.groupby("issue_year")[TARGET].mean().round(4).to_dict()
    report["rows_by_year"] = df["issue_year"].value_counts().sort_index().to_dict()
    report["missing_share_by_column"] = {
        k: round(float(v), 4) for k, v in df.isna().mean().items() if v > 0}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"Raw rows:                        {report['raw_rows']:,}")
    print(f"36-month loans issued 2012-2015: {report['rows_36m_2012_2015']:,}")
    print(f"... with a final outcome:        {report['rows_with_final_outcome']:,}")
    print(f"Value fixes: {report['value_fixes']}")
    print(f"Missing values (share of rows): {report['missing_share_by_column']}")
    print(f"Default rate by year: {report['default_rate_by_year']}")
    print(f"Saved {len(df):,} rows x {df.shape[1]} columns to data/processed/{CLEAN_FILE.name}")


if __name__ == "__main__":
    main()
