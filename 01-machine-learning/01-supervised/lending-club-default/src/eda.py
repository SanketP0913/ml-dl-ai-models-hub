"""Step 3: exploratory data analysis with statistical tests.

Charts show what the data looks like; tests tell us whether a difference is
bigger than chance. For each question we pick the test that fits the data:
* Categorical feature vs default (yes/no): chi-square test of independence,
  with Cramer's V as the effect size (0 = no association, 1 = perfect).
* Numeric feature vs default: Mann-Whitney U test, because incomes and
  balances are heavily skewed and a t-test assumes roughly normal data.
  Effect size: rank-biserial correlation (-1 to 1).

With 500k+ rows almost every p-value is tiny, so the effect sizes are what
tell us which features actually matter.

Run from the project folder:  python src/eda.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from config import CLEAN_FILE, FIGURES_DIR, REPORTS_DIR, TARGET  # noqa: E402

CATEGORICAL_TO_TEST = ["grade", "home_ownership", "verification_status", "purpose", "application_type"]
NUMERIC_TO_TEST = ["int_rate", "fico", "dti", "annual_inc", "revol_util", "loan_to_income",
                   "inq_last_6mths", "credit_history_years"]


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


def main() -> None:
    df = pd.read_csv(CLEAN_FILE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = {"rows": int(len(df)), "default_rate": round(float(df[TARGET].mean()), 4),
               "categorical": {}, "numeric": {}}

    for col in CATEGORICAL_TO_TEST:
        table = pd.crosstab(df[col], df[TARGET])
        chi2, p, _, _ = stats.chi2_contingency(table)
        results["categorical"][col] = {
            "chi2": round(float(chi2), 1), "p_value": float(f"{p:.3g}"),
            "cramers_v": round(cramers_v(table), 3),
            "default_rate_by_group": df.groupby(col)[TARGET].mean().round(3).to_dict(),
        }

    for col in NUMERIC_TO_TEST:
        a = df.loc[df[TARGET] == 1, col].dropna()
        b = df.loc[df[TARGET] == 0, col].dropna()
        u, p = stats.mannwhitneyu(a, b)
        results["numeric"][col] = {
            "median_defaulted": round(float(a.median()), 3),
            "median_paid": round(float(b.median()), 3),
            "p_value": float(f"{p:.3g}"),
            "effect_size_rank_biserial": round(float(2 * u / (len(a) * len(b)) - 1), 3),
        }

    fig, ax = plt.subplots(figsize=(6, 3.5))
    df.groupby("grade")[TARGET].mean().plot.bar(ax=ax, color="#DD8452")
    ax.set(ylabel="Default rate", title="Default rate by Lending Club grade")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "default_rate_by_grade.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    df.groupby("issue_year")[TARGET].mean().plot(ax=ax, marker="o", color="#4C72B0")
    ax.set(xlabel="Issue year", ylabel="Default rate", title="Default rate drifts over time")
    ax.set_xticks(sorted(df["issue_year"].unique()))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "default_rate_by_year.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, color, name in [(0, "#4C72B0", "Fully paid"), (1, "#DD8452", "Charged off")]:
        ax.hist(df.loc[df[TARGET] == label, "fico"].dropna(), bins=40, alpha=0.6, density=True,
                color=color, label=name)
    ax.set(xlabel="FICO score at application", title="Charged-off loans start with lower FICO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fico_by_outcome.png", dpi=120)
    plt.close("all")

    (REPORTS_DIR / "eda_tests.json").write_text(json.dumps(results, indent=2))
    print(f"{results['rows']:,} loans, default rate {results['default_rate']:.1%}")
    print("\nCategorical features (chi-square):")
    for col, r in sorted(results["categorical"].items(), key=lambda kv: -kv[1]["cramers_v"]):
        print(f"  {col:20s} p={r['p_value']:<10g} Cramer's V={r['cramers_v']}")
    print("\nNumeric features (Mann-Whitney U), sorted by effect size:")
    for col, r in sorted(results["numeric"].items(), key=lambda kv: -abs(kv[1]["effect_size_rank_biserial"])):
        print(f"  {col:22s} median defaulted={r['median_defaulted']:<10g} "
              f"paid={r['median_paid']:<10g} p={r['p_value']:<10g} "
              f"effect={r['effect_size_rank_biserial']}")
    print("\nFigures saved to reports/figures/")


if __name__ == "__main__":
    main()
