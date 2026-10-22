# Project 1: Predicting loan default at approval time (Lending Club)

**Question:** On the day a personal loan is approved, can we tell which borrowers will fail to repay it? If we can, a lender can price risk better or decline the riskiest applications.

**Data:** Lending Club accepted loans, 2007 to 2018: 2.26 million real US personal loans with about 150 columns ([Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)).

**Models compared:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, plus a no-skill baseline.

**Result:** Logistic regression wins. On 2015 loans it never saw, it reaches **ROC AUC 0.69**, and the riskiest 20% of loans it flags contain **38% of all defaults** (see [Results](#6-results)).

---

## How to run

```bash
cd 01-machine-learning/01-supervised/lending-club-default
pip install -r requirements.txt
python src/download_data.py   # 1. Extract: download from Kaggle (needs ~/.kaggle/kaggle.json)
python src/etl.py             # 2. Transform + Load: clean data -> data/processed/
python src/eda.py             # 3. Explore: statistical tests + charts -> reports/
python src/train.py           # 4. Model: train, compare, test -> reports/ and models/
```

Prefer a step-by-step walkthrough with charts inline? Open [`notebooks/01_lending_club_walkthrough.ipynb`](notebooks/01_lending_club_walkthrough.ipynb). It runs the same code as the scripts, one step per cell, with explanations.

The whole pipeline takes a few minutes on a laptop. The data is not stored in git; `download_data.py` rebuilds it.

## Project structure

```
01-lending-club-default/
├── notebooks/
│   └── 01_lending_club_walkthrough.ipynb   # the whole project, step by step
├── src/
│   ├── config.py          # paths, which loans we study, feature lists
│   ├── download_data.py   # Kaggle download
│   ├── etl.py             # selection, cleaning, feature engineering
│   ├── eda.py             # statistical tests and charts
│   └── train.py           # model comparison, threshold choice, final test
├── data/                  # raw/ and processed/ (git-ignored)
├── reports/               # JSON/CSV results and figures/
└── models/                # saved best model (git-ignored)
```

---

## 1. What we did with the data (ETL)

**Extract.** We download the accepted-loans file from Kaggle.

**Choosing which loans to study.** This step decides whether the labels can be trusted.
- We keep **36-month loans issued 2012 to 2015**. The data ends in December 2018, so every one of these loans has reached its final outcome. A 2017 loan that is still "Current" has no answer yet. Keeping only the ones that already finished would over-represent loans that went bad early, which biases the label.
- We keep loans whose final status is **Fully Paid (0)** or **Charged Off (1)**, the outcome we want to predict.

**Avoiding data leakage.** About 100 of the 150 columns describe what happened *after* the loan was approved: `total_pymnt`, `recoveries`, `last_pymnt_d`, `last_fico_range`, and others. A model trained on them would score almost perfectly in testing and be useless in real life, because none of that is known when someone applies. We load **only the 27 columns known at application time** (listed in `config.py`).

**Cleaning.**
| Problem in the raw data | What we did |
|---|---|
| Text numbers: `" 36 months"`, `"10+ years"`, `"< 1 year"` | Parsed to numbers (36, 10, 0) |
| Dates as text: `"Aug-2003"` | Parsed to dates, then credit-history length in years |
| Debt-to-income below 0 or equal to 999 | Set to missing (impossible values) |
| Annual income of 0, or in the tens of millions | 0 set to missing; top 0.1% clipped |
| Revolving utilisation far above 100% | Clipped at 150% |
| Rare home-ownership codes `ANY` / `NONE` | Merged into `OTHER` |
| Summary rows at the end of the file | Dropped |

Missing values are **filled inside the model pipeline** (median plus a "was missing" flag) and learned from training data only, so the test set never leaks into training.

**Feature engineering.** We add features a credit analyst would compute: average FICO score, years of credit history, loan-to-income and installment-to-income ratios.

**Load.** The clean table is saved to `data/processed/loans_clean.csv.gz`. A data-quality report (row counts at each step, fixes applied, missing shares, default rate by year) is saved to `reports/data_quality.json`.

## 2. Exploratory analysis with statistical tests

Charts show patterns. Tests tell us whether a pattern is bigger than chance.

| Question | Test | Why this test |
|---|---|---|
| Does default rate differ by grade, purpose, home ownership? | Chi-square test of independence, with **Cramér's V** for effect size | Both variables are categorical |
| Do defaulters have different FICO, interest rate, DTI, income? | **Mann-Whitney U**, with rank-biserial effect size | Incomes and balances are heavily skewed, and a t-test assumes roughly normal data |

With 500,000+ loans almost every p-value is tiny, so **effect sizes** decide which features matter. Results are in `reports/eda_tests.json`; charts are in `reports/figures/`.

## 3. Why these four models

| Model | How it works | Why it is in the comparison |
|---|---|---|
| **Logistic Regression** | A weighted sum of features passed through a sigmoid gives the probability of default | The industry standard for credit scorecards. Interpretable, and regulators accept it. This is the bar the others must beat. |
| **Decision Tree** | Splits borrowers into groups with yes/no rules, such as "FICO < 690 and DTI > 25" | Easy to explain as rules. Shows how far simple non-linear rules get. |
| **Random Forest** | Averages hundreds of trees, each trained on a random sample of loans and features | Captures interactions and reduces a single tree's instability (variance) |
| **Gradient Boosting** | Builds trees one after another, each correcting the errors of the ones before | Usually the strongest model on tabular data. This is the same idea as XGBoost and LightGBM. |
| Baseline | Predicts the average default rate for everyone | Any real model must beat "no skill" |

No class weighting is used, so every model outputs a **calibrated probability of default (PD)**. A lender can use that number directly for pricing.

## 4. How the models are evaluated

**Out-of-time split.** We train on loans issued 2012 to 2014 and test on 2015. A lender always uses a model built on past loans to score future applicants. A random split would mix years and give an optimistic score.

**Cross-validation for model choice.** Each model's settings are tuned with 3-fold stratified cross-validation on a 150,000-loan sample of the training years. The 2015 test set is used **once**, at the very end, for the chosen model only.

**The metrics, and what each one tells a lender:**

Every prediction falls into one of four groups: **TP** (predicted default, did default), **FP** (predicted default, actually paid), **FN** (predicted paid, actually defaulted), **TN** (predicted paid, did pay).

| Metric | Formula | Plain meaning |
|---|---|---|
| **Accuracy** | (TP + TN) / all | Share of predictions that are right. *Misleading here*: predicting "everyone pays" scores about 85% and catches no defaults. |
| **Precision** | TP / (TP + FP) | Of the loans we flag as risky, how many really default |
| **Recall** | TP / (TP + FN) | Of all loans that default, how many we catch |
| **F1 score** | 2 × P × R / (P + R) | Balances precision and recall; high only if both are high |
| **ROC AUC** | Area under the true-positive-rate vs false-positive-rate curve | Probability that a random defaulter gets a higher risk score than a random good borrower. 0.5 = coin flip, 1.0 = perfect. It does not depend on any threshold. |
| **PR AUC** | Area under the precision-recall curve | Like ROC AUC but focused on the rare default class. Baseline = the default rate. |
| **Brier score** | Mean of (predicted PD − outcome)² | Are the probabilities themselves accurate? Lower is better. |

**Choosing the threshold.** A model outputs a probability, and we must pick a cut-off above which a loan is flagged. With only about 15% defaults, the usual 0.5 flags almost nobody. We choose the threshold that **maximises F1 on the cross-validation predictions** (never on the test set), and report test results at both 0.5 and the tuned threshold.

**Risk deciles.** We sort the 2015 loans from riskiest to safest and cut them into 10 groups. This shows the business value directly: how many of all defaults sit in the riskiest 10% or 20%, and whether predicted PD matches the actual default rate in each group (calibration).

## 5. Choosing the winner

The model with the highest cross-validated ROC AUC wins. If two models are within about 0.005 AUC, the simpler and more interpreta## 6. Results

From the run on 2026-10-22 (all numbers are in `reports/`).

**Cross-validated comparison** (training years 2012-2014, 150,000-loan sample). F1, precision and recall are at each model's own F1-maximising threshold (about 0.15).

| Model | CV ROC AUC | CV PR AUC | CV F1 | Precision | Recall | Brier |
|---|---|---|---|---|---|---|
| Baseline (predict the average) | 0.500 | 0.132 | 0.000 | 0.000 | 0.000 | 0.1148 |
| **Logistic Regression** | **0.676** | **0.232** | **0.311** | 0.216 | 0.558 | **0.1096** |
| Decision Tree | 0.658 | 0.213 | 0.297 | 0.204 | 0.549 | 0.1108 |
| Random Forest | 0.674 | 0.230 | 0.310 | 0.215 | 0.557 | 0.1098 |
| Gradient Boosting | 0.675 | 0.231 | 0.310 | 0.217 | 0.545 | 0.1097 |

**Winner: Logistic Regression** (C = 0.1). It has the best ROC AUC, PR AUC and Brier score. Random forest and gradient boosting are within 0.002 AUC, so the extra complexity buys nothing here, and logistic regression is the easiest to explain to a credit committee. With only about 25 application-time features, most of the signal is close to linear, so the tree ensembles have little left to find.

**On the 2015 test set** (283,026 loans, seen once):

| Threshold | ROC AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| 0.50 (default) | 0.692 | 0.002 | 0.376 | 0.001 | 0.851 |
| **0.15 (tuned on CV)** | 0.692 | **0.344** | 0.256 | **0.524** | 0.703 |

This is the threshold lesson in one table. At 0.5 the model flags almost nobody and catches 0.1% of defaults, yet accuracy looks good (85%). At the tuned 0.15 it catches 52% of defaults, and about 1 in 4 flagged loans really defaults (against 1 in 7 on average). Test ROC AUC (0.69) is slightly higher than CV (0.68), so the model holds up on a new year.

**Risk deciles (2015 loans sorted by predicted risk):**

| Decile | Predicted PD | Actual default rate | Defaults caught so far |
|---|---|---|---|
| 1 (riskiest 10%) | 28.3% | 31.7% | 21% |
| 2 | 20.1% | 25.0% | **38%** |
| 3 | 16.5% | 20.5% | 52% |
| 5 | 12.0% | 14.6% | 74% |
| 10 (safest 10%) | 3.6% | 3.2% | 100% |

The riskiest tenth of loans defaults 10 times as often as the safest tenth. Predicted PD tracks the actual rate in every decile (for example 28% predicted vs 32% actual at the top), with a slight under-prediction because 2015 was a worse year than 2012-2014.

**Top features** (drop in test ROC AUC when the column is shuffled): loan amount, installment, installment-to-income, grade, interest rate, FICO, DTI. Loan amount and installment rank highest *together*: for a 36-month loan, the installment is fixed by the amount and the interest rate, so the pair carries Lending Club's own rate (its risk price). Shuffling either one breaks that link.

Charts: `reports/figures/roc_pr_curves.png`, `confusion_matrices.png`, `risk_deciles.png`, `feature_importance.png`.

**Winner on the 2015 test set:**

**Riskiest 10% / 20% of loans contain:** % / % of all defaults.

**Top features:**

## 7. Limitations and next steps

- **Selection bias:** we only see loans Lending Club *approved*. Rejected applicants never got a label, so the model learns "risk among approved borrowers".
- **Grade and interest rate** come from Lending Club's own risk model, so part of our accuracy is borrowed from theirs. A next step is to train without them and measure how much ours adds.
- **Next steps:** SHAP explanations for individual decisions, a profit-based threshold (the loss on a default vs the interest earned on a good loan), and testing on 2016 loans to check for drift.
