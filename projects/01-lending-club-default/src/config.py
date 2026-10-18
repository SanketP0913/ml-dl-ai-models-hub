"""Paths, dataset choices and feature lists shared by every step of the project."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"

KAGGLE_DATASET = "wordsforthewise/lending-club"
RAW_PATTERN = "accepted_2007_to_2018Q4.csv*"
CLEAN_FILE = PROCESSED_DIR / "loans_clean.csv.gz"

TARGET = "default"
SEED = 42

# Which loans we study, and why (details in the README):
# 36-month loans issued 2012-2015 had all reached their final outcome by the
# end of the data (Dec 2018), so every loan has a true label.
TERM_MONTHS = 36
FIRST_YEAR, LAST_YEAR = 2012, 2015
TEST_YEAR = 2015          # out-of-time test: train on 2012-2014, test on 2015
TUNING_SAMPLE = 150_000   # rows used for hyperparameter search (keeps it laptop-friendly)

# Only information Lending Club had on the day the loan was approved.
# Columns such as total_pymnt, recoveries or last_pymnt_d describe what happened
# *after* approval; using them would leak the answer into the model.
RAW_COLUMNS = [
    "loan_status", "issue_d", "term", "loan_amnt", "int_rate", "installment", "grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status", "purpose",
    "addr_state", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "application_type", "mort_acc", "pub_rec_bankruptcies",
]

CATEGORICAL = ["grade", "home_ownership", "verification_status", "purpose", "addr_state",
               "application_type"]
NUMERIC = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
           "fico", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
           "total_acc", "mort_acc", "pub_rec_bankruptcies", "emp_years", "credit_history_years",
           "loan_to_income", "installment_to_income"]
