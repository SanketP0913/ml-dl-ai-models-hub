# ML · DL · AI Models Hub

End-to-end machine learning projects on real, large datasets. Each project runs the full workflow a data scientist does at work:

**ETL → data cleaning → EDA with statistical tests → train 3 to 4 models → compare with cross-validation → choose the best → explain the result.**

Every project has its own README that explains what was done to the data, why each model was chosen, and how each metric (ROC AUC, F1, precision, recall and more) is calculated and what it means.

## Projects

| # | Project | Domain | Data | Models compared | Best result |
|---|---|---|---|---|---|
| 1 | [Loan default at approval time](projects/01-lending-club-default) | Credit risk | Lending Club, 2.26M loans | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting | _in progress_ |

## How every project is organised

```
projects/NN-name/
├── README.md          # question, data work, model choice, metrics, results
├── requirements.txt
├── src/               # download_data.py → etl.py → eda.py → train.py
├── reports/           # metrics (JSON/CSV) and figures
└── data/, models/     # rebuilt by the scripts, not stored in git
```

Data is never committed. Each project ships a download script, so anyone can rebuild the results from scratch.

## Author

**Sanket Patil**, MPS Data Science, UMBC · [GitHub](https://github.com/SanketP0913)
