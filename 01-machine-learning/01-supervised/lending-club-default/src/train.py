"""Step 4: train and compare 4 models, choose the best, evaluate it once on the test set.

Protocol (and why):
1. Out-of-time split: train on loans issued 2012-2014, test on 2015. A lender
   always scores future applicants with a model built on the past, so a random
   split would flatter the model.
2. Tune each model with 3-fold stratified cross-validation on a random sample
   of the training years (config.TUNING_SAMPLE rows), scored by ROC AUC. ROC
   AUC measures ranking quality at every threshold and is not fooled by class
   imbalance the way accuracy is.
3. Compare the tuned models and a "predict the base rate" baseline on their
   out-of-fold predictions: ROC AUC, PR AUC, Brier score, and F1, precision,
   recall and accuracy at each model's F1-maximising threshold. Only about 15%
   of loans default, so a 0.5 cut-off flags almost nothing.
   No class weighting is used, so predicted probabilities stay calibrated and
   can be read as probabilities of default (PD), which is what a lender prices on.
4. Pick the model with the best CV ROC AUC and keep its F1-maximising threshold.
5. Refit the winner on *all* training years, score the 2015 test set once, and
   save the model, metrics, figures and a risk-decile table.

Run from the project folder:  python src/train.py
"""
import json
import os
import time
import warnings

# numpy 2.0 on macOS (Apple Accelerate) prints false "divide by zero /
# overflow encountered in matmul" warnings; the results are correct. Silence
# them here and, through PYTHONWARNINGS, in the parallel worker processes.
_MATMUL_NOISE = [f"{kind} encountered in matmul" for kind in ("divide by zero", "overflow", "invalid value")]
os.environ["PYTHONWARNINGS"] = ",".join(f"ignore:{m}:RuntimeWarning" for m in _MATMUL_NOISE)
for _m in _MATMUL_NOISE:
    warnings.filterwarnings("ignore", message=_m, category=RuntimeWarning)

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,  # noqa: E402
                             brier_score_loss, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from config import (CATEGORICAL, CLEAN_FILE, FIGURES_DIR, MODELS_DIR, NUMERIC, REPORTS_DIR,  # noqa: E402
                    SEED, TARGET, TEST_YEAR, TUNING_SAMPLE)


def preprocessor(scale: bool) -> ColumnTransformer:
    # Fill missing numbers with the training median and add a "was missing"
    # flag (missing can itself be informative). One-hot encode categories.
    # Logistic regression needs scaled inputs; trees do not.
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                              sparse_output=False), CATEGORICAL),
        ("num", Pipeline(numeric_steps), NUMERIC),
    ], sparse_threshold=0)  # dense output: gradient boosting rejects sparse input


def candidates() -> dict:
    """Each entry: (pipeline, small hyperparameter grid)."""
    return {
        "logistic_regression": (
            Pipeline([("prep", preprocessor(scale=True)),
                      ("model", LogisticRegression(max_iter=3000))]),
            {"model__C": [0.01, 0.1, 1.0]},
        ),
        "decision_tree": (
            Pipeline([("prep", preprocessor(scale=False)),
                      ("model", DecisionTreeClassifier(random_state=SEED))]),
            {"model__max_depth": [6, 10], "model__min_samples_leaf": [100, 500]},
        ),
        "random_forest": (
            Pipeline([("prep", preprocessor(scale=False)),
                      ("model", RandomForestClassifier(n_estimators=200, max_samples=0.3,
                                                       n_jobs=-1, random_state=SEED))]),
            {"model__max_depth": [10, 16], "model__min_samples_leaf": [20, 100]},
        ),
        "gradient_boosting": (
            Pipeline([("prep", preprocessor(scale=False)),
                      ("model", HistGradientBoostingClassifier(max_iter=500, early_stopping=True,
                                                               random_state=SEED))]),
            {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31]},
        ),
    }


def metrics_at(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
    }


def best_f1_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def decile_table(y_true, proba) -> pd.DataFrame:
    """Sort loans from riskiest to safest, cut into 10 equal groups."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": proba})
    frame["risk_decile"] = pd.qcut(frame["p"].rank(method="first", ascending=False), 10,
                                   labels=range(1, 11))
    table = frame.groupby("risk_decile", observed=True).agg(
        loans=("y", "size"), predicted_pd=("p", "mean"), actual_default_rate=("y", "mean"))
    table["share_of_all_defaults"] = frame.groupby("risk_decile", observed=True)["y"].sum() / frame["y"].sum()
    table["cumulative_defaults_caught"] = table["share_of_all_defaults"].cumsum()
    return table.round(4)


def main() -> None:
    df = pd.read_csv(CLEAN_FILE)
    train, test = df[df["issue_year"] < TEST_YEAR], df[df["issue_year"] == TEST_YEAR]
    features = CATEGORICAL + NUMERIC
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]
    tune = train.sample(n=min(TUNING_SAMPLE, len(train)), random_state=SEED)
    X_tune, y_tune = tune[features], tune[TARGET]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    print(f"Train (2012-{TEST_YEAR - 1}): {len(train):,} loans, default rate {y_train.mean():.1%}")
    print(f"Test  ({TEST_YEAR}):      {len(test):,} loans, default rate {y_test.mean():.1%}")
    print(f"Tuning on a {len(tune):,}-loan sample of the training years\n")

    base = cross_val_predict(DummyClassifier(strategy="prior"), X_tune, y_tune, cv=cv,
                             method="predict_proba")[:, 1]
    rows = [{"model": "baseline_base_rate", **metrics_at(y_tune, base, 0.5), "best_params": "-",
             "seconds": 0.0}]
    tuned, oof = {}, {}
    for name, (pipe, grid) in candidates().items():
        start = time.time()
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tune, y_tune)
        proba = cross_val_predict(search.best_estimator_, X_tune, y_tune, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
        tuned[name], oof[name] = search.best_estimator_, proba
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        # F1, precision and recall depend on the cut-off, so each model is scored
        # at its own F1-maximising threshold (chosen on these out-of-fold predictions).
        rows.append({"model": name, **metrics_at(y_tune, proba, best_f1_threshold(y_tune, proba)),
                     "best_params": json.dumps(params), "seconds": round(time.time() - start, 1)})
        print(f"{name:20s} CV ROC AUC {rows[-1]['roc_auc']:.4f}  best {params}  ({rows[-1]['seconds']}s)")

    comparison = pd.DataFrame(rows).round(4)
    winner = comparison.iloc[1:].sort_values("roc_auc", ascending=False).iloc[0]["model"]
    threshold = best_f1_threshold(y_tune, oof[winner])

    print(f"\nRefitting {winner} on all {len(train):,} training loans...")
    final = tuned[winner].fit(X_train, y_train)
    test_proba = final.predict_proba(X_test)[:, 1]
    test_default = metrics_at(y_test, test_proba, 0.5)
    test_tuned = metrics_at(y_test, test_proba, threshold)
    deciles = decile_table(y_test, test_proba)

    # Permutation importance: how much test ROC AUC drops when one column is
    # shuffled. A 30k-loan sample of the test set keeps this quick.
    sample = test.sample(n=min(30_000, len(test)), random_state=SEED)
    perm = permutation_importance(final, sample[features], sample[TARGET], scoring="roc_auc",
                                  n_repeats=5, random_state=SEED, n_jobs=-1)
    importance = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False).round(4)

    for d in (REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORTS_DIR / "model_comparison_cv.csv", index=False)
    deciles.to_csv(REPORTS_DIR / "test_risk_deciles.csv")
    results = {
        "winner": winner,
        "winner_params": json.loads(comparison.set_index("model").loc[winner, "best_params"]),
        "threshold_max_f1": round(threshold, 4),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_metrics_threshold_0.5": {k: round(v, 4) for k, v in test_default.items()},
        "test_metrics_tuned_threshold": {k: round(v, 4) for k, v in test_tuned.items()},
        "top_features": importance.head(10).to_dict(),
    }
    (REPORTS_DIR / "test_results.json").write_text(json.dumps(results, indent=2))
    joblib.dump({"model": final, "threshold": threshold, "features": features},
                MODELS_DIR / "best_model.joblib")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, proba in oof.items():
        fpr, tpr, _ = roc_curve(y_tune, proba)
        ax1.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_tune, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_tune, proba)
        ax2.plot(rec, prec, label=f"{name} ({average_precision_score(y_tune, proba):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="random (0.500)")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (cross-validated)")
    ax2.axhline(y_tune.mean(), color="k", ls="--", lw=1, label=f"random ({y_tune.mean():.3f})")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (cross-validated)")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, thr in [(axes[0], 0.5), (axes[1], threshold)]:
        ConfusionMatrixDisplay.from_predictions(y_test, (test_proba >= thr).astype(int), ax=ax,
                                                display_labels=["Paid", "Default"], colorbar=False)
        ax.set_title(f"{TEST_YEAR} test set, threshold {thr:.2f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 4))
    importance.head(10)[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set(xlabel="Drop in test ROC AUC when shuffled", title=f"Top 10 features ({winner})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(deciles.index.astype(str), deciles["actual_default_rate"], color="#DD8452")
    ax.axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"average ({y_test.mean():.1%})")
    ax.set(xlabel="Risk decile (1 = riskiest 10% of loans)", ylabel="Actual default rate",
           title=f"{TEST_YEAR} loans ranked by predicted risk")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "risk_deciles.png", dpi=120)
    plt.close("all")

    print("\nCross-validated comparison (F1, precision, recall at each model's best threshold):")
    print(comparison.drop(columns=["best_params"]).to_string(index=False))
    print(f"\nWinner: {winner}  |  F1-maximising threshold: {threshold:.3f}")
    print(f"\n{TEST_YEAR} test set (seen once):")
    print(pd.DataFrame({"threshold 0.5": test_default, f"threshold {threshold:.2f}": test_tuned})
          .round(4).to_string())
    print("\nRisk deciles on the test set:")
    print(deciles.to_string())
    print("\nTop 5 features:", ", ".join(importance.head(5).index))
    print("\nSaved reports/, reports/figures/ and models/best_model.joblib")


if __name__ == "__main__":
    main()
