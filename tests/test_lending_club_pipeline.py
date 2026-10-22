"""Quick checks that run on GitHub Actions without downloading the real data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-machine-learning" / "01-supervised" / "lending-club-default" / "src"))
import etl  # noqa: E402
import train  # noqa: E402
from config import CATEGORICAL, NUMERIC  # noqa: E402


def fake_features(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(["a", "b", "c"], n) for c in CATEGORICAL})
    for c in NUMERIC:
        df[c] = rng.random(n)
    df.loc[::17, "emp_years"] = np.nan
    y = (df["int_rate"] + rng.normal(0, 0.3, n) > 0.8).astype(int)
    return df, y


def test_text_numbers_are_parsed():
    parsed = etl.to_number(pd.Series([" 36 months", "13.5%", None]))
    assert parsed.iloc[0] == 36 and parsed.iloc[1] == 13.5 and np.isnan(parsed.iloc[2])
    years = etl.parse_emp_length(pd.Series(["< 1 year", "10+ years", "3 years"]))
    assert years.tolist() == [0, 10, 3]


def test_every_model_trains_and_predicts_probabilities():
    X, y = fake_features()
    for name, (pipe, _) in train.candidates().items():
        proba = pipe.fit(X, y).predict_proba(X)[:, 1]
        assert proba.min() >= 0 and proba.max() <= 1, name


def test_risk_deciles_cover_all_defaults():
    _, y = fake_features()
    table = train.decile_table(y, np.linspace(0, 1, len(y)))
    assert len(table) == 10
    assert abs(table["cumulative_defaults_caught"].iloc[-1] - 1) < 1e-9
