"""
s5_linear_benchmarks.py  (v2 — reuses 02b feature construction)
Feedback: fit OLS and Ridge on the SAME 15 features and 5-day target, plus AR(1),
walk-forward, and report diagnostics next to the LSTM. Establishes whether the
LSTM architecture adds anything over a linear model on identical inputs.

Uses 02_lstm_model.build_panel + make_windows for identical features/target, then
takes the LAST timestep of each 60-day window as the linear models' input (the
standard way to give a linear model the same information without 900 features).

Run from project root:  python scripts_feedback/s5_linear_benchmarks.py
"""
import sys, os
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.linear_model import LinearRegression, Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts_feedback"))
import pipeline_io as io
dp = import_module("01_data_prep")
m2 = import_module("02_lstm_model")

TEST_START = pd.Timestamp("2019-01-01")


def build_data():
    returns = dp.load_index_returns()
    panel = m2.build_panel(returns)
    target_cols = [f"{c}_return" for c in returns.columns]
    X, y, dates = m2.make_windows(panel, target_cols, window=m2.WINDOW, horizon=m2.HORIZON)
    # X is [n, 60, 15]; use the LAST timestep -> [n, 15] for linear models
    X_last = X[:, -1, :]
    return X_last, y, pd.DatetimeIndex(dates), returns.columns.tolist()


def dir_acc(yt, yp):
    return float((np.sign(yt) == np.sign(yp)).mean())


def oos_r2(yt, yp):
    return float(1 - np.mean((yt - yp) ** 2) / np.mean(yt ** 2))


def walk_forward_linear(model_ctor, X, y, dates):
    """Expanding-window: refit each quarter start, predict that quarter.
    Mirrors 02b's quarterly cadence but far cheaper for a linear model."""
    preds = np.full_like(y, np.nan, dtype=float)
    q_starts = pd.date_range(TEST_START, dates.max(), freq="QS")
    for qs in q_starts:
        qe = qs + pd.offsets.QuarterEnd(0)
        train = dates < qs
        test = (dates >= qs) & (dates <= qe)
        if test.sum() == 0 or train.sum() < 100:
            continue
        m = model_ctor()
        m.fit(X[train], y[train])
        preds[test] = m.predict(X[test])
    mask = ~np.isnan(preds).any(axis=1)
    return preds[mask], y[mask]


def main():
    X, y, dates, cols = build_data()
    print(f"Data: {len(X)} windows, {X.shape[1]} features, target {y.shape[1]} indices")

    rows = {}
    for name, ctor in [("OLS", lambda: LinearRegression()),
                       ("Ridge(a=1)", lambda: Ridge(alpha=1.0))]:
        yp, yt = walk_forward_linear(ctor, X, y, dates)
        rows[name] = {"OOS_R2": oos_r2(yt, yp), "DirAcc": dir_acc(yt, yp),
                      "MSE": float(np.mean((yt - yp) ** 2))}

    # AR(1): predict each index's next 5-day cum return from its own previous one
    mask = dates >= TEST_START
    yt_ar = y[mask]
    yp_ar = np.roll(y, 1, axis=0)[mask]
    rows["AR(1)"] = {"OOS_R2": oos_r2(yt_ar, yp_ar), "DirAcc": dir_acc(yt_ar, yp_ar),
                     "MSE": float(np.mean((yt_ar - yp_ar) ** 2))}

    tbl = pd.DataFrame(rows).T
    print("\n=== Linear benchmarks on identical features/target (walk-forward) ===")
    print(tbl.round(4).to_string())
    print("\nLSTM (your 4.1, for comparison): DirAcc ~0.49-0.52, OOS R^2 ~ -0.07 to -0.14")
    print("If OLS/Ridge match the LSTM's near-chance DirAcc and similar OOS R^2, the")
    print("architecture adds nothing on these inputs -- a clean, honest finding.")
    tbl.to_csv(io.OUTPUT_DIR / "s5_linear_benchmarks.csv")


if __name__ == "__main__":
    main()