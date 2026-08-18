"""
s1_static_tilt.py
Feedback addressed: "Static tilt versus dynamic timing."

Question: does LSTM-MVO's timing add anything over a PORTFOLIO THAT NEVER MOVES,
held at LSTM-MVO's own average weights? Plus: how do the three single-index
buy-and-holds do? If the fixed-average-weight row matches LSTM-MVO, the
"active, sometimes concentrated timing" added nothing and the result is a tilt.

Output: output/s1_static_tilt.csv  and a printed table ready for the paper.
"""
import numpy as np
import pandas as pd
import pipeline_io as io

def portfolio_monthly_returns(daily: pd.DataFrame, weights_by_reb: pd.DataFrame) -> pd.Series:
    """Given daily asset returns and weights held from each rebalance date until
    the next, compound to the rebalance frequency. Weights are held CONSTANT in
    'target' terms and re-applied at each rebalance (matches your step-plot figures)."""
    reb_dates = weights_by_reb.index
    out = {}
    for i, d0 in enumerate(reb_dates):
        d1 = reb_dates[i + 1] if i + 1 < len(reb_dates) else daily.index[-1]
        window = daily.loc[(daily.index > d0) & (daily.index <= d1)]
        if window.empty:
            continue
        w = weights_by_reb.loc[d0].values
        # compound each asset over the window, then dot with weights (buy-and-hold within period)
        asset_growth = (1 + window).prod().values
        port_growth = float(np.dot(w, asset_growth))
        out[d1] = port_growth - 1.0
    return pd.Series(out).sort_index()

def constant_weight_frame(index, w):
    return pd.DataFrame([w] * len(index), index=index, columns=io.ASSETS)

def main():
    daily = io.load_returns()
    lstm_w = io.load_weights("lstm")           # actual dynamic weights held
    reb_index = lstm_w.index

    rows = {}

    # 1) LSTM-MVO as actually run (reconstructed here for apples-to-apples)
    rows["LSTM-MVO (dynamic)"] = io.metrics_block(
        portfolio_monthly_returns(daily, lstm_w))

    # 2) Fixed-average-weight: hold LSTM-MVO's MEAN weight, never re-optimise
    avg_w = lstm_w.mean().values
    avg_w = avg_w / avg_w.sum()
    rows["Fixed-avg-weight (LSTM mean)"] = io.metrics_block(
        portfolio_monthly_returns(daily, constant_weight_frame(reb_index, avg_w)))

    # 3) Three single-index buy-and-holds
    for j, a in enumerate(io.ASSETS):
        w = np.zeros(len(io.ASSETS)); w[j] = 1.0
        rows[f"Buy&Hold {a}"] = io.metrics_block(
            portfolio_monthly_returns(daily, constant_weight_frame(reb_index, w)))

    tbl = pd.DataFrame(rows).T
    tbl["AnnReturn"] = (tbl["AnnReturn"] * 100).round(2)
    tbl["AnnVol"] = (tbl["AnnVol"] * 100).round(2)
    tbl["MaxDD"] = (tbl["MaxDD"] * 100).round(2)
    tbl["Sharpe"] = tbl["Sharpe"].round(3)
    tbl["Sortino"] = tbl["Sortino"].round(3)
    tbl.to_csv(io.OUTPUT_DIR / "s1_static_tilt.csv")

    print("\n=== Static tilt vs dynamic timing ===")
    print(tbl.to_string())
    print("\nLSTM-MVO average weights:",
          dict(zip(io.ASSETS, np.round(avg_w, 3))))
    print("\nINTERPRETATION: if 'Fixed-avg-weight' Sharpe ~= 'LSTM-MVO (dynamic)' "
          "Sharpe, the timing added nothing and the result is a static tilt.\n"
          "Compare the fixed-avg row and each buy-and-hold to LSTM-MVO.")

if __name__ == "__main__":
    main()
