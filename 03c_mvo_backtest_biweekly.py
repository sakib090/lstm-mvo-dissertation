"""
03c_mvo_backtest_biweekly.py
Same walk-forward LSTM predictions and same MVO optimisation as
03b_mvo_backtest_walkforward.py, but rebalancing every 2 weeks instead of
monthly. This isolates the effect of rebalancing frequency alone -- same
predictions, same optimiser, same transaction cost rate -- to test whether
acting on the LSTM's signal more often actually helps, or whether the extra
transaction costs from more frequent trading erode the gain.

Outputs (compare directly against the monthly walk-forward results):
  data/processed/walkforward/backtest_metrics_biweekly.csv
  data/processed/walkforward/daily_returns_{name}_biweekly.csv
  data/processed/walkforward/weights_{name}_biweekly.csv
"""
import numpy as np
import pandas as pd
from importlib import import_module

dp = import_module("01_data_prep")
bt = import_module("03_mvo_backtest")

mvo_weights = bt.mvo_weights
performance_metrics = bt.performance_metrics
TXN_COST_BPS = bt.TXN_COST_BPS

PRED_PATH = "data/processed/walkforward/lstm_predictions_walkforward.csv"
REBAL_FREQ_DAYS = 14  # 2 weeks


def get_rebalance_dates(index, freq_days):
    """Pick actual trading dates spaced freq_days calendar days apart,
    starting from the first available date. Each target date snaps to the
    nearest available trading date on or before it."""
    start, end = index.min(), index.max()
    targets = pd.date_range(start, end, freq=f"{freq_days}D")
    rebal_dates = []
    for t in targets:
        candidates = index[index <= t]
        if len(candidates) > 0:
            rebal_dates.append(candidates[-1])
    return pd.DatetimeIndex(sorted(set(rebal_dates)))


def run_backtest_biweekly():
    returns = dp.load_index_returns()
    pred_df = pd.read_csv(PRED_PATH, index_col=0, parse_dates=True)
    assets = list(returns.columns)

    rebal_dates = get_rebalance_dates(pred_df.index, REBAL_FREQ_DAYS)

    weights_log = {"lstm_mvo": [], "hist_mvo": [], "equal": []}
    dates_log = []
    prev_w = {"lstm_mvo": np.ones(3) / 3, "hist_mvo": np.ones(3) / 3, "equal": np.ones(3) / 3}
    port_daily_returns = {"lstm_mvo": [], "hist_mvo": [], "equal": []}

    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        # trailing 252-day sample covariance up to the rebalance date (annualised)
        hist_window = returns.loc[:rebal_date].tail(252)
        cov = hist_window.cov().values * 252

        # expected returns for the coming 2-week period
        # LSTM: mean of its 5-day-forward predictions available around the rebalance date
        lstm_exp = pred_df.loc[:rebal_date].tail(21).mean().values * (252 / 5)
        # Classical: trailing historical mean, annualised
        hist_exp = hist_window.mean().values * 252

        w_lstm = mvo_weights(lstm_exp, cov)
        w_hist = mvo_weights(hist_exp, cov)
        w_eq = np.ones(3) / 3

        weights_log["lstm_mvo"].append(w_lstm)
        weights_log["hist_mvo"].append(w_hist)
        weights_log["equal"].append(w_eq)
        dates_log.append(rebal_date)

        period_returns = returns.loc[(returns.index > rebal_date) & (returns.index <= next_date)]

        for name, w, prev in [("lstm_mvo", w_lstm, prev_w["lstm_mvo"]),
                               ("hist_mvo", w_hist, prev_w["hist_mvo"]),
                               ("equal", w_eq, prev_w["equal"])]:
            turnover = np.abs(w - prev).sum()
            cost = turnover * (TXN_COST_BPS / 10000)
            daily_port_ret = period_returns.values @ w
            if len(daily_port_ret) > 0:
                daily_port_ret[0] -= cost
            port_daily_returns[name].append(
                pd.Series(daily_port_ret, index=period_returns.index)
            )
            prev_w[name] = w

    results = {}
    for name in port_daily_returns:
        results[name] = pd.concat(port_daily_returns[name]).sort_index()

    weights_df = {name: pd.DataFrame(weights_log[name], index=dates_log, columns=assets)
                  for name in weights_log}

    metrics = {name: performance_metrics(r) for name, r in results.items()}
    metrics_df = pd.DataFrame(metrics).T
    print(f"Rebalanced every {REBAL_FREQ_DAYS} days -- {len(rebal_dates)-1} rebalances total")
    print(metrics_df.round(4))

    metrics_df.to_csv("data/processed/walkforward/backtest_metrics_biweekly.csv")
    for name, r in results.items():
        r.to_csv(f"data/processed/walkforward/daily_returns_{name}_biweekly.csv")
    for name, w in weights_df.items():
        w.to_csv(f"data/processed/walkforward/weights_{name}_biweekly.csv")

    return results, weights_df


if __name__ == "__main__":
    run_backtest_biweekly()