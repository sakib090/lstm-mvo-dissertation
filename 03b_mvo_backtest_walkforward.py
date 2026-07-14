"""
03b_mvo_backtest_walkforward.py
Same monthly-rebalanced MVO backtest as 03_mvo_backtest.py, but fed the
walk-forward (quarterly retrain, expanding window) LSTM predictions from
02b_lstm_walkforward.py instead of the single-split predictions.

Lets you compare, side by side:
  data/processed/backtest_metrics.csv               <- single-split LSTM-MVO
  data/processed/walkforward/backtest_metrics_walkforward.csv  <- walk-forward LSTM-MVO

Reuses mvo_weights() and performance_metrics() from 03_mvo_backtest.py
directly, so the optimisation and metric definitions are identical between
the two backtests -- only the LSTM predictions feeding "lstm_mvo" differ.
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


def run_backtest_walkforward():
    returns = dp.load_index_returns()
    pred_df = pd.read_csv(PRED_PATH, index_col=0, parse_dates=True)
    assets = list(returns.columns)

    # Walk-forward predictions only cover 2019-2022 (the test quarters), so
    # rebalance dates are simply every month-end within that predicted range.
    month_ends = pred_df.resample("ME").last().index
    month_ends = month_ends[(month_ends >= pred_df.index.min()) & (month_ends <= pred_df.index.max())]

    weights_log = {"lstm_mvo": [], "hist_mvo": [], "equal": []}
    dates_log = []
    prev_w = {"lstm_mvo": np.ones(3) / 3, "hist_mvo": np.ones(3) / 3, "equal": np.ones(3) / 3}
    port_daily_returns = {"lstm_mvo": [], "hist_mvo": [], "equal": []}

    for i in range(len(month_ends) - 1):
        rebal_date = month_ends[i]
        next_date = month_ends[i + 1]

        # trailing 252-day sample covariance up to the rebalance date (annualised)
        hist_window = returns.loc[:rebal_date].tail(252)
        cov = hist_window.cov().values * 252

        # expected returns for the coming month
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

        # realised daily returns over the holding period (rebal_date, next_date]
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
        prev_w[name] = None

    weights_df = {name: pd.DataFrame(weights_log[name], index=dates_log, columns=assets)
                  for name in weights_log}

    metrics = {name: performance_metrics(r) for name, r in results.items()}
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df.round(4))

    metrics_df.to_csv("data/processed/walkforward/backtest_metrics_walkforward.csv")
    for name, r in results.items():
        r.to_csv(f"data/processed/walkforward/daily_returns_{name}_walkforward.csv")
    for name, w in weights_df.items():
        w.to_csv(f"data/processed/walkforward/weights_{name}_walkforward.csv")

    return results, weights_df


if __name__ == "__main__":
    run_backtest_walkforward()