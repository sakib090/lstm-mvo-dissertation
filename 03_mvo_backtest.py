"""
03_mvo_backtest.py
Builds monthly-rebalanced portfolios using:
  1. LSTM-MVO   - expected returns from the LSTM, sample covariance
  2. Hist-MVO   - expected returns = trailing historical mean (classical MVO)
  3. Equal-weight benchmark
Then backtests all three over the LSTM test period and reports
Sharpe, Sortino, Max Drawdown and annualised return.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from importlib import import_module

dp = import_module("01_data_prep")

RISK_FREE_DAILY = 0.0  # simplifying assumption, noted as a limitation in the paper
TXN_COST_BPS = 5        # 5 bps per unit of turnover, applied at each monthly rebalance


def mvo_weights(exp_returns: np.ndarray, cov: np.ndarray, risk_aversion=3.0):
    """Long-only, fully-invested mean-variance weights via a simple QP-style
    optimisation: maximise (w'mu - risk_aversion/2 * w'Cov w)."""
    n = len(exp_returns)
    x0 = np.ones(n) / n

    def neg_utility(w):
        return -(w @ exp_returns - 0.5 * risk_aversion * w @ cov @ w)

    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(neg_utility, x0, bounds=bounds, constraints=constraints, method="SLSQP")
    if not res.success:
        return x0  # fall back to equal-weight if the optimiser fails
    return res.x


def run_backtest():
    returns = dp.load_index_returns()
    pred_df = pd.read_csv("lstm_predictions.csv", index_col=0, parse_dates=True)
    assets = list(returns.columns)

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

        # expected returns for the coming month:
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

        # realised daily returns over the holding period [rebal_date, next_date)
        period_returns = returns.loc[(returns.index > rebal_date) & (returns.index <= next_date)]

        for name, w, prev in [("lstm_mvo", w_lstm, prev_w["lstm_mvo"]),
                               ("hist_mvo", w_hist, prev_w["hist_mvo"]),
                               ("equal", w_eq, prev_w["equal"])]:
            turnover = np.abs(w - prev).sum()
            cost = turnover * (TXN_COST_BPS / 10000)
            daily_port_ret = period_returns.values @ w
            if len(daily_port_ret) > 0:
                daily_port_ret[0] -= cost  # apply cost on rebalance day
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

    return results, weights_df


def performance_metrics(daily_returns: pd.Series, periods_per_year=252):
    ann_return = (1 + daily_returns).prod() ** (periods_per_year / len(daily_returns)) - 1
    ann_vol = daily_returns.std() * np.sqrt(periods_per_year)
    sharpe = (daily_returns.mean() * periods_per_year) / (daily_returns.std() * np.sqrt(periods_per_year) + 1e-12)
    downside = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() * periods_per_year) / (downside.std() * np.sqrt(periods_per_year) + 1e-12)
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1
    max_dd = drawdown.min()
    return {
        "Annualised Return": ann_return,
        "Annualised Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
    }


if __name__ == "__main__":
    results, weights_df = run_backtest()
    metrics = {name: performance_metrics(r) for name, r in results.items()}
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df.round(4))
    metrics_df.to_csv("backtest_metrics.csv")
    for name, r in results.items():
        r.to_csv(f"daily_returns_{name}.csv")
    for name, w in weights_df.items():
        w.to_csv(f"weights_{name}.csv")