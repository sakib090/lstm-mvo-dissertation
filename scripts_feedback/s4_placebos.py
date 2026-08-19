"""
s4_placebos.py  (v2 — uses real mvo_weights + walk-forward schedule)
Feedback: random-mu placebo, momentum-as-mu, minimum-variance, inverse-vol.

Reuses 03_mvo_backtest.mvo_weights and the SAME monthly-rebalance /
trailing-252 covariance / 5bps-cost mechanics as 03b, so every placebo is
directly comparable to LSTM-MVO. Only the expected-return vector mu changes
(or, for min-var/inv-vol, weights are set without mu).

Run from the project root (same folder as 01_data_prep.py etc.) so the
`import_module` calls resolve, e.g.:
    python scripts_feedback/s4_placebos.py
"""
import sys, os
import numpy as np
import pandas as pd
from importlib import import_module

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts_feedback"))

import pipeline_io as io
dp = import_module("01_data_prep")
bt = import_module("03_mvo_backtest")
mvo_weights = bt.mvo_weights
TXN = bt.TXN_COST_BPS

ASSETS = io.ASSETS
LSTM_WF_SHARPE = 0.84   # walk-forward LSTM-MVO Sharpe, for placebo comparison


def rebalance_dates(returns):
    """Month-end rebalance dates over the 2019-2022 test window (same as 03b)."""
    me = returns.loc["2019-01-01":"2022-12-31"].resample("ME").last().index
    return me[(me >= returns.index.min()) & (me <= returns.index.max())]


def backtest_with_mu(returns, mu_func, use_mu=True):
    """Monthly-rebalanced backtest. mu_func(rebal_date, hist_window) -> mu vector,
    or None when use_mu=False (min-variance path). Returns daily P&L series."""
    me = rebalance_dates(returns)
    prev_w = np.ones(3) / 3
    daily_chunks = []
    for i in range(len(me) - 1):
        d0, d1 = me[i], me[i + 1]
        hist = returns.loc[:d0].tail(252)
        cov = hist.cov().values * 252
        if use_mu:
            mu = mu_func(d0, hist)
            w = mvo_weights(mu, cov)
        else:
            w = mvo_weights(np.zeros(3), cov)  # mu=0 -> minimum-variance
        period = returns.loc[(returns.index > d0) & (returns.index <= d1)]
        turnover = np.abs(w - prev_w).sum()
        cost = turnover * (TXN / 10000)
        r = period.values @ w
        if len(r) > 0:
            r[0] -= cost
        daily_chunks.append(pd.Series(r, index=period.index))
        prev_w = w
    return pd.concat(daily_chunks).sort_index()


def inv_vol_backtest(returns):
    me = rebalance_dates(returns)
    prev_w = np.ones(3) / 3
    daily_chunks = []
    for i in range(len(me) - 1):
        d0, d1 = me[i], me[i + 1]
        hist = returns.loc[:d0].tail(252)
        iv = 1.0 / hist.std().values
        w = iv / iv.sum()
        period = returns.loc[(returns.index > d0) & (returns.index <= d1)]
        cost = np.abs(w - prev_w).sum() * (TXN / 10000)
        r = period.values @ w
        if len(r) > 0:
            r[0] -= cost
        daily_chunks.append(pd.Series(r, index=period.index))
        prev_w = w
    return pd.concat(daily_chunks).sort_index()


def sharpe(daily):
    return (daily.mean() * 252) / (daily.std() * np.sqrt(252) + 1e-12)


def main():
    returns = dp.load_index_returns()

    # (1) Random-mu placebo x200
    mu_mean = returns.mean().values
    mu_cov = returns.cov().values
    rng = np.random.default_rng(0)
    sharpes = []
    for _ in range(200):
        draw = rng.multivariate_normal(mu_mean, mu_cov) * 252  # annualise like the real mu
        pnl = backtest_with_mu(returns, lambda d, h, m=draw: m)
        sharpes.append(sharpe(pnl))
    sharpes = np.array(sharpes)
    pct = (sharpes < LSTM_WF_SHARPE).mean() * 100
    print("=== (1) Random-mu placebo (x200) ===")
    print(f"  Placebo Sharpe: mean {sharpes.mean():.3f}, "
          f"5th-95th [{np.percentile(sharpes,5):.3f}, {np.percentile(sharpes,95):.3f}], "
          f"max {sharpes.max():.3f}")
    print(f"  LSTM-MVO ({LSTM_WF_SHARPE}) sits at the {pct:.0f}th percentile.")
    if sharpes.max() >= LSTM_WF_SHARPE:
        print("  >>> Placebo distribution CONTAINS the LSTM Sharpe. Report this plainly.")
    else:
        print("  >>> LSTM Sharpe exceeds all 200 random-mu draws. Signal beats random mu.")

    # (2) Momentum-as-mu (avg of 63d & 252d momentum, annualised like real mu)
    def mom_mu(d0, hist):
        m63 = (1 + hist.tail(63)).prod() - 1
        m252 = (1 + hist.tail(252)).prod() - 1
        return ((m63 + m252) / 2).values
    pnl_mom = backtest_with_mu(returns, mom_mu)

    # (3) Minimum-variance and inverse-vol
    pnl_minvar = backtest_with_mu(returns, None, use_mu=False)
    pnl_invvol = inv_vol_backtest(returns)

    print("\n=== (2)-(3) Alternative-signal / no-signal portfolios ===")
    for name, pnl in [("Momentum-as-mu", pnl_mom),
                      ("Minimum-variance", pnl_minvar),
                      ("Inverse-volatility", pnl_invvol)]:
        m = io.performance_metrics_daily(pnl)
        print(f"  {name:20s}: Sharpe {m['Sharpe']:.3f}  AnnRet {m['AnnReturn']*100:5.2f}%  "
              f"MaxDD {m['MaxDD']*100:6.2f}%")
    print(f"\n  (Compare against walk-forward LSTM-MVO Sharpe {LSTM_WF_SHARPE}.)")

    pd.DataFrame({"random_mu_sharpe": sharpes}).to_csv(io.OUTPUT_DIR / "s4_random_mu.csv", index=False)


if __name__ == "__main__":
    main()