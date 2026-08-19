"""
s2_return_attribution.py  (v3 — uses saved daily P&L; reconciles by construction)
Feedback addressed: decompose LSTM-MVO minus classical MVO into
(weight difference) x (index return), per index, and say which index it came from.

v3 approach:
  * Reconciliation uses  ACTUAL saved daily portfolio returns
    (daily_returns_{lstm_mvo,hist_mvo}_walkforward.csv) run through OWN
    metric function (pipeline_io.performance_metrics_daily), so the totals match
    Table 3 exactly.
  * Attribution decomposes the gap on a DAILY basis: for each holding period,
    weights w set at the period's rebalance date multiply that period's daily
    index returns. Summed per index over the whole window, the daily weight-diff
    contributions reconcile to the reconstructed daily return gap (a small
    compounding cross-term remains and is reported explicitly).
"""
import numpy as np
import pandas as pd
import pipeline_io as io


def daily_contributions(daily_idx: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Map each daily date to the weights that were in force (weights set at the
    period's rebalance date d0, held over (d0, d1]). Returns a daily DataFrame of
    per-index contributions w_i * r_i (before costs)."""
    reb = list(weights.index)
    pieces = []
    for i, d0 in enumerate(reb):
        d1 = reb[i + 1] if i + 1 < len(reb) else daily_idx.index[-1]
        win = daily_idx.loc[(daily_idx.index > d0) & (daily_idx.index <= d1)]
        if win.empty:
            continue
        w = weights.loc[d0].values
        pieces.append(win.mul(w, axis=1))
    return pd.concat(pieces).sort_index()


def main():
    daily_idx = io.load_returns()
    w_lstm = io.load_weights("lstm")
    w_cls = io.load_weights("classical")

    pnl_lstm = io.load_pnl("lstm")
    pnl_cls = io.load_pnl("classical")
    m_lstm = io.performance_metrics_daily(pnl_lstm)
    m_cls = io.performance_metrics_daily(pnl_cls)
    print("=== Reconciliation (saved daily P&L through your own metric fn) ===")
    print(f"  LSTM-MVO   ann. return: {m_lstm['AnnReturn']*100:6.2f}%   "
          f"Sharpe {m_lstm['Sharpe']:.3f}   (Table 3: 13.05%, 0.84)")
    print(f"  Classical  ann. return: {m_cls['AnnReturn']*100:6.2f}%   "
          f"Sharpe {m_cls['Sharpe']:.3f}   (Table 3:  8.64%, 0.61)")
    gap = (m_lstm['AnnReturn'] - m_cls['AnnReturn']) * 100
    print(f"  Annualised return gap: {gap:+.2f}pp")
    m_lstm_true = io.performance_metrics_daily(pnl_lstm, io.TRADING_DAYS_TRUE)
    m_cls_true = io.performance_metrics_daily(pnl_cls, io.TRADING_DAYS_TRUE)
    print(f"  [corrected 233-day annualisation] LSTM Sharpe {m_lstm_true['Sharpe']:.3f}, "
          f"Classical Sharpe {m_cls_true['Sharpe']:.3f}")

    
    c_lstm = daily_contributions(daily_idx, w_lstm)   # daily w_i*r_i for LSTM
    c_cls = daily_contributions(daily_idx, w_cls)     # daily w_i*r_i for classical
    common = c_lstm.index.intersection(c_cls.index)
    diff = (c_lstm.loc[common] - c_cls.loc[common])   # daily per-index gap contribution

    # sum of daily contributions per index (arithmetic); compounding cross-term reported
    per_index_arith = diff.sum() * 100
    total_arith = per_index_arith.sum()
    # reconstructed daily gap (sum over indices, arithmetic) vs geometric gap
    recon_daily_gap = (c_lstm.loc[common].sum(axis=1) - c_cls.loc[common].sum(axis=1)).sum() * 100

    print("\n=== Per-index contribution to the return gap (arithmetic, pp) ===")
    for a in io.ASSETS:
        print(f"  {a:12s}: {per_index_arith[a]:+6.2f} pp")
    print(f"  {'TOTAL':12s}: {total_arith:+6.2f} pp   "
          f"(arith. daily gap {recon_daily_gap:+.2f}pp; geometric gap {gap:+.2f}pp)")
    print("  (Difference between arithmetic total and geometric gap = compounding "
          "cross-term; report it as a residual.)")

    drivers = per_index_arith.abs().sort_values(ascending=False)
    top = drivers.index[0]
    share = per_index_arith[top] / total_arith * 100 if total_arith != 0 else float('nan')
    print(f"\n  Headline: {share:.0f}% of the gap is attributable to differential "
          f"exposure to {top} ({per_index_arith[top]:+.2f}pp of {total_arith:+.2f}pp).")

    diff.to_csv(io.OUTPUT_DIR / "s2_attribution_daily.csv")
    per_index_arith.to_csv(io.OUTPUT_DIR / "s2_attribution_perindex.csv")


if __name__ == "__main__":
    main()