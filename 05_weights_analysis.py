"""
05_weights_analysis.py
Compares portfolio weight behaviour between LSTM-MVO and Classical MVO under
the walk-forward backtest, to help explain why LSTM-MVO's Sharpe improved
substantially (0.69 -> 0.84) despite directional accuracy staying near chance.

Hypothesis: LSTM-MVO produces more stable, less concentrated weights than
Classical MVO (which relies on noisy trailing historical means), leading to
lower turnover / transaction costs and less exposure to concentrated bets
during volatile periods (e.g. the 2020 COVID crash).

Outputs:
  figures/fig4_weights_comparison.png   - stacked weights over time, both strategies
  data/processed/walkforward/weights_stability_summary.csv  - turnover & concentration stats
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import import_module

bt = import_module("03_mvo_backtest")
TXN_COST_BPS = bt.TXN_COST_BPS

WEIGHTS_DIR = "data/processed/walkforward"
ASSETS = ["STOXX1800", "RUSSELL1000", "SHANGHAI_A"]

STRATEGIES = {
    "lstm_mvo": "LSTM-MVO",
    "hist_mvo": "Classical MVO",
}


def load_weights(name):
    path = f"{WEIGHTS_DIR}/weights_{name}_walkforward.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def turnover_stats(w: pd.DataFrame) -> dict:
    """Average and max month-to-month turnover: sum of |change| in weights.
    Also reports total transaction cost drag over the whole backtest, using
    the same TXN_COST_BPS applied per unit of turnover as in the backtest
    scripts (cost = turnover * TXN_COST_BPS / 10000, charged each rebalance)."""
    diffs = w.diff().abs().sum(axis=1).dropna()
    total_cost = (diffs * (TXN_COST_BPS / 10000)).sum()
    n_years = (w.index[-1] - w.index[0]).days / 365.25
    return {
        "avg_turnover": diffs.mean(),
        "max_turnover": diffs.max(),
        "total_txn_cost_pct": total_cost * 100,       # cumulative cost drag, in %
        "avg_annual_txn_cost_pct": (total_cost / n_years) * 100,  # annualised, in %
    }


def concentration_stats(w: pd.DataFrame) -> dict:
    """Average and max single-asset weight, as a measure of concentration."""
    max_w_per_period = w.max(axis=1)
    return {"avg_max_weight": max_w_per_period.mean(), "peak_max_weight": max_w_per_period.max()}


def main():
    weights = {name: load_weights(name) for name in STRATEGIES}

    summary_rows = []
    for name, label in STRATEGIES.items():
        w = weights[name]
        row = {"strategy": label}
        row.update(turnover_stats(w))
        row.update(concentration_stats(w))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("strategy")
    print("Weight stability / concentration summary (walk-forward, 2019-2022):")
    print(summary_df.round(4).to_string())
    summary_df.to_csv(f"{WEIGHTS_DIR}/weights_stability_summary.csv")

    # --- Plot: one subplot per asset, LSTM-MVO vs Classical MVO as separate
    # lines. This is far easier to read than a stacked-area chart for
    # comparing two strategies, since each asset's weight over time is
    # traced directly rather than inferred from a filled region's thickness.
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(9, 8), sharex=True)
    colors = {"lstm_mvo": "#1f77b4", "hist_mvo": "#ff7f0e"}
    styles = {"lstm_mvo": "-", "hist_mvo": "--"}

    for ax, asset in zip(axes, ASSETS):
        for name, label in STRATEGIES.items():
            w = weights[name]
            ax.plot(w.index, w[asset], label=label, color=colors[name],
                     linestyle=styles[name], linewidth=1.8)
        ax.set_title(asset, fontsize=11, loc="left")
        ax.set_ylabel("Weight")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    fig.suptitle("Portfolio Weight by Asset: LSTM-MVO vs Classical MVO (Walk-Forward)", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/fig4_weights_comparison.png", dpi=200)
    plt.close()

    print("\nSaved figures/fig4_weights_comparison.png")
    print(f"Saved {WEIGHTS_DIR}/weights_stability_summary.csv")


if __name__ == "__main__":
    main()