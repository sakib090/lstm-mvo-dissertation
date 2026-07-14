import pandas as pd
import matplotlib.pyplot as plt
from importlib import import_module

bt = import_module("03_mvo_backtest")

results, weights_df = bt.run_backtest()

# --- Figure 1: cumulative return curves ---
plt.figure(figsize=(7, 4.2))
labels = {"lstm_mvo": "LSTM-MVO", "hist_mvo": "Classical MVO", "equal": "Equal-Weight"}
for name, r in results.items():
    cum = (1 + r).cumprod()
    plt.plot(cum.index, cum.values, label=labels[name], linewidth=1.4)
plt.title("Cumulative Portfolio Growth (2019\u20132022 out-of-sample)")
plt.xlabel("Date")
plt.ylabel("Growth of \u00a31 invested")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig1_cumulative_returns.png", dpi=200)
plt.close()

# --- Figure 2: drawdown ---
plt.figure(figsize=(7, 4.2))
for name, r in results.items():
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    plt.plot(dd.index, dd.values, label=labels[name], linewidth=1.2)
plt.title("Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("figures/fig2_drawdown.png", dpi=200)
plt.close()

# --- Figure 3: LSTM-MVO monthly weight allocation ---
# Redesigned as small multiples (one panel per asset) rather than a stacked
# area chart -- the sharp monthly pivots in an unconstrained MVO weight path
# read as solid color blocks when stacked, making it hard to trace any one
# asset's actual weight over time. A separate line per asset is much clearer,
# and keeps this figure visually consistent with fig4_weights_comparison.png.
ASSETS = ["STOXX1800", "RUSSELL1000", "SHANGHAI_A"]
w = weights_df["lstm_mvo"]

fig, axes = plt.subplots(len(ASSETS), 1, figsize=(7, 6), sharex=True)
for ax, asset in zip(axes, ASSETS):
    ax.step(w.index, w[asset], where="post", color="#1f77b4", linewidth=1.8)
    ax.set_title(asset, fontsize=11, loc="left")
    ax.set_ylabel("Weight")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Date")
fig.suptitle("LSTM-MVO Monthly Portfolio Weights", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig3_lstm_weights.png", dpi=200)
plt.close()

print("Figures saved.")