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
plt.savefig("fig1_cumulative_returns.png", dpi=200)
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
plt.legend()
plt.tight_layout()
plt.savefig("fig2_drawdown.png", dpi=200)
plt.close()

# --- Figure 3: LSTM-MVO monthly weight allocation ---
plt.figure(figsize=(7, 4.2))
w = weights_df["lstm_mvo"]
plt.stackplot(w.index, w["STOXX1800"], w["RUSSELL1000"], w["SHANGHAI_A"],
              labels=["STOXX1800", "RUSSELL1000", "SHANGHAI_A"], alpha=0.85)
plt.title("LSTM-MVO Monthly Portfolio Weights")
plt.xlabel("Date")
plt.ylabel("Weight")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("fig3_lstm_weights.png", dpi=200)
plt.close()

print("Figures saved.")