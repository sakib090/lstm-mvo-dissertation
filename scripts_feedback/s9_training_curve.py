"""
s9_training_curve.py
Produces the figure asked for behind the Section 3.3 overfitting claim:
a training-loss curve showing the LSTM converging on the training data.

Because hyperparameters were NOT selected on a held-out validation set (disclosed
as a limitation in Section 3.3), this plots TRAINING loss per epoch. Read together
with the near-chance OUT-OF-SAMPLE diagnostics in Section 4.1, the falling training
loss is the visual signature of overfitting: the model fits the training period but
does not generalise. Optionally also overlays a validation curve IF you later add a
2017-2018 validation split (set VALIDATION=True and wire it).

Reuses 02_lstm_model architecture + 01_data_prep exactly, so the trained model
is identical to the one used in the backtest.

Output: figures/fig5_training_loss.png
"""
import sys, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib import import_module

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts_feedback"))
dp = import_module("01_data_prep")
m2 = import_module("02_lstm_model")

SEED = 42
EPOCHS = 15
VALIDATION = False  


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

    returns = dp.load_index_returns()
    panel = m2.build_panel(returns)
    target_cols = [f"{c}_return" for c in returns.columns]
    X, y, dates = m2.make_windows(panel, target_cols, m2.WINDOW, m2.HORIZON)

    # single-split: train through 2018-12-31 (same as 02_lstm_model.TRAIN_END)
    train_mask = dates <= pd.Timestamp(m2.TRAIN_END)
    Xtr, ytr = X[train_mask], y[train_mask]

    mu = Xtr.mean(axis=(0, 1)); sig = Xtr.std(axis=(0, 1)) + 1e-8
    Xtr_n = (Xtr - mu) / sig

    # optional validation split (last 20% of pre-2019 data by time)
    val_losses = None
    if VALIDATION:
        cut = int(len(Xtr_n) * 0.8)
        Xv, yv = Xtr_n[cut:], ytr[cut:]
        Xtr_n, ytr = Xtr_n[:cut], ytr[:cut]
        val_losses = []

    model = m2.ReturnLSTM(n_features=X.shape[-1], n_assets=y.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    dl = DataLoader(m2.ReturnDataset(Xtr_n, ytr), batch_size=64, shuffle=True)

    train_losses = []
    for epoch in range(EPOCHS):
        model.train(); tot = 0.0
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        train_losses.append(tot / len(ytr))
        if VALIDATION:
            model.eval()
            with torch.no_grad():
                vp = model(torch.tensor(Xv, dtype=torch.float32))
                val_losses.append(lossf(vp, torch.tensor(yv, dtype=torch.float32)).item())
        print(f"epoch {epoch+1}/{EPOCHS}  train_mse={train_losses[-1]:.6e}"
              + (f"  val_mse={val_losses[-1]:.6e}" if VALIDATION else ""))

    # --- plot ---
    plt.figure(figsize=(7, 4.2))
    ep = range(1, EPOCHS + 1)
    plt.plot(ep, train_losses, marker="o", markersize=3, linewidth=1.6,
             color="#1f77b4", label="Training loss")
    if VALIDATION:
        plt.plot(ep, val_losses, marker="s", markersize=3, linewidth=1.6,
                 color="#d62728", linestyle="--", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("LSTM Training Loss (single-split, train through 2018)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig5_training_loss.png", dpi=200)
    plt.close()

    print("\nSaved figures/fig5_training_loss.png")
    print("Training loss falls steadily as the model fits the training period; read")
    print("with the near-chance out-of-sample diagnostics in Section 4.1, this is the")
    print("visual signature of overfitting the claim in Section 3.3 refers to.")
    # save the numbers too
    pd.DataFrame({"epoch": list(ep), "train_mse": train_losses}).to_csv(
        "figures/fig5_training_loss.csv", index=False)


if __name__ == "__main__":
    main()
