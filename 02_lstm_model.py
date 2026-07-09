"""
02_lstm_model.py
A deliberately simple LSTM: one shared network takes a rolling window of
returns + engineered features for all three indices and predicts next-day
returns for all three simultaneously. Kept small (1 layer, 32 units) so it
trains in minutes on CPU and stays easy to explain at viva.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from importlib import import_module
dp = import_module("01_data_prep")

WINDOW = 60          # trading days of history fed to the LSTM
HORIZON = 5          # predict cumulative return over the next 5 trading days (smoother signal)
TRAIN_END = "2018-12-31"   # everything after this is the test / backtest period

torch.manual_seed(42)
np.random.seed(42)


class ReturnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ReturnLSTM(nn.Module):
    """1-layer LSTM -> linear head predicting next-day return per index."""
    def __init__(self, n_features, n_assets, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(hidden, n_assets)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # representation at final timestep
        return self.head(self.dropout(last))


def build_panel(returns: pd.DataFrame) -> pd.DataFrame:
    """Concatenate return + rolling features for all indices into one feature matrix."""
    feats = dp.engineer_features(returns)
    common_idx = None
    panels = []
    for name, feat in feats.items():
        feat = feat.add_prefix(f"{name}_")
        panels.append(feat)
        common_idx = feat.index if common_idx is None else common_idx.intersection(feat.index)
    panel = pd.concat([p.loc[common_idx] for p in panels], axis=1).sort_index()
    return panel


def make_windows(panel: pd.DataFrame, target_cols, window=WINDOW, horizon=HORIZON):
    X, y, dates = [], [], []
    values = panel.values
    raw_returns = panel[target_cols].values
    cum = pd.DataFrame(raw_returns, columns=target_cols)
    fwd_cum = (1 + cum).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
    fwd_cum = fwd_cum.values

    for i in range(window, len(panel) - horizon):
        if np.isnan(fwd_cum[i]).any():
            continue
        X.append(values[i - window:i])
        y.append(fwd_cum[i])
        dates.append(panel.index[i])
    return np.array(X), np.array(y), pd.DatetimeIndex(dates)


def train_lstm():
    returns = dp.load_index_returns()
    panel = build_panel(returns)
    target_cols = [f"{name}_return" for name in returns.columns]

    X, y, dates = make_windows(panel, target_cols)
    train_mask = dates <= pd.Timestamp(TRAIN_END)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    dates_test = dates[~train_mask]

    mu, sigma = X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1)) + 1e-8
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    model = ReturnLSTM(n_features=X.shape[-1], n_assets=y.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_ds = ReturnDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        print(f"epoch {epoch+1}/{epochs}  train_mse={total_loss/len(train_ds):.6f}")

    model.eval()
    with torch.no_grad():
        preds_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    pred_df = pd.DataFrame(preds_test, index=dates_test, columns=returns.columns)
    actual_df = pd.DataFrame(y_test, index=dates_test, columns=returns.columns)

    pred_df.to_csv("lstm_predictions.csv")
    actual_df.to_csv("lstm_actuals.csv")

    for col in returns.columns:
        mse_model = ((pred_df[col] - actual_df[col]) ** 2).mean()
        mse_naive = (actual_df[col] ** 2).mean()
        dir_acc = (np.sign(pred_df[col]) == np.sign(actual_df[col])).mean()
        print(f"{col}: MSE(model)={mse_model:.6e}  MSE(naive-zero)={mse_naive:.6e}  "
              f"directional_accuracy={dir_acc:.3f}")

    return pred_df, actual_df


if __name__ == "__main__":
    train_lstm()