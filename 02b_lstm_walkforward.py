"""
02b_lstm_walkforward.py
Walk-forward variant of the LSTM return predictor.

Instead of one train/test split (train through 2018-12-31, test on
2019-2022 all at once, as in 02_lstm_model.py), this retrains the LSTM
every quarter using an EXPANDING window (all data from 2010 up to that
quarter's start), then predicts only the following quarter out-of-sample.

Rationale (see dissertation Discussion section): markets are non-stationary,
so a single static model risks staleness by the end of a 4-year test period.
Quarterly retraining keeps the model current with recent regimes while
remaining computationally practical (~16 retrains vs. 48 for monthly).

Outputs (used in place of the single-split predictions in 03_mvo_backtest.py):
  data/processed/lstm_predictions_walkforward.csv
  data/processed/lstm_actuals_walkforward.csv
  data/processed/walkforward_quarterly_accuracy.csv   (diagnostic, per quarter)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from importlib import import_module

dp = import_module("01_data_prep")
m2 = import_module("02_lstm_model")

WINDOW = m2.WINDOW      # 60 trading days of history fed to the LSTM
HORIZON = m2.HORIZON    # predict cumulative return over the next 5 trading days
TEST_START = "2019-01-01"
TEST_END = "2022-12-31"

torch.manual_seed(42)
np.random.seed(42)

ReturnDataset = m2.ReturnDataset
ReturnLSTM = m2.ReturnLSTM
build_panel = m2.build_panel
make_windows = m2.make_windows


def train_one_window(X_train, y_train, n_assets, epochs=15):
    """Train a fresh LSTM on one expanding-window slice. Same architecture
    and hyperparameters as the single-split model in 02_lstm_model.py, so
    results are directly comparable."""
    mu = X_train.mean(axis=(0, 1))
    sigma = X_train.std(axis=(0, 1)) + 1e-8
    X_train_norm = (X_train - mu) / sigma

    model = ReturnLSTM(n_features=X_train.shape[-1], n_assets=n_assets)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_ds = ReturnDataset(X_train_norm, y_train)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    return model, mu, sigma


def run_walkforward():
    returns = dp.load_index_returns()
    panel = build_panel(returns)
    target_cols = [f"{name}_return" for name in returns.columns]

    X_all, y_all, dates_all = make_windows(panel, target_cols, window=WINDOW, horizon=HORIZON)
    n_assets = y_all.shape[-1]

    quarter_starts = pd.date_range(TEST_START, TEST_END, freq="QS")

    all_preds, all_actuals, all_dates, quarter_log = [], [], [], []

    for i, q_start in enumerate(quarter_starts):
        q_end = q_start + pd.offsets.QuarterEnd(0)
        train_mask = dates_all < np.datetime64(q_start)
        test_mask = (dates_all >= np.datetime64(q_start)) & (dates_all <= np.datetime64(q_end))

        if test_mask.sum() == 0:
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        dates_test = dates_all[test_mask]

        model, mu, sigma = train_one_window(X_train, y_train, n_assets=n_assets)

        X_test_norm = (X_test - mu) / sigma
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test_norm, dtype=torch.float32)).numpy()

        all_preds.append(preds)
        all_actuals.append(y_test)
        all_dates.append(dates_test)

        dir_acc = (np.sign(preds) == np.sign(y_test)).mean()
        q_label = f"{q_start.year}-Q{(q_start.month - 1) // 3 + 1}"
        quarter_log.append({"quarter": q_label, "n_train": len(X_train),
                             "n_test": len(X_test), "directional_accuracy": dir_acc})
        print(f"[{i+1}/{len(quarter_starts)}] retrain up to {q_start.date()} "
              f"(n_train={len(X_train)})  test={q_label}  dir_acc={dir_acc:.3f}")

    pred_df = pd.DataFrame(np.concatenate(all_preds), index=np.concatenate(all_dates),
                            columns=returns.columns).sort_index()
    actual_df = pd.DataFrame(np.concatenate(all_actuals), index=np.concatenate(all_dates),
                              columns=returns.columns).sort_index()
    acc_df = pd.DataFrame(quarter_log)

    print("\nPer-quarter directional accuracy:")
    print(acc_df.to_string(index=False))

    pred_df.to_csv("data/processed/walkforward/lstm_predictions_walkforward.csv")
    actual_df.to_csv("data/processed/walkforward/lstm_actuals_walkforward.csv")
    acc_df.to_csv("data/processed/walkforward/walkforward_quarterly_accuracy.csv", index=False)

    print("\nOverall (all quarters pooled):")
    for col in returns.columns:
        mse_model = ((pred_df[col] - actual_df[col]) ** 2).mean()
        mse_naive = (actual_df[col] ** 2).mean()
        dir_acc = (np.sign(pred_df[col]) == np.sign(actual_df[col])).mean()
        print(f"{col}: MSE(model)={mse_model:.6e}  MSE(naive-zero)={mse_naive:.6e}  "
              f"directional_accuracy={dir_acc:.3f}")

    return pred_df, actual_df


if __name__ == "__main__":
    run_walkforward()