"""
s6_seed_sweep.py  (v2 — reuses your real training + backtest)
Feedback: run the pipeline over 5-10 seeds; report mean & range of the Sharpe.
If it moves by >~0.1, report a distribution not a point estimate. Add a 4/8
hidden-unit capacity ablation. Also prints the LSTM parameter count vs
independent-block count for the overfitting discussion.

This re-runs LSTM training (02b logic) per seed. It is the SLOWEST script
"""
import sys, os
import numpy as np
import pandas as pd
import torch
from importlib import import_module

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts_feedback"))
import pipeline_io as io
dp = import_module("01_data_prep")
m2 = import_module("02_lstm_model")
b2 = import_module("02b_lstm_walkforward")
bt = import_module("03_mvo_backtest")

SEEDS = [0, 1, 2, 3, 4]        
HIDDEN_ABLATION = [4, 8, 32]


def param_count(hidden=32, n_features=15, n_assets=3):
    p = 4 * (hidden * (hidden + n_features) + hidden)   # LSTM gates
    p += hidden * n_assets + n_assets                    # linear head
    return p


def run_once(seed, hidden):
    """Reproduce 02b walk-forward predictions for a given seed+hidden, then run the
    03b-style backtest and return the LSTM-MVO Sharpe. Reuses your own functions so
    the pipeline is identical apart from seed/hidden."""
    torch.manual_seed(seed); np.random.seed(seed)

    returns = dp.load_index_returns()
    panel = m2.build_panel(returns)
    target_cols = [f"{c}_return" for c in returns.columns]
    X_all, y_all, dates_all = m2.make_windows(panel, target_cols, m2.WINDOW, m2.HORIZON)
    n_assets = y_all.shape[-1]

    q_starts = pd.date_range(b2.TEST_START, b2.TEST_END, freq="QS")
    preds, dts = [], []
    for qs in q_starts:
        qe = qs + pd.offsets.QuarterEnd(0)
        tr = dates_all < np.datetime64(qs)
        te = (dates_all >= np.datetime64(qs)) & (dates_all <= np.datetime64(qe))
        if te.sum() == 0:
            continue
        # train with overridden hidden size (patch ReturnLSTM default via kwarg)
        Xtr, ytr = X_all[tr], y_all[tr]
        mu = Xtr.mean(axis=(0,1)); sig = Xtr.std(axis=(0,1)) + 1e-8
        model = m2.ReturnLSTM(n_features=Xtr.shape[-1], n_assets=n_assets, hidden=hidden)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = torch.nn.MSELoss()
        from torch.utils.data import DataLoader
        dl = DataLoader(m2.ReturnDataset((Xtr-mu)/sig, ytr), batch_size=64, shuffle=True)
        model.train()
        for _ in range(15):
            for xb, yb in dl:
                opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(torch.tensor((X_all[te]-mu)/sig, dtype=torch.float32)).numpy()
        preds.append(p); dts.append(dates_all[te])

    pred_df = pd.DataFrame(np.concatenate(preds), index=np.concatenate(dts),
                           columns=returns.columns).sort_index()

    # 03b-style backtest using these predictions -> LSTM-MVO daily P&L -> Sharpe
    me = pred_df.resample("ME").last().index
    me = me[(me >= pred_df.index.min()) & (me <= pred_df.index.max())]
    prev = np.ones(3)/3; chunks = []
    for i in range(len(me)-1):
        d0, d1 = me[i], me[i+1]
        hist = returns.loc[:d0].tail(252); cov = hist.cov().values*252
        mu_l = pred_df.loc[:d0].tail(21).mean().values * (252/5)
        w = bt.mvo_weights(mu_l, cov)
        period = returns.loc[(returns.index>d0)&(returns.index<=d1)]
        cost = np.abs(w-prev).sum()*(bt.TXN_COST_BPS/10000)
        r = period.values @ w
        if len(r)>0: r[0]-=cost
        chunks.append(pd.Series(r, index=period.index)); prev=w
    pnl = pd.concat(chunks).sort_index()
    return io.performance_metrics_daily(pnl)["Sharpe"]


def main():
    print("=== Overfitting counts ===")
    print(f"  LSTM params (hidden=32): ~{param_count(32):,}")
    print(f"  Independent 5-day blocks in test: ~186 (from s7). "
          f"Parameter-to-effective-sample ratio is high -> overfitting expected.\n")

    print("=== Seed sweep (hidden=32) ===")
    sh = []
    for s in SEEDS:
        v = run_once(s, 32); sh.append(v)
        print(f"  seed {s}: LSTM-MVO Sharpe {v:.3f}")
    sh = np.array(sh)
    print(f"\n  mean {sh.mean():.3f}, range [{sh.min():.3f}, {sh.max():.3f}], "
          f"spread {sh.max()-sh.min():.3f}")
    if sh.max()-sh.min() > 0.1:
        print("  >>> Spread > 0.1: report Sharpe as a DISTRIBUTION, not a point estimate.")
    else:
        print("  >>> Spread <= 0.1: the 0.84 point estimate is reasonably seed-stable.")

    print("\n=== Capacity ablation (seed 0) ===")
    for h in HIDDEN_ABLATION:
        v = run_once(0, h)
        print(f"  hidden={h:2d}: Sharpe {v:.3f}  (params ~{param_count(h):,})")

    pd.DataFrame({"seed": SEEDS, "sharpe": sh}).to_csv(io.OUTPUT_DIR / "s6_seed_sweep.csv", index=False)


if __name__ == "__main__":
    main()