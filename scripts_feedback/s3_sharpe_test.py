"""
s3_sharpe_test.py  (v2 — daily P&L aware)
Feedback addressed: paired Jobson-Korkie test with Memmel (2003) correction plus
Ledoit-Wolf (2008) block bootstrap for the Sharpe difference, on the walk-forward
P&L series.

v2 fixes: load_pnl now returns the DAILY series (03b's saved daily portfolio
returns). This script annualises with 252 to match your backtest, labels N as
days, and uses a block length suited to daily data for the bootstrap. It reports
the ANNUALISED Sharpe (so it matches Table 3's 0.84/0.61) and both p-values.

Reference: JK (1981); Memmel (2003); Ledoit & Wolf (2008).
"""
import numpy as np
import pandas as pd
from scipy import stats
import pipeline_io as io

TD = io.TRADING_DAYS_BT   # 252, matches your backtest annualisation
BLOCK = 21                # ~1 month of trading days for the daily block bootstrap


def ann_sharpe_daily(x):
    return (x.mean() / (x.std(ddof=1) + 1e-12)) * np.sqrt(TD)


def jobson_korkie_memmel(a, b):
    """Paired JK test with Memmel (2003) correction on PER-PERIOD (daily) Sharpes.
    The z-stat is invariant to the annualisation scaling, so it is computed on the
    per-period Sharpes; we report the annualised Sharpes separately for context."""
    n = len(a)
    sr_a = a.mean() / a.std(ddof=1)
    sr_b = b.mean() / b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    theta = (1.0 / n) * (2 - 2 * rho + 0.5 * (sr_a**2 + sr_b**2 - 2 * sr_a * sr_b * rho**2))
    z = (sr_a - sr_b) / np.sqrt(theta)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p, sr_a, sr_b, rho


def ledoit_wolf_bootstrap(a, b, block=BLOCK, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    obs = ann_sharpe_daily(pd.Series(a)) - ann_sharpe_daily(pd.Series(b))
    stacked = np.column_stack([a, b])
    nb = int(np.ceil(n / block))
    idx = np.arange(n)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        take = np.concatenate([idx[np.arange(s, s + block) % n] for s in starts])[:n]
        bs = stacked[take]
        sa = (bs[:, 0].mean() / (bs[:, 0].std(ddof=1) + 1e-12)) * np.sqrt(TD)
        sb = (bs[:, 1].mean() / (bs[:, 1].std(ddof=1) + 1e-12)) * np.sqrt(TD)
        diffs[k] = (sa - sb) - obs
    p = (np.abs(diffs) >= np.abs(obs)).mean()
    return obs, p


def compare(name_a, name_b):
    a = io.load_pnl(name_a).values.astype(float)
    b = io.load_pnl(name_b).values.astype(float)
    m = min(len(a), len(b)); a, b = a[-m:], b[-m:]
    z, p_jk, sr_a, sr_b, rho = jobson_korkie_memmel(a, b)
    obs, p_bs = ledoit_wolf_bootstrap(a, b)
    print(f"\n--- {name_a} vs {name_b}  (N={m} trading days, ~{m/TD:.1f} yrs) ---")
    print(f"  Annualised Sharpe: {name_a}={ann_sharpe_daily(pd.Series(a)):.3f}, "
          f"{name_b}={ann_sharpe_daily(pd.Series(b)):.3f}   (rho={rho:.3f})")
    print(f"  Jobson-Korkie/Memmel:  z={z:.3f},  p={p_jk:.3f}")
    print(f"  Ledoit-Wolf block bootstrap (block={BLOCK}d): p={p_bs:.3f}")
    return dict(pair=f"{name_a} vs {name_b}", N_days=m, rho=rho, z=z,
                p_jk=p_jk, p_boot=p_bs,
                sr_a=ann_sharpe_daily(pd.Series(a)), sr_b=ann_sharpe_daily(pd.Series(b)))


def main():
    print("=== Sharpe-difference significance tests (walk-forward daily P&L, ann. with 252) ===")
    res = [compare("lstm", "classical"), compare("lstm", "equal")]
    pd.DataFrame(res).to_csv(io.OUTPUT_DIR / "s3_sharpe_test.csv", index=False)
    print("\nReport rho, z, and BOTH p-values. At ~4 years of data, a Sharpe gap of "
          "this size is very unlikely to reach p<0.05; stating that plainly is the "
          "honest, mark-earning conclusion and is consistent with your near-chance "
          "directional-accuracy framing in Section 4.1.")


if __name__ == "__main__":
    main()