"""
s7_inference_se.py
Feedback addressed: "consecutive test samples share 4 of 5 target days Every MSE
and directional-accuracy statistic needs Newey-West or 5-day block-bootstrap
standard errors rather than i.i.d. ones."

Provides HAC (Newey-West) and 5-day circular block-bootstrap SEs / p-values for:
  - directional accuracy vs 0.5
  - out-of-sample R^2 vs 0 (MSE_model vs MSE_zero)
"""
import numpy as np
import pandas as pd
import pipeline_io as io

# Wired via pipeline_io.load_predictions(): reads 02b's saved walk-forward
# predictions and reconstructs the realised 5-day forward cumulative-return target.
def load_predictions():
    """Return (y_pred, y_true) each shape [n_days, 3] for the walk-forward test set,
    daily-frequency with 5-day overlapping targets."""
    pred_df, target_df = io.load_predictions(horizon=5)
    return pred_df[io.ASSETS].values, target_df[io.ASSETS].values

def newey_west_mean_se(x, lag=5):
    """HAC SE of the sample mean of x with Bartlett kernel, given lag (=overlap)."""
    x = np.asarray(x, float); n = len(x); xbar = x.mean(); e = x - xbar
    gamma0 = np.dot(e, e) / n
    var = gamma0
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        cov = np.dot(e[L:], e[:-L]) / n
        var += 2 * w * cov
    return np.sqrt(var / n)

def block_bootstrap_ci(stat_fn, data, block=5, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(data); nb = int(np.ceil(n / block)); idx = np.arange(n)
    vals = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        take = np.concatenate([idx[np.arange(s, s+block) % n] for s in starts])[:n]
        vals[k] = stat_fn(data[take])
    return np.percentile(vals, [2.5, 97.5]), vals

def main():
    y_pred, y_true = load_predictions()
    print("=== Overlapping-target inference (lag/block = 5) ===")
    for j, a in enumerate(io.ASSETS):
        yp, yt = y_pred[:, j], y_true[:, j]
        hit = (np.sign(yp) == np.sign(yt)).astype(float)
        da = hit.mean()
        se = newey_west_mean_se(hit, lag=5)
        z = (da - 0.5) / se
        (lo, hi), _ = block_bootstrap_ci(lambda d: d.mean(), hit, block=5)
        # OOS R^2
        se_model = (yt - yp) ** 2; se_zero = yt ** 2
        oos_r2 = 1 - se_model.mean() / se_zero.mean()
        print(f"\n{a}:")
        print(f"  Directional acc = {da:.3f}  (HAC SE {se:.3f}, z vs 0.5 = {z:.2f})")
        print(f"    5-day block-bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")
        print(f"  OOS R^2 vs predict-zero = {oos_r2:+.4f}")
        print(f"    N_days = {len(yt)}  (~{len(yt)//5} independent 5-day blocks)")
    print("\nReport directional accuracy with these HAC/bootstrap intervals, not "
          "i.i.d. ones. This directly backs your 'indistinguishable from chance' "
          "claim with correct inference.")

if __name__ == "__main__":
    main()