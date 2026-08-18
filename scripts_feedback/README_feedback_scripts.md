# Marker-feedback analysis scripts

These scripts address the empirical items in the August 2026 marker feedback.
They are written to slot into your existing pipeline (01–07) with minimal change.
Each one is self-contained and prints a small results table you can paste into
the paper, plus saves a CSV/PNG under `output/`.

## What you need to wire up (once)

Every script imports a tiny shim, `pipeline_io.py`, that is the ONLY place that
knows how your data and backtest live. Edit `pipeline_io.py` so its functions
return your real objects. Everything else then just works. Placeholder logic is
clearly marked with `# >>> EDIT ME`.

The functions the scripts rely on:

- `load_returns()` -> DataFrame of daily simple returns, columns
  ['STOXX1800','RUSSELL1000','SHANGHAI_A'], DatetimeIndex, aligned (your 3,267 rows).
- `load_lstm_mu()` -> DataFrame of the LSTM's expected-return vector used at each
  rebalance date (index = rebalance dates, same 3 columns), walk-forward version.
- `load_weights(strategy)` -> DataFrame of portfolio weights actually held, index =
  rebalance dates, 3 columns. strategy in {'lstm','classical','equal'}.
- `load_pnl(strategy)` -> Series of periodic (e.g. monthly) portfolio returns,
  net of costs, for the backtest window. Used for Sharpe-difference tests.
- `mvo_backtest(mu_provider, cov_provider=None, rebalance='M', cost_bps=5)` ->
  dict of metrics + a P&L Series. This is your existing 03b engine wrapped so a
  script can pass an arbitrary mu. If wrapping is awkward, each script explains
  the fallback.

## Run order (cheapest / highest-impact first)

1. `s1_static_tilt.py`      – fixed-average-weight + buy-and-hold rows  (Feedback: static tilt)
2. `s2_return_attribution.py` – where the +3.68pp came from            (Feedback: attribution)
3. `s3_sharpe_test.py`      – Jobson–Korkie/Memmel + Ledoit–Wolf bootstrap (Feedback: significance)
4. `s4_placebos.py`         – random-mu / momentum-mu / minvar / invvol (Feedback: controls)
5. `s5_linear_benchmarks.py`– OLS, ridge, AR(1) vs LSTM                (Feedback: linear benchmark)
6. `s6_seed_sweep.py`       – 5–10 seeds, Sharpe distribution, ablation (Feedback: overfitting/seeds)
7. `s7_inference_se.py`     – Newey–West / block-bootstrap SEs          (Feedback: overlapping targets)
8. `s8_data_diagnostics.py` – correlation matrix, per-index ret/vol, ann. factor (Feedback: data)

All are independent except s2 needs weights (from your pipeline) and s3 needs P&L series.
