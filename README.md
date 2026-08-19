# LSTM-MVO Dissertation - Supporting Material

**Author:** Sakib Islam (250994104)
**Programme:** MSc Data Science, Queen Mary University of London
**Supervisor:** Dr Yongxin Yang
**Project:** Deep Learning-Enhanced Portfolio Optimisation: LSTM-Based Return Prediction with Mean-Variance Allocation

This repository contains all source code and processed outputs required to reproduce
every table and figure in the dissertation research paper.

---

## 1. Environment

- **Language:** Python 3.13
- **Key libraries:** pandas, NumPy, PyTorch, scipy, scikit-learn, matplotlib, Plotly
- **OS used:** Windows (VS Code + venv)

### Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install pandas numpy torch scipy scikit-learn matplotlib plotly
```

---

## 2. Repository layout

```
Final Year Project/
├── index_data/                      # raw supplied index return series (.json.bz2)
│   ├── STOXX1800.json.bz2
│   ├── RUSSELL1000.json.bz2
│   └── SHANGHAI_A.json.bz2
├── data/processed/                  # generated intermediate + result files
│   ├── returns_aligned.csv
│   ├── backtest_metrics.csv                 # single-split metrics
│   ├── lstm_predictions.csv / lstm_actuals.csv
│   ├── weights_{lstm_mvo,hist_mvo,equal}.csv
│   └── walkforward/                          # walk-forward outputs
│       ├── backtest_metrics_walkforward.csv
│       ├── daily_returns_{lstm_mvo,hist_mvo,equal}_walkforward.csv
│       ├── weights_{lstm_mvo,hist_mvo}_walkforward.csv
│       └── lstm_predictions_walkforward.csv
├── figures/                         # generated figures
├── 01_data_prep.py … 07_*.py        # core pipeline
├── scripts_feedback/                # additional analysis (marker-feedback response)
│   ├── pipeline_io.py               # shared data/metrics loader
│   └── s1_*.py … s9_*.py
└── README.md                        # this file
```

---

## 3. Core pipeline - run in order

| Step | Script | Produces |
|------|--------|----------|
| 1 | `01_data_prep.py` | `data/processed/returns_aligned.csv` (aligned daily returns + features) |
| 2 | `02_lstm_model.py` | single-split LSTM predictions |
| 2b | `02b_lstm_walkforward.py` | walk-forward LSTM predictions (`walkforward/lstm_predictions_walkforward.csv`) |
| 3 | `03_mvo_backtest.py` | single-split backtest (`backtest_metrics.csv`, weights, daily returns) |
| 3b | `03b_mvo_backtest_walkforward.py` | walk-forward backtest (`walkforward/*`) |
| 3c | `03c_mvo_backtest_biweekly.py` | bi-weekly rebalancing robustness check (§4.4 / Table IV) |
| 4 | `04_plots.py` | Figures 1–4 (growth, drawdown, weights) |
| 5 | `05_weights_analysis.py` | turnover / transaction-cost analysis (§5) |
| 6 | `06_interactive_growth_chart.py` | interactive Plotly growth chart (HTML) |
| 7 | `07_portfolio_dashboard.py` | interactive combined dashboard (HTML) |

Run each from the project root, e.g. `python 01_data_prep.py`.

---

## 4. Additional analysis scripts (marker-feedback response)

All in `scripts_feedback/`. **Run from the project root** (they import the core
pipeline modules). `pipeline_io.py` is the shared loader and does not need editing -
it is already wired to the file paths above.

| Script | Addresses | Regenerates |
|--------|-----------|-------------|
| `s1_static_tilt.py` | Static-tilt critique | Table V (fixed-avg-weight + buy-and-holds) |
| `s2_return_attribution.py` | Return attribution | Per-index attribution of the LSTM–classical gap (§4.6) |
| `s3_sharpe_test.py` | Significance testing | Jobson–Korkie/Memmel + Ledoit–Wolf p-values (§4.3) |
| `s4_placebos.py` | Placebo controls | Random-μ, momentum-μ, min-variance, inverse-vol (§4.6) |
| `s5_linear_benchmarks.py` | Linear benchmark | OLS / Ridge / AR(1) diagnostics (§4.1) |
| `s6_seed_sweep.py` | Overfitting / seed stability | Seed distribution + capacity ablation (§4.6) |
| `s7_inference_se.py` | Overlapping-target inference | HAC / bootstrap SEs for Table I (§4.1) |
| `s8_data_diagnostics.py` | Data framing | Correlation matrix, per-index return/vol, annualisation factor (§3.1) |
| `s9_training_curve.py` | Overfitting evidence | Figure 5 (training-loss curve, §3.3) |

Example:

```bash
python scripts_feedback/s1_static_tilt.py
python scripts_feedback/s8_data_diagnostics.py
```

Note: `s4`, `s5` and `s6` retrain models and are slower; `s6` (5 seeds + ablation)
is the slowest (several minutes on CPU).

---

## 5. Reproducing each paper element

| Paper element | How to reproduce |
|---------------|------------------|
| §3.1 data diagnostics (corr 0.94, per-index ret/vol) | `s8_data_diagnostics.py` |
| Figure 5 (training loss) | `s9_training_curve.py` |
| Table I (LSTM diagnostics + HAC SEs) | `02b` then `s7_inference_se.py`; linear rows from `s5` |
| Figures 1–4 | `04_plots.py` |
| Table II (single-split backtest) | `03_mvo_backtest.py` → `backtest_metrics.csv` |
| Table III (walk-forward backtest) | `03b_mvo_backtest_walkforward.py` → `backtest_metrics_walkforward.csv` |
| Table IV (bi-weekly robustness) | `03c_mvo_backtest_biweekly.py` |
| §4.3 significance tests | `s3_sharpe_test.py` |
| Table V (static-tilt controls) | `s1_static_tilt.py` |
| §4.6 return attribution | `s2_return_attribution.py` |
| §4.6 placebos | `s4_placebos.py` |
| §4.6 seed stability + ablation | `s6_seed_sweep.py` |

---

## 6. Generative AI usage

Generative AI (Claude, Anthropic) was used for code scaffolding/debugging, for
organising examiner feedback, and for drafting/editing prose. All experimental
results, numerical values, tables and figures were produced by the author's own
code on the author's own data and verified by the author. See Appendix A of the
research paper for the full signed accountability statement.

---

## 7. Notes and known limitations

- Annualisation uses 252 trading days for comparability; the aligned three-market
  calendar averages ~234 days/year (see §3.5).
- The walk-forward Sharpe is seed-sensitive (range 0.45–0.91 over five seeds);
  the mean (0.72) is the more representative figure (see §4.6).
- Hyperparameters were not selected on a separate validation split (see §3.3).
