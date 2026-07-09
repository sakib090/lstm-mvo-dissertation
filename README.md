# LSTM-MVO Pipeline (Draft Stage)

Run in order:
1. `01_data_prep.py` — loads + aligns STOXX1800 / RUSSELL1000 / SHANGHAI_A
   from https://github.com/qmfin/index_data, engineers rolling features.
2. `02_lstm_model.py` — trains the 1-layer LSTM (60-day window, 5-day forward
   target), trains on data up to 2018-12-31, saves predictions on 2019-2022.
3. `03_mvo_backtest.py` — builds LSTM-MVO, classical-MVO and equal-weight
   portfolios, monthly rebalance, 5bps transaction cost, saves metrics.
4. `04_plots.py` — generates the three figures used in the draft paper.

Requires: pandas, numpy, torch, scipy, matplotlib.
Clone the dataset first: `git clone https://github.com/qmfin/index_data.git`
into a folder named `index_data/` alongside these scripts.

## Known limitations (see Discussion/Future Work in the draft paper)
- Single train/test split, not yet walk-forward.
- LSTM directional accuracy is near chance on raw daily returns — the model
  is currently overfitting; treat backtest results as preliminary.
- Feature set is intentionally minimal (4 rolling stats) for this draft.