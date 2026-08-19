import numpy as np
import pandas as pd
from pathlib import Path

ASSETS = ["STOXX1800", "RUSSELL1000", "SHANGHAI_A"]

DATA_DIR = Path("data/processed")          
OUTPUT_DIR = Path("output"); OUTPUT_DIR.mkdir(exist_ok=True)

TRADING_DAYS_BT = 252            # matches 03_mvo_backtest.performance_metrics
TRADING_DAYS_TRUE = 233         # empirically correct (s8): 3,267 days / ~14 yrs
TRADING_DAYS = TRADING_DAYS_BT  # default: reproduce  backtest
RF_ANNUAL = 0.0                 # RISK_FREE_DAILY = 0.0  backtest
REBALANCES_PER_YEAR = 12        # monthly 


def performance_metrics_daily(daily: pd.Series, periods_per_year: int = TRADING_DAYS_BT) -> dict:
    """Byte-for-byte match of 03_mvo_backtest.performance_metrics, so scripts
    reconcile exactly against your saved backtest. Pass periods_per_year=
    TRADING_DAYS_TRUE to see the corrected-annualisation version side by side."""
    r = daily
    ann_return = (1 + r).prod() ** (periods_per_year / len(r)) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / (r.std() * np.sqrt(periods_per_year) + 1e-12)
    downside = r[r < 0]
    sortino = (r.mean() * periods_per_year) / (downside.std() * np.sqrt(periods_per_year) + 1e-12)
    cum = (1 + r).cumprod()
    max_dd = ((cum / cum.cummax()) - 1).min()
    return {"AnnReturn": ann_return, "AnnVol": ann_vol, "Sharpe": sharpe,
            "Sortino": sortino, "MaxDD": max_dd, "N": len(r)}


def load_returns() -> pd.DataFrame:
    """Daily simple returns, aligned, columns=ASSETS, DatetimeIndex.

    Wired to your 01_data_prep.py: reads data/processed/returns_aligned.csv if it
    exists; otherwise rebuilds the aligned panel from index_data/*.json.bz2 exactly
    as 01_data_prep.load_index_returns() does.
    """
    csv_path = DATA_DIR / "returns_aligned.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return df[ASSETS]

    # Fallback: rebuild from raw files (mirrors 01_data_prep.load_index_returns)
    raw = {
        "STOXX1800": "index_data/STOXX1800.json.bz2",
        "RUSSELL1000": "index_data/RUSSELL1000.json.bz2",
        "SHANGHAI_A": "index_data/SHANGHAI_A.json.bz2",
    }
    series = {}
    for name, path in raw.items():
        d = pd.read_json(path, compression="bz2", orient="index")
        s = d["return"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        series[name] = s
    returns = pd.DataFrame(series).loc["2009-01-01":"2022-12-31"].dropna(how="any")
    return returns[ASSETS]


def load_lstm_mu() -> pd.DataFrame:
    """Walk-forward LSTM expected-return vectors, index=rebalance dates."""
    raise NotImplementedError("Wire load_lstm_mu() to your saved walk-forward mu.")


WALKFORWARD_DIR = DATA_DIR / "walkforward"   # data/processed/walkforward

# Maps the generic strategy names the scripts use saved-file keys.
_WEIGHT_KEY = {"lstm": "lstm_mvo", "classical": "hist_mvo"}


def load_weights(strategy: str) -> pd.DataFrame:
    """Weights actually held. strategy in {'lstm','classical','equal'}.

    Wired to 05_weights_analysis.py: reads
    data/processed/walkforward/weights_{lstm_mvo|hist_mvo}_walkforward.csv
    (index = rebalance dates, columns = ASSETS). 'equal' is synthesised as a
    constant 1/3 split on the LSTM rebalance schedule.
    """
    if strategy == "equal":
        ref = load_weights("lstm")
        eq = pd.DataFrame(1.0 / len(ASSETS), index=ref.index, columns=ASSETS)
        return eq
    key = _WEIGHT_KEY.get(strategy)
    if key is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Use lstm|classical|equal.")
    path = WALKFORWARD_DIR / f"weights_{key}_walkforward.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[ASSETS]


_PNL_KEY = {"lstm": "lstm_mvo", "classical": "hist_mvo", "equal": "equal"}


def load_pnl(strategy: str) -> pd.Series:
    """Periodic net-of-cost portfolio returns for the backtest window.

    Wired to 03b_mvo_backtest_walkforward.py, which saves the continuous DAILY
    portfolio return series (transaction costs already deducted) to
    data/processed/walkforward/daily_returns_{lstm_mvo|hist_mvo|equal}_walkforward.csv.

    Returns the DAILY series by default. For the Sharpe-difference tests (s3),
    resample to month-end with resample_to_monthly() below if a monthly series
    is preferred; the JK/bootstrap tests work on either as long as both legs use
    the same frequency and REBALANCES_PER_YEAR is set to match.
    """
    key = _PNL_KEY.get(strategy)
    if key is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Use lstm|classical|equal.")
    path = WALKFORWARD_DIR / f"daily_returns_{key}_walkforward.csv"
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.name = strategy
    return s


def resample_to_monthly(daily: pd.Series) -> pd.Series:
    """Compound a daily return series to month-end returns."""
    return (1 + daily).resample("ME").prod() - 1


def load_predictions(horizon: int = 5):
    """Return (pred_df, target_df) aligned on common dates for the walk-forward
    test set. Predictions come from 02b's saved file; the realised target is the
    horizon-day forward cumulative return reconstructed from the aligned returns
    (the same quantity 02b was trained to predict).

    pred_df, target_df: DataFrames indexed by date, columns = ASSETS.
    """
    pred_path = WALKFORWARD_DIR / "lstm_predictions_walkforward.csv"
    pred = pd.read_csv(pred_path, index_col=0, parse_dates=True)[ASSETS]

    daily = load_returns()
    # forward cumulative return over 
    fwd = (1 + daily).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
    common = pred.index.intersection(fwd.dropna().index)
    return pred.loc[common], fwd.loc[common]

def ann_return_geom(monthly: pd.Series) -> float:
    """Geometric annualised return (CAGR) from a monthly return series."""
    g = (1.0 + monthly).prod()
    yrs = len(monthly) / REBALANCES_PER_YEAR
    return g ** (1.0 / yrs) - 1.0


def ann_vol(monthly: pd.Series) -> float:
    return monthly.std(ddof=1) * np.sqrt(REBALANCES_PER_YEAR)


def sharpe(monthly: pd.Series, rf_annual: float = RF_ANNUAL) -> float:
    rf_m = (1 + rf_annual) ** (1 / REBALANCES_PER_YEAR) - 1
    ex = monthly - rf_m
    return (ex.mean() / ex.std(ddof=1)) * np.sqrt(REBALANCES_PER_YEAR)


def sortino(monthly: pd.Series, mar_annual: float = 0.0) -> float:
    mar_m = (1 + mar_annual) ** (1 / REBALANCES_PER_YEAR) - 1
    ex = monthly - mar_m
    downside = ex[ex < 0]
    dd = np.sqrt((downside ** 2).mean())
    return (ex.mean() / dd) * np.sqrt(REBALANCES_PER_YEAR)


def max_drawdown(monthly: pd.Series) -> float:
    equity = (1 + monthly).cumprod()
    peak = equity.cummax()
    return (equity / peak - 1).min()


def metrics_block(monthly: pd.Series, mar_annual: float = 0.0) -> dict:
    return {
        "AnnReturn": ann_return_geom(monthly),
        "AnnVol": ann_vol(monthly),
        "Sharpe": sharpe(monthly),
        "Sortino": sortino(monthly, mar_annual),
        "MaxDD": max_drawdown(monthly),
        "N": len(monthly),
    }