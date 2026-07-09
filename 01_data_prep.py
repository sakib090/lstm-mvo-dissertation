"""
01_data_prep.py
Loads the three index return series (STOXX1800, RUSSELL1000, SHANGHAI_A),
aligns trading calendars, and engineers rolling features used as LSTM inputs.
"""
import pandas as pd
import numpy as np

INDICES = {
    "STOXX1800": "index_data/STOXX1800.json.bz2",
    "RUSSELL1000": "index_data/RUSSELL1000.json.bz2",
    "SHANGHAI_A": "index_data/SHANGHAI_A.json.bz2",
}

def load_index_returns():
    series = {}
    for name, path in INDICES.items():
        df = pd.read_json(path, compression="bz2", orient="index")
        s = df["return"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        series[name] = s
    returns = pd.DataFrame(series)
    # Restrict to the 2009-2022 window used in the Project Definition
    returns = returns.loc["2009-01-01":"2022-12-31"]
    # Keep only dates where all three indices traded (aligned calendar)
    returns = returns.dropna(how="any")
    return returns

def engineer_features(returns: pd.DataFrame) -> dict:
    """Builds a feature panel per index: rolling mean, rolling vol, momentum."""
    features = {}
    for col in returns.columns:
        r = returns[col]
        feat = pd.DataFrame(index=r.index)
        feat["return"] = r
        feat["roll_mean_21"] = r.rolling(21).mean()
        feat["roll_std_21"] = r.rolling(21).std()
        feat["momentum_63"] = (1 + r).rolling(63).apply(np.prod, raw=True) - 1
        feat["momentum_252"] = (1 + r).rolling(252).apply(np.prod, raw=True) - 1
        features[col] = feat.dropna()
    return features

if __name__ == "__main__":
    returns = load_index_returns()
    print("Aligned daily returns:", returns.shape)
    print(returns.describe())
    feats = engineer_features(returns)
    for k, v in feats.items():
        print(k, v.shape, v.index.min(), v.index.max())
    returns.to_csv("returns_aligned.csv")