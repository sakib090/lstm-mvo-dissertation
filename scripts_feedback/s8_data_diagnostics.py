"""
s8_data_diagnostics.py
Feedback addressed: "report the 3x3 correlation matrix of daily returns and each
index's standalone annualised return and volatility over 2019-2022 ... your aligned
calendar gives about 233 trading days per year (3,267 over 14 years), not 252."

Fully self-contained once load_returns() is wired. No modelling needed.
"""
import numpy as np
import pandas as pd
import pipeline_io as io

def main():
    daily = io.load_returns()
    n_days = len(daily)
    span_years = (daily.index[-1] - daily.index[0]).days / 365.25
    td_per_year = n_days / span_years
    print("=== Data diagnostics ===")
    print(f"Aligned daily observations: {n_days}")
    print(f"Calendar span: {span_years:.2f} years")
    print(f"Implied trading days/year: {td_per_year:.1f}  "
          f"(you are currently annualising with 252 — flagged by the marker)")
    print(f"  -> use {td_per_year:.0f} in the annualisation factor, or justify 252.")

    test = daily.loc["2019-01-01":"2022-12-31"]
    print("\n3x3 daily-return correlation (2019-2022):")
    print(test.corr().round(3).to_string())

    print("\nStandalone annualised return & volatility (2019-2022):")
    for a in io.ASSETS:
        r = test[a]
        cum = (1 + r).prod()
        yrs = len(r) / td_per_year
        cagr = cum ** (1 / yrs) - 1
        vol = r.std(ddof=1) * np.sqrt(td_per_year)
        print(f"  {a:12s}: CAGR {cagr*100:5.2f}%   Vol {vol*100:5.2f}%")

    corr = test.corr().values
    off = corr[np.triu_indices(3, 1)]
    print(f"\nMax off-diagonal correlation: {off.max():.3f}")
    if off.max() > 0.8:
        print("  >>> Two assets are >0.8 correlated: the feasible long-only Sharpe "
              "spread is narrow, so 0.54–0.69 is unsurprising. State this.")

    test.corr().to_csv(io.OUTPUT_DIR / "s8_correlation.csv")

if __name__ == "__main__":
    main()
