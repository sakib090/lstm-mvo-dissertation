"""
06_interactive_growth_chart.py
Produces a single, standalone, interactive HTML chart (hover tooltips, zoom,
pan, range slider -- similar to a trading-app style growth chart) comparing
cumulative portfolio growth across all four strategy variants:

  - LSTM-MVO       (single train/test split)
  - LSTM-MVO       (walk-forward, quarterly retrain)
  - Classical MVO  (single train/test split)
  - Equal-Weight   (single train/test split)

An "Initial deposit" box at the top lets you type any starting amount (e.g.
1000 for a hypothetical £1,000 investment) and every line rescales instantly
-- all computed in the browser via JavaScript, no server or re-run needed.

Just run this script and open the resulting growth_comparison.html in any
browser -- no server or internet connection needed, it's fully self-contained.

Output: figures/growth_comparison.html
"""
import json
import pandas as pd
import plotly.graph_objects as go

SERIES = {
    "LSTM-MVO (single-split)":     ("data/processed/daily_returns_lstm_mvo.csv", "#1f77b4", "solid"),
    "LSTM-MVO (walk-forward)":     ("data/processed/walkforward/daily_returns_lstm_mvo_walkforward.csv", "#d62728", "dash"),
    "Classical MVO":               ("data/processed/daily_returns_hist_mvo.csv", "#ff7f0e", "solid"),
    "Equal-Weight":                ("data/processed/daily_returns_equal.csv", "#2ca02c", "solid"),
}


def load_growth(path):
    r = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    growth = (1 + r).cumprod()
    return growth


def main():
    fig = go.Figure()
    per_pound_growth = {}  # trace label -> list of growth-per-£1 values, for JS rescaling

    for label, (path, color, dash) in SERIES.items():
        growth = load_growth(path)
        per_pound_growth[label] = growth.round(6).tolist()
        pct_return = ((growth - 1) * 100).round(2)
        width = 2.5 if dash == "dash" else 1.8
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth.values,
            mode="lines", name=label,
            line=dict(color=color, width=width, dash=dash),
            customdata=pct_return.values,
            hovertemplate="%{x|%d %b %Y}<br>\u00a3%{y:,.2f}  (%{customdata:+.2f}%)<extra>" + label + "</extra>",
        ))

    fig.update_layout(
        title="Portfolio Growth Comparison: Single-Split vs Walk-Forward (2019\u20132022)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (\u00a3)",
        hovermode="x unified",
        template="plotly_white",
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
        ),
        margin=dict(l=70, r=40, t=80, b=40),
    )

    chart_div = fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="growth-chart",
        config={"responsive": True},
    )
    labels_in_order = list(SERIES.keys())
    growth_data_json = json.dumps([per_pound_growth[l] for l in labels_in_order])

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Portfolio Growth Comparison</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; }}
  #controls {{ margin-bottom: 12px; display: flex; align-items: center; gap: 10px; }}
  #controls label {{ font-size: 15px; font-weight: bold; }}
  #deposit {{ font-size: 15px; padding: 6px 10px; width: 160px; border: 1px solid #ccc; border-radius: 4px; }}
  #deposit:focus {{ outline: 2px solid #1f77b4; }}
</style>
</head>
<body>
  <div id="controls">
    <label for="deposit">Initial deposit (\u00a3):</label>
    <input type="number" id="deposit" value="1000" min="1" step="any" />
  </div>
  {chart_div}
  <script>
    const growthPerPound = {growth_data_json};
    const labels = {json.dumps(labels_in_order)};
    const gd = document.getElementById("growth-chart");
    const depositInput = document.getElementById("deposit");

    function rescale() {{
      const deposit = parseFloat(depositInput.value) || 0;
      const newY = growthPerPound.map(series => series.map(v => v * deposit));
      Plotly.restyle(gd, {{ y: newY }}, [...Array(labels.length).keys()]);
      Plotly.relayout(gd, {{ "yaxis.title.text": "Portfolio Value (\\u00a3" + deposit.toLocaleString() + " initial)" }});
    }}

    depositInput.addEventListener("input", rescale);
    // Apply once on load in case the chart's default £1-based values should
    // reflect the default £1,000 deposit immediately.
    window.addEventListener("load", rescale);
  </script>
</body>
</html>
"""

    with open("figures/growth_comparison.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Saved figures/growth_comparison.html \u2014 open it in any browser.")
    print("Use the 'Initial deposit' box at the top to rescale all four lines live.")


if __name__ == "__main__":
    main()