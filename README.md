# Moneyball: Undervalued Player Finder

Interactive demo for the Inspirit AI Moneyball project. Fits a model of what Major
League Baseball pays for offensive production (OPS) each season, then flags the
hitters producing far more than their salary implies -- the "undervalued players"
Billy Beane built the 2002 Oakland A's around.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## What's here

- `moneyball.py` -- data pipeline: builds one row per qualified hitter-season with
  OBP / SLG / OPS, primary position, salary, and a per-season value model
  (`predicted salary from OPS` minus `actual salary`). Also the roster builder.
- `app.py` -- the Streamlit UI: bargain scatter, top-underpaid table, and a
  "build a starting nine under budget" mode.
- Player data loads from the public Lahman baseball databank
  (`xorq-labs/baseballdatabank` mirror; salary years 1985-2016) at runtime.

## Method, briefly

For each season we fit `log(salary) = m * OPS + c` across all qualified hitters.
A player's value is `predicted salary - actual salary`; a large positive number
means the market underpaid them for their production. It is a correlation-based
teaching model: it ignores defense, park effects, and age, and salary data ends
in 2016.
