# TrendLens


A fast stock trend analysis tool designed to minimize market noise and provide clearer insight into the general trend of price movements.

Fetches 60 days of 5-minute data from Yahoo Finance, builds weighted EMAs for each timeframes, and fits cubic splines using interpolants computed based on EMA analysis.


**Tech Stack:** Python 3.13, SQLite, NumPy, Pandas, Matplotlib, yfinance

## Update Note:
_Version 1.0 — August 2025_
_Updated Version 2.0 — February 2026_
Version 2.0 introduces user-facing APIs, resolves plotting issues, and refactors the database workflow for more efficient access.\
Database and yfinance calls have been significantly reduced to eliminate primary performance bottlenecks. The project structure has also been reorganized, organizing scattered data and improving overall maintainability.

## Installation:
```bash
pip install trendlens
```

## Usage Example:

```python
from trendlens import DataLoader

dl = DataLoader()
dl.visualize("AAPL")
```

One call fetches data, runs the full analysis, and plots everything on a single chart such as:
raw price, sliding average, EMA, daily spline curve, and the long-range trend spline.


---

## What It Does

The core idea: smaller intervals carry too much noise to be useful on their own, but they contain real signal when aggregated properly. 

Instead of picking one timeframe and hoping for the best, this builds upward:

- 5-minute bars => per-bar volatility weights => EMA
- 12 five-minute betas averaged => hourly EMA
- ~8 hourly EMA values => daily cubic spline => one "interpolant" per day
- all daily interpolants => one long cubic spline across the full range

Each stage feeds into the next (cummulative) stage.

The weight at every level adapts to how volatile that interval actually was, so calm periods get trusted more and noisy periods get smoothed harder.

---

## Analysis

### 5-Minute EMA

Each trading day (09:30–15:55) has about 80 five-minute intervals. 

Before running the EMA, every bar gets a weight based on how much of the day's total swing it represents.

First, the day's full range:

```
day_range = max(all highs) - min(all lows)
```

Then each bar accumulates:

```
w[0] = (high[0] - low[0]) / day_range
w[i] = (w[i-1] + (high[i] - low[i])) / day_range
```

This means early bars start with small weights and the weight grows as more volatility accumulates. 

The weight becomes beta (the EMA trust factor) after clamping to [0.001, 0.95]:

```
β[i] = clamp(w[i], 0.001, 0.95)
```
__Note: You may want to reduce the upper bound value to reduce the affect of each sample: the equations I've used was to mainly capture the trend by examining how volatile each sample is compared to the maximum volatility, the idea was that in daily analysis capturing trend is often more important than smoothing the curve.__

On a volatile day, each bar's range is small relative to the total swing, so beta stays low and the EMA moves slowly. On a calm day, individual bars matter more. The EMA itself is standard:

```
ema[i] = (1 - β[i]) * ema[i-1] + β[i] * sample[i]
```

The seed value for the first bar uses a ratio-weighted midpoint: `((1 + close/open) / 2) * ((close + open) / 2)`. After that, the EMA carries over across days.

A 13-bar centered sliding average is computed alongside for visual comparison.

### Hourly EMA

Each hour has 12 five-minute bars. The hourly beta is just their mean:

```
β_hourly[j] = sum(5min_betas[j*12 .. (j+1)*12]) / 12
```

Then the same EMA equation runs on hourly OHLCV bars with these aggregated betas. This gives ~8 data points per trading day.

### Daily Cubic Spline

Those ~8 hourly EMA values per day get connected by a natural cubic spline — a smooth curve that passes through every point without the sharp corners you'd get from linear interpolation.

Each segment between adjacent knots is a cubic polynomial `S(t) = at³ + bt² + ct + d` with t normalized to [0, 1]. Coefficients are solved via the Thomas algorithm with natural boundary conditions (second derivatives zero at the endpoints).

From the daily spline, a single number is extracted to represent the whole day. This uses the spline's arc length (how "wiggly" the curve is) to decide how much to trust the curve vs the simple median:

```
β_day = 1 - ema_range / arc_length
interpolant = (1 - β_day) * median(hourly_ema) + β_day * spline_integral_mean
```


The arc length is computed with 5-point Gauss-Legendre quadrature (hardcoded constants, no scipy dependency). 
The arc length was mainly used to determine the volatility by comparing it with `|arc_length|/|max-min|.` That sort of contains information regarding how much fluctuation happened between the min and max.

The integral mean is analytic: `(1/N) * Σ(a/4 + b/3 + c/2 + d)` per segment.

so (1 - β_day) in this context gives how much we should trust the median while β_day shows how much we should trust mean.

A straight day (range ≈ arc length) gets β_day near 0, so the interpolant is mostly the median. A curvy day (range << arc length) gets β_day near 1, trusting the integral mean more.

### Long Spline

All daily interpolants across the full date range (typically ~40 trading days) get collected and connected by one more cubic spline. If there's a gap longer than 5 calendar days (holidays, missing data), the spline splits at that gap and each contiguous segment gets its own curve.

This is the macro trend line — it shows where the price was "trying to go" on a day-to-day basis, with all the intraday noise already removed by the stages above.

---

## Database Management

Three separate SQLite files, each chosen for how consistent the data needs to be.

Each database is organized into separate files based on specific criteria:

- **Consistency requirements:** Data that is tightly coupled and must remain synchronized is grouped together to prevent cascading inconsistencies if one dataset becomes invalid.
- **Analysis Phase:** Databases are separated according to the required analysis interval (e.g., 5-minute, hourly, daily), allowing optimized storage and querying strategies for each timeframe.
- **Separation of concerns:** Core functions are designed to interact with the database only once per operation. Interface layers handle data aggregation, transformation, and cumulative computations before storing the processed results into dedicated databases.

**`ticker_list.db`** — just a list of validated ticker symbols. Cheap to query, rarely changes.\
When a ticker is requested for the first time, it gets validated against Yahoo Finance and stored here. Every subsequent call skips the yfinance round-trip entirely.

**`stock_price.db`** — raw 5-minute OHLCV. 
Written once when data is fetched, read many times during analysis. 
The fetcher checks what date range already exists locally before hitting Yahoo Finance, so it only pulls what's missing.

**`analysis.db`** — all computed results (5min EMA, hourly EMA, spline coefficients, per-day metrics). Gets dropped and rebuilt on each analysis run. This is intentional: analysis is deterministic given the raw data, so there's no point in careful incremental updates. Cheaper to just recompute.

### Validation

Every user-facing string passes through `safetyguard.py` before touching SQL or yfinance. 

It rejects SQL injection patterns (`DROP`, `DELETE`, `--`, `;`, quotes).
__Note: this may not be needed since this api is designed to be used locally, but its always nice have some safety mechanism__ 

Validation enforces ticker format (1–10 chars, uppercase alphanumeric), and validates date strings.
The validation layer acts as a pre-check mechanism, blocking costly database operations and yfinance API calls when the request is determined to be invalid

### Caching

Spline lambdas (the callable functions rebuilt from stored coefficients) live in an in-memory LRU cache with clock-sweep eviction).\
Two pools: 16 slots for daily splines, 8 for weekly data. This avoids re-reading and re-parsing coefficients stroed in the DB when `visualize()` is called repeatedly on the same ticker.

The clock-sweep works like a circular buffer with reference bits.\
On eviction, the hand sweeps around clearing reference bits until it finds one that's already zero and slot gets reused.\
Accessed entries get their bit set back to 1. Simple, O(1) amortized, no heap allocation.

### Quick DB Pinging

Before running the full cascade, `store.has_data()` does a lightweight existence check (`SELECT 1 ... LIMIT 1`) per table. If all stages already have data for that ticker, the cascade short-circuits and loads from DB instead of recomputing. `weekly_is_stale()` checks if the weekly metrics still have placeholder zeros (meaning daily analysis ran but weekly hasn't finalized yet).

---

## Visualization

`depict()` draws everything on a single chart:
```python
dl.visualize("AAPL")                                       # default 60d, hourly EMA, daily spline
dl.visualize("AAPL", ema_interval="5min")                  # 5-min EMA resolution
dl.visualize("AAPL", cubic_interval="week")                # adds long spline
dl.visualize("META", start="2026-01-01", end="2026-02-15") # custom range
```

- `ema_interval` is `"5min"` or `"1hr"`.
- `cubic_interval` is `"day"` (daily spline only) or `"week"` (daily + long spline).
- `data_type` is `"close"` or `"open"` for closing and open price.

---

## Project Structure

```
trendlens/
├── Database/                    # mkdir this before first run
├── example.py
└── src/
    ├── __init__.py              # db_path config, exports DataLoader
    ├── data_loader.py           # DataLoader: fetch, analyze, visualize
    ├── safetyguard.py           # input validation
    ├── cache.py                 # clock-sweep LRU for spline lambdas
    ├── db/
    │   ├── queries.py           # all SQL (parameterized)
    │   ├── ticker.py            # ticker_list.db access
    │   ├── price.py             # stock_price.db: yfinance fetch + local storage
    │   └── store.py             # analysis.db: init, load, store
    ├── analysis/
    │   ├── __init__.py          # cascade entry point
    │   ├── utils.py             # cubic spline fitter, date helpers
    │   ├── weights.py           # all weight/EMA math
    │   ├── minute.py            # 5-min per-bar weights → EMA
    │   ├── hourly.py            # hourly aggregated betas → EMA
    │   ├── daily.py             # daily spline + interpolant
    │   └── weekly.py            # long spline (gap-split)
    └── visualization/
        ├── utils.py             # matplotlib backend
        └── plot.py              # single-panel chart
```




