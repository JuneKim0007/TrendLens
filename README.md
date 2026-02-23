# Stock Trend Analyzer

**Language:** Python 3.13  
**Stack:** SQLite, NumPy, Pandas, Matplotlib, yfinance  

Builds multi-timeframe exponential moving averages (5min → hourly) and cubic spline curves (daily → long-range) from Yahoo Finance data. Stores everything in a local SQLite database for incremental analysis.

```
dl = DataLoader()
dl.visualize("AAPL")
```

One call fetches 60 days of 5-minute OHLCV, runs the full analysis cascade, and plots a single chart with four layers: raw price, sliding average, EMA, daily cubic spline, and long-range cubic spline.

---

## Quick Start

```bash
mkdir -p Database
pip install yfinance numpy pandas matplotlib
python example.py
```

The `Database/` directory must exist at project root. It holds three SQLite files created automatically on first run.

---

## Analysis Cascade

Each stage builds on the one before it. The cascade runs automatically when you call `visualize()` or `analyze()`.

```
Raw OHLCV (5min bars from Yahoo Finance)
    │
    ▼
Stage 1: 5-Minute Analysis  (minute.py)
    │   per-bar weights → beta → EMA
    │   stores to: analysis_5min
    ▼
Stage 2: Hourly Analysis  (hourly.py)
    │   aggregates 5min betas → hourly beta → EMA
    │   stores to: analysis_1hr
    ▼
Stage 3: Daily Spline  (daily.py)
    │   fits cubic spline through ~8 hourly EMA values per day
    │   computes per-day interpolant via arc-length trust weighting
    │   stores to: daily_spline_coefficients, weekly_metrics
    ▼
Stage 4: Long Spline  (weekly.py → build_long_spline)
    │   collects ALL daily interpolants across the full date range
    │   splits on gaps (holidays, missing data)
    │   fits one cubic spline per contiguous segment
    ▼
Visualization  (plot.py → depict)
    all four layers on a single chart
```

---

## Equations

All equations live in `analysis/weights.py`. They are referenced as EQ1–EQ19 throughout the codebase.

### Stage 1: 5-Minute Per-Bar Weights

For each trading day (09:30–15:55), every 5-minute bar gets a weight that reflects how much of the day's total volatility it represents.

**EQ3 — Day range:**  
The full price swing for the day, used to normalize all bar weights.

```
day_range = max(all_highs) - min(all_lows)
```

**EQ4 — Initial weight:**  
The first bar's relative contribution.

```
w[0] = (high[0] - low[0]) / day_range
```

**EQ5 — Accumulated bar weight:**  
Each bar accumulates the previous weight plus its own bar range, normalized by the day range. This makes the weight grow with cumulative volatility.

```
w[i] = (w[i-1] + (high[i] - low[i])) / day_range
```

**EQ6 — Beta (trust factor):**  
Clamped version of the bar weight. Controls how much the current sample influences the EMA. On volatile days, individual bars are smaller relative to the day range, keeping beta low and the EMA smooth.

```
β[i] = clamp(w[i], 0.001, 0.95)
```

**EQ7 — Alpha:**  
A secondary dampened weight (used for volume analysis).

```
α[i] = clamp(β[i] * 0.8, 0.001, 0.80)
```

**EQ2 — EMA initial value:**  
Seed value for the first bar of the first day (or carried over from the previous day).

```
ema_init = ((1 + close/open) / 2) * ((close + open) / 2)
```

**EQ1 — EMA update:**  
The core equation. Each bar blends the previous EMA with the new sample, weighted by beta.

```
ema[i] = (1 - β[i]) * ema[i-1] + β[i] * sample[i]
```

**EQ8 — Sliding average:**  
A 13-bar centered moving average computed alongside the EMA for comparison.

```
avg[i] = mean(samples[i-6 .. i+6])
```

### Stage 2: Hourly Aggregation

**EQ9 — Hourly beta:**  
Each hour contains 12 five-minute bars. The hourly beta is their mean, giving a smoother weight that reflects the hour's overall character.

```
β_h[j] = sum(β_5min[j*12 .. (j+1)*12]) / 12
```

The hourly EMA then uses EQ1 with these aggregated betas applied to hourly OHLCV bars.

### Stage 3: Daily Spline + Interpolant

A trading day produces ~8 hourly EMA values (09:00–16:00). A natural cubic spline is fitted through these points.

**EQ10 — Natural cubic spline:**  
Each segment between adjacent hourly knots is a cubic polynomial normalized to t ∈ [0, 1].

```
S_i(t) = a_i·t³ + b_i·t² + c_i·t + d_i
```

Coefficients are solved via the Thomas algorithm with natural boundary conditions (second derivatives = 0 at endpoints). Implementation is in `analysis/utils.py → fit_natural_cubic_spline()`.

**EQ11 — Arc length:**  
The total length of the spline curve, computed via 5-point Gauss-Legendre quadrature. Measures how "wiggly" the day was.

```
arc_length = Σ_segments ∫₀¹ √(1 + S'(t)²) dt
```

**EQ12 — EMA range:**

```
ema_range = max(hourly_ema) - min(hourly_ema)
```

**EQ13 — Daily beta (β_day):**  
The ratio of range to arc length. If the day was straight (range ≈ arc), beta is near 0 (trust the simple median). If the day was curvy (range << arc), beta is near 1 (trust the spline integral mean more).

```
β_day = 1 - ema_range / arc_length
```

**EQ14 — Spline integral mean:**  
The average value of the cubic spline function over one segment, computed analytically.

```
f_mean = (1/N) * Σ_segments (a/4 + b/3 + c/2 + d)
```

**EQ15 — Daily interpolant:**  
Blends the median hourly EMA with the spline integral mean, weighted by the daily beta. This single number represents the day's "characteristic price."

```
interpolant = (1 - β_day) * median(hourly_ema) + β_day * f_mean
```

### Stage 4: Long Spline

Collects all daily interpolants across the full date range (typically ~40 trading days over 60 calendar days). This is NOT per-calendar-week — it is one continuous spline, split only where data gaps exceed 5 calendar days.

**Gap detection:** Adjacent trading dates separated by > 5 calendar days (holidays, missing data) are treated as a break. Each contiguous segment gets its own cubic spline.

**Spline fitting:** Same natural cubic spline (EQ10) applied to the sequence of daily interpolants. Requires ≥ 3 points per segment; segments with exactly 2 points use linear interpolation.

The per-day metrics (EQ16–EQ19) are still computed for DB storage but are not used for the long spline construction:

**EQ16 — Pearson ρ:** Correlation between day index and interpolant values.  
**EQ17 — Recency:** Exponential recency weighting: `exp(|ρ| * i / (n-1))`.  
**EQ18 — Raw weight:** `(1 - β_day) * recency`.  
**EQ19 — Normalized weight:** `w_i / Σ w_j`.

---

## Visualization

`depict()` draws a single chart with all layers overlaid on one datetime axis:

| Layer | Color | Description |
|-------|-------|-------------|
| Raw price | Gray (#d0d0d0) | 5-min close prices, faded background |
| Sliding avg | Blue (#0072B2) | 13-bar centered window |
| EMA | Orange (#E69F00) | Linear connections between discrete EMA points |
| Daily spline | Red (#CC0000) | Cubic curve through hourly EMA knots, per day |
| Long spline | Purple (#7B2D8E) | Cubic curve through all daily interpolants |
| Interpolant knots | Purple dots | One per trading day, at midday |

The daily spline shows intraday curvature. The long spline shows the macro trend. The EMA shows the discrete weighted estimates. All share the same y-axis (price) and x-axis (datetime).

**Parameters:**

```python
dl.visualize("AAPL",
    start="2026-01-01",       # default: auto (60 days back)
    end="2026-02-20",         # default: auto (yesterday)
    ema_interval="5min",      # "5min" or "1hr"
    cubic_interval="day",     # "day" (daily spline only) or "week" (daily + long spline)
    data_type="close",        # "close" or "open"
)
```

---

## Database Schema

Three SQLite files in `Database/`:

### `ticker_list.db`

| Table | Columns | Purpose |
|-------|---------|---------|
| `ticker_list` | ticker (PK) | Registry of known valid tickers |

### `stock_price.db`

| Table | Columns | Purpose |
|-------|---------|---------|
| `meta_data` | ticker (PK), name, sector, exchange | Ticker metadata |
| `price_data` | ticker + Datetime (PK), OHLCV, volatility | Raw 5-min OHLCV from Yahoo Finance |

### `analysis.db`

| Table | Columns | Purpose |
|-------|---------|---------|
| `analysis_5min` | ticker + Datetime (PK), close_avg, close_ema, open_avg, open_ema, vol_avg, beta_weight, alpha_weight, week_id, weekday | 5-min EMA results |
| `analysis_1hr` | Same schema as analysis_5min | Hourly EMA results |
| `daily_ema_init` | ticker + trade_date (PK), close_ema_init, open_ema_init, day_range, init_weight | Per-day precomputed values |
| `missing_segments` | ticker + segment_date (PK), week_id, segment_type | Tracks holidays and missing trading days |
| `daily_spline_coefficients` | ticker + trade_date + data_col (PK), seg_1..seg_7 (JSON) | Cubic spline coefficients per day |
| `weekly_metrics` | ticker + trade_date (PK), week_id, weekday, beta_day, arc_length, ema_range, f_mean, interpolant, pearson_rho, recency, raw_weight, norm_weight | Per-day metrics used by long spline |
| `weekly_spline_coefficients` | ticker + week_id + data_col (PK), seg_1..seg_4 (JSON) | Per-week spline coefficients (legacy, kept for compatibility) |

All queries use parameterized `?` placeholders. All user inputs pass through `safetyguard.py` before reaching SQL.

### DB Lifecycle

1. `DataLoader()` constructor calls `ticker_db.init_db()` and `price_db.init_db()` to create tables if absent.
2. `_ensure_data()` fetches from Yahoo Finance and stores to `stock_price.db` if no local data exists.
3. `analyze()` cascade creates `analysis.db` tables, drops and recreates them on each run.
4. `build_long_spline()` reads from `weekly_metrics` (populated by `daily.analyze()` and `weekly.analyze()`).

---

## File Walkthrough

```
project/
├── Database/                          # created by user, populated automatically
│   ├── ticker_list.db
│   ├── stock_price.db
│   └── analysis.db
├── example.py                         # usage examples
└── src/
    ├── __init__.py                    # db_path config, exports DataLoader
    ├── data_loader.py                 # DataLoader class: fetch, analyze, visualize
    ├── safetyguard.py                 # input validation (ticker, dates, SQL injection)
    ├── cache.py                       # clock-sweep LRU cache for spline lambdas
    ├── db/
    │   ├── __init__.py
    │   ├── queries.py                 # all SQL templates (parameterized)
    │   ├── ticker.py                  # ticker_list.db access + yfinance validation
    │   ├── price.py                   # stock_price.db: fetch from yf, store, load OHLCV
    │   └── store.py                   # analysis.db: init tables, load/store analysis data
    ├── analysis/
    │   ├── __init__.py                # cascade entry: analyze("AAPL", "week")
    │   ├── utils.py                   # date helpers, sliding avg, cubic spline fitter
    │   ├── weights.py                 # EQ1-19: all weight and EMA computations
    │   ├── minute.py                  # stage 1: 5-min per-bar weights → EMA
    │   ├── hourly.py                  # stage 2: hourly aggregated betas → EMA
    │   ├── daily.py                   # stage 3: daily cubic spline + interpolant
    │   └── weekly.py                  # stage 4: per-week metrics + build_long_spline()
    └── visualization/
        ├── __init__.py                # exports depict(), validate_range()
        ├── utils.py                   # matplotlib backend helper
        └── plot.py                    # single-panel 4-layer chart
```

### `data_loader.py` — User-facing API

`DataLoader` is the only class users interact with. `visualize()` runs the full pipeline: validate ticker → ensure raw data in DB → validate date range → cascade analysis → plot. `analyze()` runs the cascade without plotting. `fetch()` and `load()` provide raw Stock objects.

### `safetyguard.py` — Input Validation

Rejects SQL injection patterns, enforces ticker format (1–10 chars, uppercase alphanumeric + `.` `-` `^`), validates date strings. Every user-facing string passes through here before reaching SQL or yfinance.

### `cache.py` — LRU Cache

Clock-sweep eviction policy. Two instances: `DailyCache` (16 slots) for daily spline lambdas, `WeeklyCache` (8 slots) for weekly data. Avoids redundant DB reads during repeated `visualize()` calls.

### `analysis/weights.py` — Core Math

Pure NumPy functions implementing EQ1–EQ19. No DB access, no side effects. Takes arrays in, returns arrays out.

### `analysis/minute.py` — 5-Minute Analysis

Loads raw OHLCV, organizes by trading days, computes per-bar weights (EQ3–7), runs EMA (EQ1–2), stores results to `analysis_5min`. The EMA carry value propagates across days for continuity.

### `analysis/hourly.py` — Hourly Aggregation

Loads 5-min analysis data, aggregates betas into hourly values (EQ9), resamples raw OHLCV to 1-hour bars, runs hourly EMA. Stores to `analysis_1hr`.

### `analysis/daily.py` — Daily Spline

Loads hourly EMA values, fits a natural cubic spline per day (EQ10), computes arc length (EQ11), daily beta (EQ13), interpolant (EQ15). Stores spline coefficients as JSON and per-day metrics to DB.

### `analysis/weekly.py` — Long Spline

`analyze()` computes per-week Pearson correlation, recency, and normalized weights (EQ16–19) and stores them to `weekly_metrics`. `build_long_spline()` reads ALL daily interpolants, splits on gaps > 5 calendar days, and fits one cubic spline per contiguous segment.

### `visualization/plot.py` — Rendering

`_spline_to_datetimes()` sweeps one continuous parameter across all spline segments and maps each evaluation point to its real datetime via linear interpolation between knot timestamps. `validate_range()` pings the DB for actual trading dates before attempting to draw. `depict()` overlays all four layers on a single matplotlib axis.

---

## Usage Examples

```python
from src.data_loader import DataLoader

dl = DataLoader()

# default: 60-day range, hourly EMA, daily spline only
dl.visualize("AAPL")

# 5-minute EMA resolution + daily spline
dl.visualize("AAPL", ema_interval="5min")

# hourly EMA + daily spline + long spline (macro trend)
dl.visualize("AAPL", cubic_interval="week")

# custom date range
dl.visualize("META", start="2026-01-01", end="2026-02-15", cubic_interval="week")

# manual analysis without plotting
result = dl.analyze("AAPL", mode="week")
long = dl.load_long_spline("AAPL")
```

---

## Dependencies

- Python 3.13
- NumPy
- Pandas
- Matplotlib
- SQLite3 (built-in)
- yfinance