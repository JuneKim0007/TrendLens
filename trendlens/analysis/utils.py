# analysis/utils.py — shared computation helpers (no DB writes)
#
# validation, date helpers, WeekBlock, sliding avg, spline fitter
# parameterized by interval: "5min" or "1h"

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from trendlens import safetyguard as sg
from trendlens.db import ticker as ticker_db
from . import weights as W


#CONSTANTS
PREDICTION_START = pd.Timestamp("2026-01-01")
LOOKBACK_DAYS = 60

INTERVAL_CONFIG = {
    "5min": {"freq": "5min", "start_time": (9, 30), "end_time": (15, 55), "window": 13},
    "1h":   {"freq": "1h",   "start_time": (9, 0),  "end_time": (16, 0),  "window": 3},
}


@dataclass
class WeekBlock:
    week_id: str
    available: list = field(default_factory=list)
    missing:   list = field(default_factory=list)


# ── date helpers ──

def yesterday() -> pd.Timestamp:
    return pd.Timestamp((datetime.now() - timedelta(days=1)).date())

def auto_range() -> tuple[str, str]:
    end = yesterday()
    start = max(PREDICTION_START, end - pd.Timedelta(days=LOOKBACK_DAYS))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def week_id(day: pd.Timestamp) -> str:
    iso = day.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


# ── validation (syntax → DB → yfinance) ──

def validate_ticker(ticker: str) -> str | None:
    name = sg.validate_ticker(ticker)
    if name is None:
        return None
    if ticker_db.ticker_exists(name):
        return name
    print(f"  '{name}': not found in local DB or Yahoo Finance")
    return None

def validate_company(company_name: str, fallback: str) -> str | None:
    co = company_name or fallback
    v = sg.validate_company_key(co)
    if v is None:
        print(f"  Invalid company name '{co}'")
    return v


# ── array helpers ──

def fill_nearest(s: pd.Series) -> pd.Series:
    return s.ffill().bfill()

def sliding_avg(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean().values

def resample_to_hourly(raw_5min: pd.DataFrame) -> pd.DataFrame:
    return raw_5min.resample('1h').agg({
        'Open': 'first', 'Close': 'last',
        'High': 'max', 'Low': 'min', 'Volume': 'sum',
    }).dropna(subset=['Open', 'Close'])


#day index (full market hours grid for an interval) ──

def day_index(day: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    cfg = INTERVAL_CONFIG[interval]
    s = day.replace(hour=cfg["start_time"][0], minute=cfg["start_time"][1], second=0, microsecond=0)
    e = day.replace(hour=cfg["end_time"][0],   minute=cfg["end_time"][1],   second=0, microsecond=0)
    return pd.date_range(s, e, freq=cfg["freq"])


#weekly framing

def build_week_blocks(trading_days, raw_dates, raw, interval) -> list[WeekBlock]:
    weeks: dict[str, WeekBlock] = {}
    for day in trading_days:
        wid = week_id(day)
        if wid not in weeks:
            weeks[wid] = WeekBlock(week_id=wid)
        if day.date() in raw_dates:
            day_data = raw[raw.index.date == day.date()].copy()
            if not day_data.empty:
                full_idx = day_index(day, interval)
                day_data = day_data.reindex(full_idx)
                weeks[wid].available.append((day, day_data))
            else:
                weeks[wid].missing.append(day)
        else:
            weeks[wid].missing.append(day)
    return [weeks[k] for k in sorted(weeks.keys())]


#precompute per-day range + ema init

def precompute_days(week_blocks: list[WeekBlock]) -> dict:
    out = {}
    for wb in week_blocks:
        for day, day_df in wb.available:
            valid = day_df.dropna(subset=['Close', 'Open'])
            if valid.empty:
                continue
            highs = valid['High'].values if 'High' in valid else valid['Close'].values
            lows = valid['Low'].values if 'Low' in valid else valid['Open'].values
            dr = W.compute_day_range(highs, lows)
            first = valid.iloc[0]
            out[day.strftime("%Y-%m-%d")] = {
                "day_range": dr,
                "init_weight": (highs[0] - lows[0]) / dr,
                "c_ema_init": W.compute_ema_init(first['Open'], first['Close']),
                "o_ema_init": W.compute_ema_init(first['Close'], first['Open']),
            }
    return out


#natural cubic spline normalized t ∈ [0,1]

def fit_natural_cubic_spline(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    n = len(x) - 1
    h = np.diff(x).astype(float)
    rhs = np.zeros(n + 1)
    for i in range(1, n):
        rhs[i] = 6.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    M = np.zeros(n + 1)
    if n > 1:
        size = n - 1
        diag  = np.zeros(size)
        upper = np.zeros(size)
        lower = np.zeros(size)
        b_vec = np.zeros(size)
        for i in range(size):
            ii = i + 1
            diag[i]  = 2.0 * (h[ii-1] + h[ii])
            b_vec[i] = rhs[ii]
            if i > 0:        lower[i] = h[ii-1]
            if i < size - 1: upper[i] = h[ii]
        for i in range(1, size):
            factor = lower[i] / diag[i-1]
            diag[i]  -= factor * upper[i-1]
            b_vec[i] -= factor * b_vec[i-1]
        sol = np.zeros(size)
        sol[-1] = b_vec[-1] / diag[-1]
        for i in range(size - 2, -1, -1):
            sol[i] = (b_vec[i] - upper[i] * sol[i+1]) / diag[i]
        M[1:n] = sol

    segments = []
    for i in range(n):
        hi2 = h[i] ** 2
        A = M[i]   * hi2 / 6.0
        B = M[i+1] * hi2 / 6.0
        a = B - A
        b = 3.0 * A
        c = (y[i+1] - y[i]) - M[i] * hi2 / 3.0 - M[i+1] * hi2 / 6.0
        d = float(y[i])
        segments.append([a, b, c, d])
    return segments