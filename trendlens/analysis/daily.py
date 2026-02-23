# analysis/daily.py — daily cubic spline over hourly EMA (EQ10)
# computes per-day metrics
# stores spline coefficients + metrics to analysis.db

import sqlite3
import json
import numpy as np
import pandas as pd

from trendlens import db_path
from trendlens import safetyguard as sg
from trendlens.db import queries, store
from trendlens import cache as C
from . import utils
from . import weights as W

N_DAILY_SEGMENTS = 7
SPLINE_COLS = ["close_ema", "open_ema"]

#daily analysis takes hourly data to build the cubic splines. 
#since weekly splines will create a nice, continuous curves, daily splines dont need to have connected splines
#between each days.

def analyze(ticker: str, company_name: str = None,
            storage: 'C.AnalysisCache | None' = None) -> dict | None:
    name = utils.validate_ticker(ticker)
    if name is None:
        return None

    start, end = utils.auto_range()

    from . import hourly as hourly_mod
    hourly_df = hourly_mod.load(name, start, end)
    if hourly_df is None:
        hourly_df = hourly_mod.analyze(name, company_name)
    if hourly_df is None:
        return None

    dates = sorted(set(hourly_df.index.date))
    all_spline_rows, all_metric_rows = [], []
    result = {}
    daily_cache = storage.daily if storage else None

    for d in dates:
        date_str = str(d)
        day_data = hourly_df[hourly_df.index.date == d]

        if daily_cache is not None:
            cached = daily_cache.get(name, date_str)
            if cached is not None:
                result[date_str] = cached
                continue

        day_fns, day_segments = {}, {}

        for col in SPLINE_COLS:
            if col not in day_data.columns:
                continue
            vals = day_data[col].dropna().values
            if len(vals) < 3:
                continue
            x = np.arange(len(vals), dtype=float)
            segments = utils.fit_natural_cubic_spline(x, vals.astype(float))
            day_segments[col] = (segments, vals)
            packed = C.pack_coefficients(segments)
            while len(packed) < N_DAILY_SEGMENTS:
                packed.append(json.dumps([0.0, 0.0, 0.0, 0.0]))
            all_spline_rows.append((name, date_str, col) + tuple(packed[:N_DAILY_SEGMENTS]))
            day_fns[col] = C.coefficients_to_lambdas(segments)

        # per-day metrics from close_ema spline 
        primary = day_segments.get("close_ema")
        wid = utils.week_id(pd.Timestamp(d))
        wd = pd.Timestamp(d).weekday()

        if primary:
            segs, ema_vals = primary
            beta_day = W.compute_beta_day(ema_vals, segs)
            arc_len  = W.spline_arc_length(segs)
            ema_rng  = float(np.nanmax(ema_vals) - np.nanmin(ema_vals))
            f_mean   = W.spline_integral_mean(segs)
            interp   = W.compute_daily_interpolant(ema_vals, segs, beta_day)
            # pearson/recency/weights = 0 (filled by weekly)
            all_metric_rows.append((
                name, date_str, wid, wd,
                beta_day, arc_len, ema_rng, f_mean, interp,
                0.0, 0.0, 0.0, 0.0))

        if day_fns:
            result[date_str] = day_fns
            if daily_cache is not None:
                daily_cache.put(name, date_str, day_fns)

    with sqlite3.connect(db_path + "/analysis.db") as conn:
        if all_spline_rows:
            conn.execute(queries.DELETE_DAILY_SPLINE_RANGE, (name, start, end))
            conn.executemany(queries.UPSERT_DAILY_SPLINE, all_spline_rows)
        if all_metric_rows:
            conn.execute(queries.DELETE_WEEKLY_METRICS_RANGE, (name, start, end))
            conn.executemany(queries.UPSERT_WEEKLY_METRICS, all_metric_rows)
        conn.commit()

    print(f"  {name}: stored daily spline ({len(result)} days) + per-day metrics")
    return result


def load_spline(ticker: str, storage: 'C.AnalysisCache | None' = None,
                start: str = None, end: str = None) -> dict | None:
    name = sg.validate_ticker(ticker)
    if name is None:
        return None
    if start is None or end is None:
        start, end = utils.auto_range()

    daily_cache = storage.daily if storage else None
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            cur = conn.cursor()
            cur.execute(queries.SELECT_DAILY_SPLINE_RANGE, (name, start, end))
            rows = cur.fetchall()
    except Exception:
        return None
    if not rows:
        return None

    result = {}
    for row in rows:
        date_str, col = row[0], row[1]
        seg_jsons = list(row[2:])
        if daily_cache is not None:
            cached = daily_cache.get(name, date_str)
            if cached is not None and col in cached:
                if date_str not in result:
                    result[date_str] = cached
                continue
        coeffs = C.unpack_coefficients(seg_jsons)
        coeffs = [c for c in coeffs if any(v != 0.0 for v in c)]
        fns = C.coefficients_to_lambdas(coeffs)
        if date_str not in result:
            result[date_str] = {}
        result[date_str][col] = fns
        if daily_cache is not None:
            daily_cache.put(name, date_str, result[date_str])
    return result


# bulk-load DB => DailyCache
def populate_cache(name: str, daily_cache):
    start, end = utils.auto_range()
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            cur = conn.cursor()
            cur.execute(queries.SELECT_DAILY_SPLINE_RANGE, (name, start, end))
            for row in cur.fetchall():
                date_str, col = row[0], row[1]
                coeffs = C.unpack_coefficients(list(row[2:]))
                coeffs = [c for c in coeffs if any(v != 0.0 for v in c)]
                fns = C.coefficients_to_lambdas(coeffs)
                existing = daily_cache.get(name, date_str) or {}
                existing[col] = fns
                daily_cache.put(name, date_str, existing)
    except Exception:
        pass
