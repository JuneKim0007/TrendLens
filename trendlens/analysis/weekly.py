# analysis/weekly.py — weekly spline over daily interpolants (EQ16-19)
# pearson correlation → recency → weights → cubic spline through daily points
# stores coefficients + metrics to analysis.db

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

N_WEEKLY_SEGMENTS = 4


def analyze(ticker: str, company_name: str = None,
            storage: 'C.AnalysisCache | None' = None) -> dict | None:
    name = utils.validate_ticker(ticker)
    if name is None:
        return None

    start, end = utils.auto_range()

    if not store.has_data(name, "weekly_metrics"):
        from . import daily as daily_mod
        daily_mod.analyze(name, company_name, storage)

    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            metrics_df = pd.read_sql_query(
                queries.SELECT_WEEKLY_METRICS_RANGE, conn, params=(name, start, end))
    except Exception:
        return None
    if metrics_df.empty:
        return None

    week_groups = metrics_df.groupby('week_id')
    all_spline_rows, all_metric_updates = [], []
    result = {}
    weekly_cache = storage.weekly if storage else None

    for wid, wk_df in week_groups:
        wk_df = wk_df.sort_values('weekday')
        n_days = len(wk_df)

        if weekly_cache is not None:
            cached = weekly_cache.get(name, wid)
            if cached is not None:
                result[wid] = cached
                continue

        interpolants = wk_df['interpolant'].values.astype(float)
        beta_days = wk_df['beta_day'].values.astype(float)

        #handles n_days=1 
        rho, norm_weights = W.compute_weekly_weights(interpolants, beta_days)
        abs_rho = abs(rho)
        day_idx = np.arange(n_days, dtype=float)
        recencies = np.exp(abs_rho * day_idx / max(n_days - 1, 1))
        confidences = 1.0 - beta_days
        raw_weights = confidences * recencies

        for i, (_, row) in enumerate(wk_df.iterrows()):
            all_metric_updates.append((
                name, row['trade_date'], wid, int(row['weekday']),
                float(row['beta_day']), float(row['arc_length']),
                float(row['ema_range']), float(row['f_mean']),
                float(row['interpolant']),
                rho, float(recencies[i]),
                float(raw_weights[i]), float(norm_weights[i])))

        # fit spline: >=3 -> cubic, 2 -> linear, 1 -> metrics only
        segments = None
        if n_days >= 3:
            segments = utils.fit_natural_cubic_spline(day_idx, interpolants)
        elif n_days == 2:
            segments = [[0.0, 0.0, float(interpolants[1] - interpolants[0]),
                         float(interpolants[0])]]

        if segments is not None:
            packed = C.pack_coefficients(segments)
            while len(packed) < N_WEEKLY_SEGMENTS:
                packed.append(json.dumps([0.0, 0.0, 0.0, 0.0]))
            all_spline_rows.append((name, wid, "close_ema") + tuple(packed[:N_WEEKLY_SEGMENTS]))

        week_fns = {
            "_meta": {
                "pearson": rho, "n_days": n_days,
                "weights": norm_weights.tolist(),
                "interpolants": interpolants.tolist(),
                "beta_days": beta_days.tolist(),
            },
        }
        if segments is not None:
            week_fns["close_ema"] = C.coefficients_to_lambdas(segments)

        result[wid] = week_fns
        if weekly_cache is not None:
            weekly_cache.put(name, wid, week_fns)

    with sqlite3.connect(db_path + "/analysis.db") as conn:
        if all_metric_updates:
            conn.executemany(queries.UPSERT_WEEKLY_METRICS, all_metric_updates)
        if all_spline_rows:
            s_wid = min(wid for wid, _ in week_groups)
            e_wid = max(wid for wid, _ in week_groups)
            conn.execute(queries.DELETE_WEEKLY_SPLINE_RANGE, (name, s_wid, e_wid))
            conn.executemany(queries.UPSERT_WEEKLY_SPLINE, all_spline_rows)
        conn.commit()

    print(f"  {name}: stored weekly spline for {len(result)} weeks")
    return result


#loads weekly spline from DB, reconstructs _meta from weekly_metrics
def load_spline(ticker: str, storage: 'C.AnalysisCache | None' = None,
                start_wid: str = None, end_wid: str = None) -> dict | None:
    name = sg.validate_ticker(ticker)
    if name is None:
        return None

    weekly_cache = storage.weekly if storage else None
    if start_wid is None or end_wid is None:
        s, e = utils.auto_range()
        start_wid = utils.week_id(pd.Timestamp(s))
        end_wid = utils.week_id(pd.Timestamp(e))

    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            cur = conn.cursor()
            cur.execute(queries.SELECT_WEEKLY_SPLINE_RANGE, (name, start_wid, end_wid))
            spline_rows = cur.fetchall()
    except Exception:
        spline_rows = []

    s_date, e_date = utils.auto_range()
    meta_by_week = _load_meta(name, s_date, e_date)

    result = {}
    for row in spline_rows:
        wid, col = row[0], row[1]
        if weekly_cache is not None:
            cached = weekly_cache.get(name, wid)
            if cached is not None:
                result[wid] = cached
                continue
        coeffs = C.unpack_coefficients(list(row[2:]))
        coeffs = [c for c in coeffs if any(v != 0.0 for v in c)]
        if wid not in result:
            result[wid] = {}
        result[wid][col] = C.coefficients_to_lambdas(coeffs)

    #attach _meta to all weeks
    for wid in set(list(result.keys()) + list(meta_by_week.keys())):
        if wid not in result:
            result[wid] = {}
        result[wid]["_meta"] = meta_by_week.get(wid, {
            "pearson": 0.0, "n_days": 0,
            "weights": [], "interpolants": [], "beta_days": []})
        if weekly_cache is not None:
            weekly_cache.put(name, wid, result[wid])

    return result if result else None


def load_metrics(ticker: str, start: str = None, end: str = None):
    name = sg.validate_ticker(ticker)
    if name is None:
        return None
    if start is None or end is None:
        start, end = utils.auto_range()
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            df = pd.read_sql_query(queries.SELECT_WEEKLY_METRICS_RANGE, conn,
                                   params=(name, start, end))
        return df if not df.empty else None
    except Exception:
        return None


#batch load DB → WeeklyCache
def populate_cache(name: str, weekly_cache):
    s, e = utils.auto_range()
    start_wid = utils.week_id(pd.Timestamp(s))
    end_wid = utils.week_id(pd.Timestamp(e))
    meta_by_week = _load_meta(name, s, e)
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            cur = conn.cursor()
            cur.execute(queries.SELECT_WEEKLY_SPLINE_RANGE, (name, start_wid, end_wid))
            for row in cur.fetchall():
                wid, col = row[0], row[1]
                coeffs = C.unpack_coefficients(list(row[2:]))
                coeffs = [c for c in coeffs if any(v != 0.0 for v in c)]
                existing = weekly_cache.get(name, wid) or {}
                existing[col] = C.coefficients_to_lambdas(coeffs)
                existing["_meta"] = meta_by_week.get(wid, {
                    "pearson": 0.0, "n_days": 0,
                    "weights": [], "interpolants": [], "beta_days": []})
                weekly_cache.put(name, wid, existing)
    except Exception:
        pass


#long spline: ONE spline through ALL daily interpolants, gap-split
# collects every daily interpolant across the full range, detects gaps > GAP_DAYS,
# splits into contiguous segments, fits cubic spline per segment
# returns: [{"dates": [str], "values": [float], "fns": [lambda]|None}]

GAP_DAYS = 5  # > 5 calendar days between adjacent trading dates = gap

def build_long_spline(ticker: str, start: str = None, end: str = None) -> list[dict]:
    name = sg.validate_ticker(ticker)
    if name is None:
        return []
    if start is None or end is None:
        start, end = utils.auto_range()

    metrics = load_metrics(name, start, end)
    if metrics is None or metrics.empty:
        return []

    metrics = metrics.sort_values('trade_date')
    dates = metrics['trade_date'].tolist()
    interps = metrics['interpolant'].values.astype(float)

    # split on gaps
    raw_segments = []
    cur_d, cur_v = [dates[0]], [interps[0]]

    for i in range(1, len(dates)):
        gap = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days
        if gap > GAP_DAYS:
            raw_segments.append((cur_d, cur_v))
            cur_d, cur_v = [dates[i]], [interps[i]]
        else:
            cur_d.append(dates[i])
            cur_v.append(interps[i])
    raw_segments.append((cur_d, cur_v))

    # fit cubic spline per contiguous segment (>= 3 points required)
    result = []
    for seg_dates, seg_vals in raw_segments:
        n = len(seg_dates)
        vals = np.array(seg_vals, dtype=float)
        if n >= 3:
            x = np.arange(n, dtype=float)
            coeffs = utils.fit_natural_cubic_spline(x, vals)
            fns = C.coefficients_to_lambdas(coeffs)
        elif n == 2:
            # linear segment: a=0, b=0, c=slope, d=y0
            fns = [lambda t, y0=vals[0], y1=vals[1]: y0 + t * (y1 - y0)]
        else:
            fns = None

        result.append({
            "dates": seg_dates,
            "values": seg_vals,
            "fns": fns,
        })

    return result


def _load_meta(name, start, end) -> dict:
    meta = {}
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            df = pd.read_sql_query(queries.SELECT_WEEKLY_METRICS_RANGE, conn,
                                   params=(name, start, end))
        if not df.empty:
            for wid, grp in df.groupby('week_id'):
                grp = grp.sort_values('weekday')
                meta[wid] = {
                    "pearson": float(grp['pearson_rho'].iloc[0]),
                    "n_days": len(grp),
                    "weights": grp['norm_weight'].tolist(),
                    "interpolants": grp['interpolant'].tolist(),
                    "beta_days": grp['beta_day'].tolist(),
                }
    except Exception:
        pass
    return meta