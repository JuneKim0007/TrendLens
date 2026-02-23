# analysis/hourly.py — hourly EMA from aggregated 5-min betas 
# resamples raw OHLCV to 1h, applies hourly betas, stores to analysis_1hr

import sqlite3
import numpy as np
import pandas as pd

from trendlens import db_path
from trendlens.db import queries, store
from . import utils
from . import weights as W


def analyze(ticker: str, company_name: str = None) -> pd.DataFrame | None:
    name = utils.validate_ticker(ticker)
    if name is None:
        return None
    v_co = utils.validate_company(company_name, name)
    if v_co is None:
        return None

    start, end = utils.auto_range()

    from . import minute as minute_mod
    fivemin_df = minute_mod.load(name, start, end)
    if fivemin_df is None:
        fivemin_df = minute_mod.analyze(name, company_name)
    if fivemin_df is None:
        return None

    raw = store.load_price_data(name, start, end)
    if raw is None or raw.empty:
        return None

    c_carry, o_carry = store.get_ema_carry(name, queries.Q_1HR)
    cfg = utils.INTERVAL_CONFIG["1h"]
    all_rows, all_frames = [], []

    trading_days = list(pd.bdate_range(start, end, freq='B'))
    raw_dates = set(raw.index.date)

    for day in trading_days:
        if day.date() not in raw_dates:
            continue

        wid = utils.week_id(day)
        day_5min = fivemin_df[fivemin_df.index.date == day.date()]
        if day_5min.empty or 'beta_weight' not in day_5min.columns:
            continue

        hourly_betas  = W.compute_hourly_betas(day_5min['beta_weight'].values, bars_per_hour=12)
        hourly_alphas = W.beta_to_alpha(hourly_betas)

        hourly_bars = utils.resample_to_hourly(raw[raw.index.date == day.date()])
        if hourly_bars.empty:
            continue
        hourly_bars = hourly_bars.between_time(
            f"{cfg['start_time'][0]:02d}:{cfg['start_time'][1]:02d}",
            f"{cfg['end_time'][0]:02d}:{cfg['end_time'][1]:02d}")
        if hourly_bars.empty:
            continue

        n = len(hourly_bars)
        close_v, open_v = hourly_bars['Close'].values, hourly_bars['Open'].values
        vol_v, timestamps = hourly_bars['Volume'].values, hourly_bars.index

        betas = hourly_betas[:n] if len(hourly_betas) >= n else np.pad(
            hourly_betas, (0, n - len(hourly_betas)), constant_values=W.BETA_FLOOR)
        alphas = hourly_alphas[:n] if len(hourly_alphas) >= n else np.pad(
            hourly_alphas, (0, n - len(hourly_alphas)), constant_values=W.ALPHA_FLOOR)

        close_avg = utils.sliding_avg(close_v, cfg["window"])
        open_avg  = utils.sliding_avg(open_v, cfg["window"])
        vol_avg   = utils.sliding_avg(vol_v, cfg["window"])

        if c_carry is not None:
            c_init, o_init = c_carry, o_carry
        else:
            c_init = W.compute_ema_init(open_v[0], close_v[0])
            o_init = W.compute_ema_init(close_v[0], open_v[0])

        close_ema = W.compute_ema_variable_beta(close_v, c_init, betas)
        open_ema  = W.compute_ema_variable_beta(open_v,  o_init, betas)
        c_carry, o_carry = close_ema[-1], open_ema[-1]

        for i in range(n):
            ts = timestamps[i]
            all_rows.append((
                name, v_co, str(ts), wid, ts.weekday(),
                close_avg[i], close_ema[i], open_avg[i], open_ema[i],
                vol_avg[i], betas[i], alphas[i]))

        all_frames.append(pd.DataFrame({
            'Datetime': timestamps, 'ticker': name, 'company_name': v_co,
            'week_id': wid, 'weekday': [ts.weekday() for ts in timestamps],
            'close_avg': close_avg, 'close_ema': close_ema,
            'open_avg': open_avg, 'open_ema': open_ema,
            'vol_avg': vol_avg, 'beta_weight': betas, 'alpha_weight': alphas,
            'raw_close': close_v, 'raw_open': open_v,
        }).set_index('Datetime'))

    if not all_frames:
        print(f"  {name}: no hourly data produced")
        return None

    with sqlite3.connect(db_path + "/analysis.db") as conn:
        conn.execute(queries.Q_1HR["delete_range"], (name, start, end))
        conn.executemany(queries.Q_1HR["upsert"], all_rows)
        conn.commit()

    result = pd.concat(all_frames)
    print(f"  {name}: stored {len(result)} hourly rows")
    return result


def load(ticker, start=None, end=None):
    return store.load_table(ticker, queries.Q_1HR, start, end)

def load_week(ticker, week_id):
    return store.load_week(ticker, week_id, queries.Q_1HR)