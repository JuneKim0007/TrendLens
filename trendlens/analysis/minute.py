# analysis/minute.py — 5-min EMA analysis
#
# analyze() loads raw data, computes per-bar weights + EMA, stores to analysis_5min
# the core loop was previously in utils.run_analysis(), now inlined here

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
    store.init_db()

    raw = store.load_price_data(name, start, end)
    if raw is None or raw.empty:
        return None

    trading_days = list(pd.bdate_range(start, end, freq='B'))
    raw_dates = set(raw.index.date)
    week_blocks = utils.build_week_blocks(trading_days, raw_dates, raw, "5min")

    if sum(len(w.available) for w in week_blocks) == 0:
        print(f"  {name}: no trading days with data")
        return None

    day_precomp = utils.precompute_days(week_blocks)
    store.store_precomp(name, week_blocks, day_precomp)
    c_carry, o_carry = store.get_ema_carry(name, queries.Q_5MIN)

    rows, frames = _run_loop(name, v_co, week_blocks, day_precomp, c_carry, o_carry)

    if not frames:
        print(f"  {name}: no 5min data produced")
        return None

    with sqlite3.connect(db_path + "/analysis.db") as conn:
        conn.execute(queries.Q_5MIN["delete_range"], (name, start, end))
        conn.executemany(queries.Q_5MIN["upsert"], rows)
        conn.commit()

    result = pd.concat(frames)
    n_days = sum(len(w.available) for w in week_blocks)
    print(f"  {name}: stored {len(result)} 5min rows — {n_days} days")
    return result

# first we fill data that are missing by fetching the nearest data.
# then it populates the weight for the moving average first.
# after building the weight, 
def _run_loop(name, v_co, week_blocks, day_precomp, c_carry, o_carry):
    cfg = utils.INTERVAL_CONFIG["5min"]
    window = cfg["window"]
    all_rows, all_frames = [], []

    for wb in week_blocks:
        for day, day_df in wb.available:
            day_str = day.strftime("%Y-%m-%d")
            for col in ['Close', 'Open', 'High', 'Low']:
                day_df[col] = utils.fill_nearest(day_df[col])
            day_df['Volume'] = utils.fill_nearest(day_df['Volume']).fillna(0)
            day_df = day_df.dropna(subset=['Close', 'Open'])
            if day_df.empty:
                continue

            n = len(day_df)
            close_v, open_v = day_df['Close'].values, day_df['Open'].values
            vol_v = day_df['Volume'].values
            high_v = day_df['High'].values if 'High' in day_df else close_v
            low_v  = day_df['Low'].values  if 'Low'  in day_df else open_v
            timestamps = day_df.index

            pre = day_precomp.get(day_str)
            if pre is None:
                continue

            bar_w  = W.compute_bar_weights(high_v, low_v, pre["day_range"])
            betas  = W.weights_to_beta(bar_w)
            alphas = W.beta_to_alpha(betas)
            close_avg = utils.sliding_avg(close_v, window)
            open_avg  = utils.sliding_avg(open_v, window)
            vol_avg   = utils.sliding_avg(vol_v, window)

            if c_carry is not None:
                c_init, o_init = c_carry, o_carry
            else:
                c_init, o_init = pre["c_ema_init"], pre["o_ema_init"]

            close_ema = W.compute_ema_variable_beta(close_v, c_init, betas)
            open_ema  = W.compute_ema_variable_beta(open_v,  o_init, betas)
            c_carry, o_carry = close_ema[-1], open_ema[-1]

            for i in range(n):
                ts = timestamps[i]
                all_rows.append((
                    name, v_co, str(ts), wb.week_id, ts.weekday(),
                    close_avg[i], close_ema[i], open_avg[i], open_ema[i],
                    vol_avg[i], betas[i], alphas[i]))

            all_frames.append(pd.DataFrame({
                'Datetime': timestamps, 'ticker': name, 'company_name': v_co,
                'week_id': wb.week_id, 'weekday': [ts.weekday() for ts in timestamps],
                'close_avg': close_avg, 'close_ema': close_ema,
                'open_avg': open_avg, 'open_ema': open_ema,
                'vol_avg': vol_avg, 'beta_weight': betas, 'alpha_weight': alphas,
                'raw_close': close_v, 'raw_open': open_v, 'day_range': pre["day_range"],
            }).set_index('Datetime'))

    return all_rows, all_frames


def load(ticker, start=None, end=None):
    return store.load_table(ticker, queries.Q_5MIN, start, end)

def load_week(ticker, week_id):
    return store.load_week(ticker, week_id, queries.Q_5MIN)