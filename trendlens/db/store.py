# db/store.py — analysis.db access layer
#
# init_db(): creates all analysis tables
# has_data(): existence check per table
# load_table(): generic DF loader for 5min/hourly
# load_price_data():  reads OHLCV from stock_price.db (moved from analysis/utils)

import sqlite3
import pandas as pd

from trendlens import db_path
from . import queries
from . import price as price_db


def init_db():
    with sqlite3.connect(db_path + "/analysis.db") as conn:
        for t in ["analysis_5min", "analysis_1hr", "daily_ema_init",
                   "missing_segments", "daily_spline_coefficients",
                   "weekly_metrics", "weekly_spline_coefficients"]:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.execute(queries.Q_5MIN["create"])
        conn.execute(queries.Q_1HR["create"])
        conn.execute(queries.CREATE_DAILY_EMA_INIT)
        conn.execute(queries.CREATE_MISSING_SEGMENTS)
        conn.execute(queries.CREATE_DAILY_SPLINE)
        conn.execute(queries.CREATE_WEEKLY_METRICS)
        conn.execute(queries.CREATE_WEEKLY_SPLINE)
        conn.commit()


def has_data(ticker: str, table: str) -> bool:
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT 1 FROM {table} WHERE ticker = ? LIMIT 1", (ticker,))
            return cur.fetchone() is not None
    except Exception:
        return False


# weekly_metrics has placeholder zeros until analyze_weekly fills them
def weekly_is_stale(ticker: str) -> bool:
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM weekly_metrics "
                "WHERE ticker = ? AND norm_weight = 0.0 LIMIT 1", (ticker,))
            return cur.fetchone() is not None
    except Exception:
        return True


def get_ema_carry(name: str, q_dict: dict):
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            cur = conn.cursor()
            cur.execute(q_dict["last_ema"], (name,))
            row = cur.fetchone()
            if row:
                return row[0], row[1]
    except Exception:
        pass
    return None, None


def store_precomp(name, week_blocks, day_precomp):
    rows = [(name, ds, p["c_ema_init"], p["o_ema_init"], p["day_range"], p["init_weight"])
            for ds, p in day_precomp.items()]
    with sqlite3.connect(db_path + "/analysis.db") as conn:
        conn.executemany(queries.UPSERT_DAILY_EMA_INIT, rows)
        for wb in week_blocks:
            for day in wb.missing:
                conn.execute(queries.UPSERT_MISSING_SEGMENT,
                             (name, day.strftime("%Y-%m-%d"), wb.week_id, "full_day"))
        conn.commit()


# ── OHLCV loader: local db first, yfinance fallback ──

def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        with sqlite3.connect(db_path + "/stock_price.db") as conn:
            df = pd.read_sql_query(queries.SELECT_PRICE_RANGE, conn,
                                   params=(ticker, start, end))
        if not df.empty:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df = df.set_index('Datetime').sort_index()
            print(f"  {ticker}: {len(df)} rows from local DB")
            return df
    except Exception:
        pass

    print(f"  {ticker}: fetching from yfinance...")
    fetched = price_db.ensure_data(ticker, start, end)
    if fetched is None or fetched.empty:
        print(f"  {ticker}: no data for {start} → {end}")
        return None
    if isinstance(fetched.index, pd.MultiIndex):
        fetched = fetched.droplevel('ticker')
    fetched.index = pd.to_datetime(fetched.index)
    print(f"  {ticker}: fetched {len(fetched)} rows")
    return fetched.sort_index()


# ── generic analysis table loaders ──

def load_table(ticker, q_dict, start=None, end=None):
    from trendlens import safetyguard as sg
    name = sg.validate_ticker(ticker)
    if name is None:
        return None
    if start is None or end is None:
        from trendlens.analysis.utils import auto_range
        start, end = auto_range()
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            df = pd.read_sql_query(q_dict["select_range"], conn, params=(name, start, end))
        if df.empty:
            return None
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df.set_index('Datetime')
    except Exception:
        return None


def load_week(ticker, week_id, q_dict):
    from trendlens import safetyguard as sg
    name = sg.validate_ticker(ticker)
    if name is None:
        return None
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            df = pd.read_sql_query(q_dict["select_week"], conn, params=(name, week_id))
        if df.empty:
            return None
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df.set_index('Datetime')
    except Exception:
        return None


def get_week_ids(ticker, interval="5min"):
    from trendlens import safetyguard as sg
    q = queries.Q_5MIN if interval == "5min" else queries.Q_1HR
    name = sg.validate_ticker(ticker)
    if name is None:
        return []
    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            cur = conn.cursor()
            cur.execute(q["week_ids"], (name,))
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return []