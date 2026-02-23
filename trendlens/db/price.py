# db/price.py — stock_price.db: fetch from yfinance, read/write local
#
# get_stock_from_yf(): 60d/5min fetch, merges into local db
# load_stock_from_db():reads from local db by length window
# ensure_data(): local first, yfinance fallback

import sqlite3
import pandas as pd
import yfinance as yf

from trendlens import db_path
from . import queries
from .ticker import ticker_exists


def init_db():
    with sqlite3.connect(db_path + "/stock_price.db") as conn:
        conn.execute(queries.CREATE_META_DATA)
        conn.execute(queries.CREATE_PRICE_DATA)
        conn.commit()


# fetches 60d/5min from yfinance, appends new rows to local db
def get_stock_from_yf(ticker_name: str) -> pd.DataFrame:
    ticker_name = ticker_name.upper()
    if not ticker_exists(ticker_name):
        return None

    temp = yf.Ticker(ticker_name)
    data = temp.history(period="60d", interval="5m")[['Open', 'Close', 'High', 'Low', 'Volume']]
    data['ticker'] = ticker_name
    data = data.reset_index().set_index(['ticker', 'Datetime'])
    data.index = data.index.set_levels(
        data.index.levels[1].tz_localize(None), level=1)

    with sqlite3.connect(db_path + '/stock_price.db') as conn:
        try:
            cur = conn.cursor()
            cur.execute(queries.EARLIEST_DATE, (ticker_name,))
            db_earliest = pd.to_datetime(cur.fetchone()[0], errors='coerce')
            filtered = data[data.index.get_level_values('Datetime') < db_earliest]
            if filtered.empty:
                print(f"DB already up-to-date for {ticker_name}")
                return data
            to_insert = data.loc[:filtered.index[-1]]
            to_insert.to_sql('price_data', conn, if_exists='append', index=True)
            print(f"Inserted {len(to_insert)} new rows for {ticker_name}")
        except Exception:
            try:
                data.to_sql('price_data', conn, if_exists='append',
                            index=True, index_label=['ticker', 'Datetime'])
                print(f"Saved {ticker_name} to stock_price.db")
            except Exception:
                print(f"ERROR: failed to insert {ticker_name} into stock_price.db")
                return None
    return data


# reads from local db for a specified length window
def load_stock_from_db(ticker_name: str, length: str = '60d') -> pd.DataFrame:
    ticker_name = ticker_name.upper()
    if not ticker_exists(ticker_name):
        return None
    try:
        with sqlite3.connect(db_path + "/stock_price.db") as conn:
            cur = conn.cursor()
            cur.execute(queries.LATEST_DATE, (ticker_name,))
            latest = pd.to_datetime(cur.fetchone()[0], errors='coerce')
            cur.execute(queries.EARLIEST_DATE, (ticker_name,))
            earliest = pd.to_datetime(cur.fetchone()[0], errors='coerce')

            if (latest - earliest) < pd.Timedelta(length):
                df = pd.read_sql_query(queries.SELECT_ALL_PRICE, conn, params=(ticker_name,))
            else:
                cutoff = earliest - pd.Timedelta(length)
                df = pd.read_sql_query(queries.SELECT_PRICE_AFTER, conn,
                                       params=(ticker_name, str(cutoff)))
            return df.reset_index().set_index(['ticker', 'Datetime'])
    except Exception:
        print(f"ERROR: failed to read stock_price.db")
        return None


# checks how much data exists locally for a date range
def check_coverage(ticker: str, start: str, end: str) -> tuple[int, str | None, str | None]:
    ticker = ticker.upper()
    try:
        with sqlite3.connect(db_path + "/stock_price.db") as conn:
            cur = conn.cursor()
            cur.execute(queries.COUNT_IN_RANGE, (ticker, start, end))
            count = cur.fetchone()[0]
            if count == 0:
                return (0, None, None)
            cur.execute(queries.COVERAGE_BOUNDS, (ticker, start, end))
            row = cur.fetchone()
            return (count, row[0], row[1])
    except Exception:
        return (0, None, None)


# checks local first, yfinance fallback for gaps
def ensure_data(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    ticker = ticker.upper()
    count, _, _ = check_coverage(ticker, start, end)
    if count > 0:
        expected = max(1, int((pd.to_datetime(end) - pd.to_datetime(start)).days * 252 / 365)) * 78
        if count / expected > 0.8:
            return _read_range(ticker, start, end)

    fetched = _fetch_range_yf(ticker, start, end)
    if fetched is not None:
        return fetched
    if count > 0:
        return _read_range(ticker, start, end)
    return None


def _read_range(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        with sqlite3.connect(db_path + "/stock_price.db") as conn:
            df = pd.read_sql_query(queries.SELECT_PRICE_RANGE, conn,
                                   params=(ticker, start, end))
            if df.empty:
                return None
            df['ticker'] = ticker
            return df.set_index(['ticker', 'Datetime'])
    except Exception:
        return None


def _fetch_range_yf(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    ticker = ticker.upper()
    if not ticker_exists(ticker):
        return None
    delta = (pd.to_datetime(end) - pd.to_datetime(start)).days
    interval = "5m" if delta <= 60 else ("1h" if delta <= 730 else "1d")
    try:
        data = yf.Ticker(ticker).history(start=start, end=end, interval=interval)[
            ['Open', 'Close', 'High', 'Low', 'Volume']]
        if data.empty:
            return None
        data['ticker'] = ticker
        data = data.reset_index().set_index(['ticker', 'Datetime'])
        data.index = data.index.set_levels(data.index.levels[1].tz_localize(None), level=1)
        with sqlite3.connect(db_path + '/stock_price.db') as conn:
            data.to_sql('price_data', conn, if_exists='append',
                        index=True, index_label=['ticker', 'Datetime'])
        print(f"Fetched {len(data)} rows for {ticker} ({interval}, {start} → {end})")
        return data
    except Exception as e:
        print(f"ERROR: yfinance fetch failed for {ticker}: {e}")
        return None