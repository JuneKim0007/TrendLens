# db/ticker.py — ticker_list.db: check/register valid tickers via yfinance

import sqlite3
import yfinance as yf

from trendlens import db_path
from . import queries


def init_db():
    with sqlite3.connect(db_path + "/ticker_list.db") as conn:
        conn.execute(queries.CREATE_TICKER_LIST)
        conn.commit()


# checks local db first, falls back to yfinance validation
def ticker_exists(name: str) -> bool:
    with sqlite3.connect(db_path + "/ticker_list.db") as conn:
        cur = conn.cursor()
        cur.execute(queries.CHECK_TICKER, (name,))
        if cur.fetchone() is not None:
            return True
        try:
            info = yf.Ticker(name).info
            if info.get("regularMarketPrice") is None:
                return False
            cur.execute(queries.INSERT_TICKER, (name,))
            conn.commit()
            return True
        except Exception:
            return False