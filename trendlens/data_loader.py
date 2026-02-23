# data_loader.py — user-facing API
#
# dl = DataLoader()
# dl.visualize("AAPL")
# dl.visualize("AAPL", ema_interval="5min")
# dl.visualize("AAPL", cubic_interval="week")
# dl.visualize("AAPL", start="2026-01-01", end="2026-02-01")
# dl.analyze("AAPL", "week")

import pandas as pd

from . import safetyguard as sg
from .db import ticker as ticker_db
from .db import price as price_db
from .db import store
from . import analysis as analysis_mod
from .analysis import minute, hourly, daily, weekly, utils as autils
from .visualization import depict, validate_range
from .cache import AnalysisCache


class Stock:
    def __init__(self):
        self.df = pd.DataFrame()
        self.name: str = ""

    def __repr__(self):
        if self.df.empty:
            return "Stock(Empty)"
        return f"Stock(ticker={self.name}, rows={len(self.df)})"

    @classmethod
    def from_ticker(cls, ticker):
        df = price_db.get_stock_from_yf(ticker)
        if df is None or df.empty:
            return None
        obj = cls(); obj.df = df; obj.name = ticker
        return obj

    @classmethod
    def db_from_ticker(cls, ticker):
        df = price_db.load_stock_from_db(ticker)
        if df is None or df.empty:
            return None
        obj = cls(); obj.df = df; obj.name = ticker
        return obj


class DataLoader:

    def __init__(self):
        self.stocks: dict[str, Stock] = {}
        self._cache = AnalysisCache()
        ticker_db.init_db()
        price_db.init_db()

    def _validate(self, name: str) -> str | None:
        clean = sg.validate_ticker(name)
        if clean is None:
            return None
        if not ticker_db.ticker_exists(clean):
            print(f"'{clean}' not found in local DB or Yahoo Finance")
            return None
        return clean

    def _ensure_data(self, name: str):
        if name not in self.stocks:
            stk = Stock.from_ticker(name)
            if stk is not None:
                self.stocks[name] = stk
                print(f"-{name}: fetched {len(stk.df)} rows")

    def fetch(self, *tickers: str) -> list[Stock]:
        out = []
        for t in tickers:
            name = self._validate(t)
            if name is None:
                continue
            stk = Stock.from_ticker(name)
            if stk is not None:
                self.stocks[name] = stk
                out.append(stk)
                print(f"-{name}: fetched {len(stk.df)} rows")
        return out

    def load(self, *tickers: str) -> list[Stock]:
        out = []
        for t in tickers:
            name = self._validate(t)
            if name is None:
                continue
            stk = Stock.db_from_ticker(name)
            if stk is not None:
                self.stocks[name] = stk
                out.append(stk)
                print(f"-{name}: loaded {len(stk.df)} rows from local DB")
        return out


    def analyze(self, ticker: str, mode: str = "minute",
                company_name: str = None, auto: bool = True,
                cache: 'AnalysisCache | None' = None) -> dict | None:
        return analysis_mod.analyze(
            ticker, mode=mode, company_name=company_name,
            auto=auto, storage=cache or self._cache)

    def visualize(self, ticker: str,
                  start: str = None, end: str = None,
                  ema_interval: str = "1hr",
                  cubic_interval: str = "day",
                  data_type: str = "close",
                  figsize: tuple = (18, 7)):

        # input validation
        if ema_interval not in ("5min", "1hr"):
            print(f"-ema_interval must be '5min' or '1hr', got '{ema_interval}'")
            return
        if cubic_interval not in ("day", "week"):
            print(f"-cubic_interval must be 'day' or 'week', got '{cubic_interval}'")
            return

        name = self._validate(ticker)
        if name is None:
            return

        if start is None or end is None:
            start, end = autils.auto_range()

        # step 1: ensure raw OHLCV data exists
        self._ensure_data(name)

        # step 2: validate date range against actual trading days in DB
        ok, trading_dates, warns = validate_range(name, start, end, cubic_interval)
        if not ok:
            for w in warns:
                print(f"  {w}")
            print(f"  {name}: insufficient data for {cubic_interval} spline in {start} -> {end}")
            return

        # step 3: run cascade to the required depth
        if cubic_interval == "week":
            mode = "week"
        else:
            mode = "day"

        result = self.analyze(name, mode=mode, cache=self._cache)
        if result is None:
            print(f"-{name}: analysis failed")
            return

        # step 4: extract EMA dataframe
        if ema_interval == "5min":
            ema_df = result.get("5min")
        else:
            ema_df = result.get("1hr")
            if ema_df is None:
                ema_df = result.get("5min")

        if ema_df is None:
            print(f"-{name}: no EMA data available")
            return

        #trim to date range
        ema_df = ema_df[(ema_df.index >= start) & (ema_df.index <= end + " 23:59:59")]

        #load daily spline data
        daily_splines = None
        if cubic_interval in ("day", "week"):
            daily_splines = result.get("day")
            if daily_splines is None:
                daily_splines = daily.load_spline(name, self._cache, start, end)

            if daily_splines:
                # only keep days that have EMA data
                valid = set(str(d) for d in set(ema_df.index.date))
                daily_splines = {d: v for d, v in daily_splines.items() if d in valid}
                if not daily_splines:
                    daily_splines = None
                else:
                    print(f"  {name}: {len(daily_splines)} daily splines in range")

        # one cubic spline through ALL daily interpolants, gap-split
        long_spline = None
        if cubic_interval == "week":
            long_spline = result.get("long_spline")
            if long_spline is None:
                long_spline = weekly.build_long_spline(name, start, end)

            if long_spline:
                total_pts = sum(len(s["dates"]) for s in long_spline)
                n_segs = len(long_spline)
                print(f"  {name}: long spline — {total_pts} interpolants, {n_segs} segment(s)")
            else:
                print(f"  {name}: no daily interpolants for long spline")

        #call function to plot
        depict(name, ema_df,
               daily_splines=daily_splines,
               long_spline=long_spline,
               ema_interval=ema_interval,
               data_type=data_type,
               figsize=figsize,
               warnings=warns)

    def load_analysis(self, ticker, interval="5min", start=None, end=None):
        if interval == "1hr":
            return hourly.load(ticker, start, end)
        return minute.load(ticker, start, end)

    def load_daily(self, ticker, start=None, end=None):
        return daily.load_spline(ticker, self._cache, start, end)

    def load_long_spline(self, ticker, start=None, end=None):
        return weekly.build_long_spline(ticker, start, end)

    def load_metrics(self, ticker, start=None, end=None):
        return weekly.load_metrics(ticker, start, end)

    def get(self, ticker) -> Stock | None:
        name = sg.validate_ticker(ticker)
        return self.stocks.get(name) if name else None

    def __repr__(self):
        if not self.stocks:
            return "DataLoader(empty)"
        return f"DataLoader([{', '.join(self.stocks.keys())}])"

    def __len__(self):
        return len(self.stocks)