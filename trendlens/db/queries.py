# db/queries.py: all SQL templates
#
# ticker_list.db: ticker registry
# stock_price.db: OHLCV (open, high, Low, close, Volume) price data
# analysis.db: 5min, hourly, daily spline, weekly spline, metrics

#ticker_list.db for quickly checking the ticker input validity without making heavy db call. 

CREATE_TICKER_LIST = "CREATE TABLE IF NOT EXISTS ticker_list (ticker TEXT PRIMARY KEY)"
CHECK_TICKER = "SELECT 1 FROM ticker_list WHERE ticker = ? LIMIT 1"
INSERT_TICKER = "INSERT INTO ticker_list (ticker) VALUES (?)"

#stock_price.db─

CREATE_META_DATA = """
    CREATE TABLE IF NOT EXISTS meta_data (
        ticker TEXT PRIMARY KEY, name TEXT, sector TEXT, exchange TEXT)
"""
CREATE_PRICE_DATA = """
    CREATE TABLE IF NOT EXISTS price_data (
        ticker TEXT, Datetime TEXT,
        Open REAL, 
        Close REAL, 
        Low REAL, 
        High REAL,
        Volume INTEGER, volatility REAL,
        PRIMARY KEY (ticker, Datetime))
"""
EARLIEST_DATE     = "SELECT Datetime FROM price_data WHERE ticker = ? ORDER BY Datetime ASC LIMIT 1"
LATEST_DATE       = "SELECT Datetime FROM price_data WHERE ticker = ? ORDER BY Datetime DESC LIMIT 1"
SELECT_ALL_PRICE  = "SELECT * FROM price_data WHERE ticker = ? ORDER BY Datetime ASC"
SELECT_PRICE_AFTER = "SELECT * FROM price_data WHERE ticker = ? AND Datetime > ? ORDER BY Datetime ASC"
COUNT_IN_RANGE    = "SELECT COUNT(*) FROM price_data WHERE ticker = ? AND Datetime BETWEEN ? AND ?"
COVERAGE_BOUNDS   = "SELECT MIN(Datetime), MAX(Datetime) FROM price_data WHERE ticker = ? AND Datetime BETWEEN ? AND ?"
SELECT_PRICE_RANGE = ("SELECT Datetime, Open, Close, High, Low, Volume "
                      "FROM price_data WHERE ticker = ? AND Datetime BETWEEN ? AND ? "
                      "ORDER BY Datetime ASC")


#5min + 1hr 

def _analysis_table(table: str) -> dict:
    return {
        "create": f"""CREATE TABLE IF NOT EXISTS {table} (
            ticker TEXT, company_name TEXT, Datetime TEXT,
            week_id TEXT, weekday INTEGER,
            close_avg REAL, close_ema REAL, open_avg REAL, open_ema REAL,
            vol_avg REAL, beta_weight REAL, alpha_weight REAL,
            PRIMARY KEY (ticker, Datetime))""",
        "upsert": f"""INSERT OR REPLACE INTO {table}
            (ticker, company_name, Datetime, week_id, weekday,
             close_avg, close_ema, open_avg, open_ema,
             vol_avg, beta_weight, alpha_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        "delete_range": f"DELETE FROM {table} WHERE ticker = ? AND Datetime BETWEEN ? AND ?",
        "select_range": f"SELECT * FROM {table} WHERE ticker = ? AND Datetime BETWEEN ? AND ? ORDER BY Datetime ASC",
        "select_week": f"SELECT * FROM {table} WHERE ticker = ? AND week_id = ? ORDER BY Datetime ASC",
        "last_ema": f"SELECT close_ema, open_ema FROM {table} WHERE ticker = ? ORDER BY Datetime DESC LIMIT 1",
        "week_ids": f"SELECT DISTINCT week_id FROM {table} WHERE ticker = ? ORDER BY week_id ASC",
    }

Q_5MIN = _analysis_table("analysis_5min")
Q_1HR  = _analysis_table("analysis_1hr")


# daily_ema_init + finding missing_segments

CREATE_DAILY_EMA_INIT = """
    CREATE TABLE IF NOT EXISTS daily_ema_init (
        ticker TEXT, trade_date TEXT,
        close_ema_init REAL, open_ema_init REAL,
        day_range REAL, init_weight REAL,
        PRIMARY KEY (ticker, trade_date))
"""
CREATE_MISSING_SEGMENTS = """
    CREATE TABLE IF NOT EXISTS missing_segments (
        ticker TEXT, segment_date TEXT, week_id TEXT, segment_type TEXT,
        PRIMARY KEY (ticker, segment_date))
"""
UPSERT_DAILY_EMA_INIT = """
    INSERT OR REPLACE INTO daily_ema_init
    (ticker, trade_date, close_ema_init, open_ema_init, day_range, init_weight)
    VALUES (?, ?, ?, ?, ?, ?)
"""
UPSERT_MISSING_SEGMENT = """
    INSERT OR REPLACE INTO missing_segments
    (ticker, segment_date, week_id, segment_type) VALUES (?, ?, ?, ?)
"""


#daily_spline_coefficients

CREATE_DAILY_SPLINE = """
    CREATE TABLE IF NOT EXISTS daily_spline_coefficients (
        ticker TEXT, trade_date TEXT, data_col TEXT,
        seg_1 TEXT, seg_2 TEXT, seg_3 TEXT, seg_4 TEXT,
        seg_5 TEXT, seg_6 TEXT, seg_7 TEXT,
        PRIMARY KEY (ticker, trade_date, data_col))
"""
UPSERT_DAILY_SPLINE = """
    INSERT OR REPLACE INTO daily_spline_coefficients
    (ticker, trade_date, data_col, seg_1, seg_2, seg_3, seg_4, seg_5, seg_6, seg_7)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
SELECT_DAILY_SPLINE_RANGE = (
    "SELECT trade_date, data_col, seg_1, seg_2, seg_3, seg_4, seg_5, seg_6, seg_7 "
    "FROM daily_spline_coefficients "
    "WHERE ticker = ? AND trade_date BETWEEN ? AND ? ORDER BY trade_date ASC")
DELETE_DAILY_SPLINE_RANGE = (
    "DELETE FROM daily_spline_coefficients WHERE ticker = ? AND trade_date BETWEEN ? AND ?")


# weekly_metrics for splines. (same in structure with  daily metrics)

CREATE_WEEKLY_METRICS = """
    CREATE TABLE IF NOT EXISTS weekly_metrics (
        ticker TEXT, trade_date TEXT, week_id TEXT, weekday INTEGER,
        beta_day REAL, arc_length REAL, ema_range REAL,
        f_mean REAL, interpolant REAL,
        pearson_rho REAL, recency REAL, raw_weight REAL, norm_weight REAL,
        PRIMARY KEY (ticker, trade_date))
"""
UPSERT_WEEKLY_METRICS = """
    INSERT OR REPLACE INTO weekly_metrics
    (ticker, trade_date, week_id, weekday,
     beta_day, arc_length, ema_range, f_mean, interpolant,
     pearson_rho, recency, raw_weight, norm_weight)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
SELECT_WEEKLY_METRICS_RANGE = (
    "SELECT * FROM weekly_metrics "
    "WHERE ticker = ? AND trade_date BETWEEN ? AND ? ORDER BY trade_date ASC")
DELETE_WEEKLY_METRICS_RANGE = (
    "DELETE FROM weekly_metrics WHERE ticker = ? AND trade_date BETWEEN ? AND ?")


# weekly_spline_coefficients (4 segments from 5 daily points).

CREATE_WEEKLY_SPLINE = """
    CREATE TABLE IF NOT EXISTS weekly_spline_coefficients (
        ticker TEXT, week_id TEXT, data_col TEXT,
        seg_1 TEXT, seg_2 TEXT, seg_3 TEXT, seg_4 TEXT,
        PRIMARY KEY (ticker, week_id, data_col))
"""
UPSERT_WEEKLY_SPLINE = """
    INSERT OR REPLACE INTO weekly_spline_coefficients
    (ticker, week_id, data_col, seg_1, seg_2, seg_3, seg_4)
    VALUES (?, ?, ?, ?, ?, ?, ?)
"""
SELECT_WEEKLY_SPLINE_RANGE = (
    "SELECT week_id, data_col, seg_1, seg_2, seg_3, seg_4 "
    "FROM weekly_spline_coefficients "
    "WHERE ticker = ? AND week_id BETWEEN ? AND ? ORDER BY week_id ASC")
DELETE_WEEKLY_SPLINE_RANGE = (
    "DELETE FROM weekly_spline_coefficients WHERE ticker = ? AND week_id BETWEEN ? AND ?")