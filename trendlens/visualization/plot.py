
import sqlite3
import numpy as np
import pandas as pd

from trendlens import db_path
from . import utils as viz_utils

_PTS = 80


def _spline_to_datetimes(fns, knot_ts):
    n_seg = len(fns)
    if n_seg == 0 or len(knot_ts) < n_seg + 1:
        return [], np.array([])

    total = n_seg * _PTS + 1
    xs = np.linspace(0.0, float(n_seg), total)
    dt_out = []
    y_out = np.empty(total)

    for i in range(total):
        seg = min(int(xs[i]), n_seg - 1)
        t = xs[i] - seg
        y_out[i] = fns[seg](t)
        t0 = knot_ts[seg]
        t1 = knot_ts[min(seg + 1, len(knot_ts) - 1)]
        dt_out.append(t0 + t * (t1 - t0))

    return dt_out, y_out

def validate_range(ticker, start, end, cubic_interval):
    warns = []
    trading_dates = []

    try:
        with sqlite3.connect(db_path + "/stock_price.db") as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT DATE(Datetime) FROM price_data "
                "WHERE ticker = ? AND Datetime BETWEEN ? AND ? "
                "ORDER BY Datetime ASC",
                (ticker, start, end + " 23:59:59"))
            trading_dates = [r[0] for r in cur.fetchall()]
    except Exception:
        warns.append("could not query trading dates from stock_price.db")
        return False, [], warns

    if not trading_dates:
        warns.append(f"no trading data in {start} -> {end}")
        return False, trading_dates, warns

    if cubic_interval == "week":
        if len(trading_dates) < 3:
            warns.append(f"need >= 3 trading days for long spline, found {len(trading_dates)}")
            return False, trading_dates, warns

    return True, trading_dates, warns



def depict(
    name, ema_df,
    daily_splines=None, long_spline=None,
    ema_interval="1hr", data_type="close",
    figsize=(18, 7), warnings=None,
):
    viz_utils.ensure_backend()
    import matplotlib.pyplot as plt

    if ema_df is None or ema_df.empty:
        print("  no EMA data to plot")
        return

    ema_col = f"{data_type}_ema"
    avg_col = f"{data_type}_avg"
    raw_col = f"raw_{data_type}"
    has_daily = daily_splines is not None and len(daily_splines) > 0
    has_long = long_spline is not None and len(long_spline) > 0

    if warnings:
        for w in warnings:
            print(f" warning: {w}")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    #AVG/ raw data
    if raw_col in ema_df.columns:
        ax.plot(ema_df.index, ema_df[raw_col],
                color='#d0d0d0', lw=0.3, alpha=0.5, label='Raw price')
    if avg_col in ema_df.columns:
        ax.plot(ema_df.index, ema_df[avg_col],
                color='#0072B2', lw=0.6, alpha=0.45, label='Average')
    #EMA
    if ema_col in ema_df.columns:
        ax.plot(ema_df.index, ema_df[ema_col],
                color='#E69F00', lw=0.9, alpha=0.65,
                label=f'EMA ({ema_interval})')
    #cupib splines
    if has_daily:
        drawn = False
        for date_str in sorted(daily_splines):
            fns = daily_splines[date_str].get(f"{data_type}_ema")
            if not fns:
                continue

            day_date = pd.Timestamp(date_str).date()
            day_ema = ema_df[ema_df.index.date == day_date]
            if day_ema.empty:
                continue

            if ema_interval in ("5min", "min"):
                knot_ts = day_ema.resample('1h').first().dropna(how='all').index.tolist()
                if len(knot_ts) < 2:
                    knot_ts = day_ema.index[::12].tolist()
            else:
                knot_ts = day_ema.index.tolist()

            n_seg = len(fns)
            if len(knot_ts) < n_seg + 1:
                continue
            knot_ts = knot_ts[:n_seg + 1]

            curve_dt, curve_y = _spline_to_datetimes(fns, knot_ts)
            if len(curve_dt) > 0:
                ax.plot(curve_dt, curve_y,
                        color='#CC0000', lw=1.8, alpha=0.75,
                        label='Daily spline' if not drawn else None)

                drawn = True

    if has_long:
        for si, seg in enumerate(long_spline):
            seg_dates = seg["dates"]
            seg_fns = seg["fns"]
            n = len(seg_dates)
            if n == 0:
                continue

            # map each date to midday of that trading day
            knot_ts = [pd.Timestamp(d).replace(hour=12, minute=30) for d in seg_dates]

            # cubic spline curve
            if seg_fns is not None and n >= 2:
                n_seg = min(len(seg_fns), n - 1)
                cdt, cy = _spline_to_datetimes(seg_fns[:n_seg], knot_ts[:n_seg + 1])
                if len(cdt) > 0:
                    ax.plot(cdt, cy, color='#7B2D8E', lw=2.8, alpha=0.9,
                            label='Long spline' if si == 0 else None)


    # week boundary lines
    if 'week_id' in ema_df.columns:
        for wid in ema_df['week_id'].unique():
            first_idx = ema_df[ema_df['week_id'] == wid].index[0]
            ax.axvline(first_idx, color='#e8e8e8', lw=0.3, ls=':', alpha=0.5)

    lbl = "5min" if ema_interval in ("5min", "min") else "hourly"
    layers = [f'{data_type} EMA ({lbl})']
    if has_daily:
        layers.append('daily spline')
    if has_long:
        layers.append('long spline')

    ax.set_ylabel(f'{data_type.title()} Price')
    ax.set_title(f'{name}=={" + ".join(layers)}',
                 fontsize=13, fontweight='bold', pad=8)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
    ax.grid(True, axis='y', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show(block=True)