# visualization/exponential.py — EMA-based charts
#
# plot_analysis(): close EMA, open EMA, beta weights
# visualize() : single or per-week chart of raw/avg/ema

import sqlite3
import pandas as pd

from trendlens import db_path
from trendlens.db import queries
from trendlens.analysis import utils as engine
from . import utils as viz_utils


def plot_analysis(result_df: pd.DataFrame, ticker: str = None,
                  interval: str = "5min", show_days: int = None,
                  figsize: tuple = (16, 10)):
    viz_utils.ensure_backend()
    import matplotlib.pyplot as plt

    if result_df is None or result_df.empty:
        print("No data to plot")
        return

    name = ticker or (result_df['ticker'].iloc[0] if 'ticker' in result_df.columns else "?")
    df = result_df
    if show_days and show_days > 0:
        dates = sorted(set(df.index.date))
        if len(dates) > show_days:
            df = df[df.index.date >= dates[-show_days]]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)
    fig.patch.set_facecolor('#FAFAFA')

    week_ids = df['week_id'].unique() if 'week_id' in df.columns else []
    ws = [df.index[df['week_id'] == w][0] for w in week_ids]
    def _vl(ax):
        for s in ws:
            ax.axvline(s, color='#bbb', lw=0.6, ls=':', alpha=0.5)

    lbl = "5min" if interval == "5min" else "hourly"

    ax = axes[0]; ax.set_facecolor('#FAFAFA')
    ax.plot(df.index, df['raw_close'], color='#aaa', lw=0.5, alpha=0.4, label='Raw')
    ax.plot(df.index, df['close_avg'], color='#0072B2', lw=1.0, label='Sliding avg')
    ax.plot(df.index, df['close_ema'], color='#D55E00', lw=1.3, label='EMA')
    _vl(ax); ax.set_ylabel('Close')
    ax.set_title(f'{name} — close ({lbl})', fontsize=13, fontweight='bold', pad=8)
    ax.legend(loc='upper left', fontsize=8)

    ax = axes[1]; ax.set_facecolor('#FAFAFA')
    ax.plot(df.index, df['raw_open'], color='#aaa', lw=0.5, alpha=0.4, label='Raw')
    ax.plot(df.index, df['open_avg'], color='#009E73', lw=1.0, label='Sliding avg')
    ax.plot(df.index, df['open_ema'], color='#CC79A7', lw=1.3, label='EMA')
    _vl(ax); ax.set_ylabel('Open')
    ax.set_title(f'{name} — open ({lbl})', fontsize=13, fontweight='bold', pad=8)
    ax.legend(loc='upper left', fontsize=8)

    ax = axes[2]; ax.set_facecolor('#FAFAFA')
    if 'beta_weight' in df.columns:
        ax.plot(df.index, df['beta_weight'], color='#E69F00', lw=0.6, alpha=0.8)
    _vl(ax); ax.set_ylabel('\u03B2')
    ax.set_title(f'{name}: beta ({lbl})', fontsize=13, fontweight='bold', pad=8)

    for ax in axes:
        ax.grid(True, axis='y', alpha=0.2)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ccc'); ax.spines['bottom'].set_color('#ccc')
        ax.tick_params(colors='#555', labelsize=9)

    plt.tight_layout(h_pad=1.5); plt.show(block=True)


# single or per-week chart of raw/avg/ema
def visualize(ticker: str, data_type: str = "close",
              start_date: str = None, end_date: str = None,
              date_range: int = None, interval: str = "5min",
              by_week: bool = False, figsize: tuple = (14, 5)):
    viz_utils.ensure_backend()
    import matplotlib.pyplot as plt

    name = engine.validate_ticker(ticker)
    if name is None:
        return

    q = queries.Q_5MIN if interval == "5min" else queries.Q_1HR

    if date_range is not None:
        end_dt = engine.yesterday()
        start_dt = end_dt - pd.Timedelta(days=date_range)
        start_date, end_date = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    elif start_date is None or end_date is None:
        start_date, end_date = engine.auto_range()

    try:
        with sqlite3.connect(db_path + "/analysis.db") as conn:
            df = pd.read_sql_query(q["select_range"], conn,
                                   params=(name, start_date, end_date))
    except Exception:
        print(f"  {name}: no analysis DB found. Run analyze() first.")
        return
    if df.empty:
        print(f"  {name}: no data in range. Run analyze() first.")
        return

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')

    avg_col, ema_col, raw_col = f"{data_type}_avg", f"{data_type}_ema", f"raw_{data_type}"
    if avg_col not in df.columns:
        print(f"  data_type '{data_type}' not found. Use 'close' or 'open'.")
        return

    lbl = "5min" if interval == "5min" else "hourly"

    if not by_week:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
        if raw_col in df.columns:
            ax.plot(df.index, df[raw_col], color='#aaa', lw=0.5, alpha=0.4, label='Raw')
        ax.plot(df.index, df[avg_col], color='#0072B2', lw=1.0, label='Sliding avg')
        if ema_col in df.columns:
            ax.plot(df.index, df[ema_col], color='#D55E00', lw=1.3, label='EMA')
        ax.set_ylabel(data_type.title())
        ax.set_title(f'{name} — {data_type} ({lbl})', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, axis='y', alpha=0.2)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); plt.show(block=True)
    else:
        if 'week_id' not in df.columns:
            print("  by_week requires week_id column"); return
        week_ids = sorted(df['week_id'].unique())
        n_weeks = len(week_ids)
        print(f"  {name}: {n_weeks} weeks to display")
        fig, axes = plt.subplots(n_weeks, 1,
                                 figsize=(figsize[0], figsize[1] * n_weeks), sharex=False)
        fig.patch.set_facecolor('#FAFAFA')
        if n_weeks == 1: axes = [axes]
        for idx, wid in enumerate(week_ids):
            ax = axes[idx]; ax.set_facecolor('#FAFAFA')
            wk = df[df['week_id'] == wid]
            n_d = len(set(wk.index.date))
            if raw_col in wk.columns:
                ax.plot(wk.index, wk[raw_col], color='#aaa', lw=0.5, alpha=0.4, label='Raw')
            ax.plot(wk.index, wk[avg_col], color='#0072B2', lw=1.0, label='Avg')
            if ema_col in wk.columns:
                ax.plot(wk.index, wk[ema_col], color='#D55E00', lw=1.3, label='EMA')
            ax.set_ylabel(data_type.title())
            ax.set_title(f'{wid} ({n_d} days)', fontsize=11, fontweight='bold')
            ax.legend(loc='upper left', fontsize=7)
            ax.grid(True, axis='y', alpha=0.2)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.suptitle(f'{name} — {data_type} by week ({lbl})',
                     fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout(); plt.show(block=True)