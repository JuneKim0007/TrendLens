import numpy as np
import pandas as pd
from . import utils as viz_utils

PTS_PER_SEG = 100

def _eval_spline(fns: list, pts_per_seg: int = PTS_PER_SEG) -> tuple[np.ndarray, np.ndarray]:
    if not fns:
        return np.array([]), np.array([])

    n_seg = len(fns)
    total = n_seg * pts_per_seg + 1
    xs = np.linspace(0.0, float(n_seg), total)
    ys = np.empty(total)

    # single continuous sweep: map global x =>(segment_idx, local t)
    for i in range(total):
        seg = min(int(xs[i]), n_seg - 1)
        t = xs[i] - seg
        ys[i] = fns[seg](t)

    return xs, ys


def _knot_points(fns: list) -> tuple[np.ndarray, np.ndarray]:
    if not fns:
        return np.array([]), np.array([])
    kx = [float(i) for i in range(len(fns))]
    ky = [fn(0.0) for fn in fns]
    kx.append(float(len(fns)))
    ky.append(fns[-1](1.0))
    return np.array(kx), np.array(ky)


# daily spline: hourly EMA knots + cubic curve (+ linear for comparison)
def plot_daily_spline(date_str: str, lambdas: dict,
                      hourly_df: pd.DataFrame = None,
                      cols: list = None, figsize: tuple = (14, 5)):
    viz_utils.ensure_backend()
    import matplotlib.pyplot as plt

    cols = cols or [c for c in lambdas.keys() if c != "_meta"]
    if not cols:
        print(f"  No spline data for {date_str}")
        return

    fig, axes = plt.subplots(1, len(cols), figsize=figsize, squeeze=False)
    fig.patch.set_facecolor('#FAFAFA')

    # time labels for 8 hourly knots (9:00-16:00)
    hour_labels = [f"{h}:00" for h in range(9, 17)]

    for idx, col in enumerate(cols):
        ax = axes[0][idx]
        ax.set_facecolor('#FAFAFA')
        fns = lambdas.get(col)
        if fns is None:
            continue

        # knot points (the 8 hourly EMA values the spline passes through)
        kx, ky = _knot_points(fns)

        # linear interpolation (what connect-the-dots looks like)
        lin_x = np.linspace(kx[0], kx[-1], 300)
        lin_y = np.interp(lin_x, kx, ky)
        ax.plot(lin_x, lin_y, color='#aaa', lw=1.0, ls='--', alpha=0.6, label='Linear interp')

        # cubic spline (smooth continuous curve through all segments)
        sx, sy = _eval_spline(fns, PTS_PER_SEG)
        ax.plot(sx, sy, color='#D55E00', lw=2.2, alpha=0.95, label='Cubic spline')

        # knot markers
        ax.scatter(kx, ky, color='#0072B2', s=60, zorder=5,
                   edgecolors='white', linewidths=1.2, label='Hourly EMA')

        # hourly raw overlay
        if hourly_df is not None and col in hourly_df.columns:
            day_data = hourly_df[hourly_df.index.date == pd.Timestamp(date_str).date()]
            if not day_data.empty:
                vals = day_data[col].dropna().values
                ax.scatter(range(len(vals)), vals, color='#ccc', s=20, alpha=0.5,
                           marker='x', label='Raw hourly')

        # time axis
        n_knots = len(kx)
        ax.set_xticks(range(n_knots))
        ax.set_xticklabels(hour_labels[:n_knots], fontsize=8, rotation=45)
        ax.set_ylabel(col.replace('_', ' ').title())
        ax.set_title(f'{date_str} — {col} ({len(fns)} segments)',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, axis='y', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Daily Cubic Spline — {date_str}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show(block=True)


# weekly spline: daily interpolant knots + cubic curve + weights panel
def plot_weekly_spline(week_id: str, week_data: dict, figsize: tuple = (14, 5)):
    viz_utils.ensure_backend()
    import matplotlib.pyplot as plt

    meta = week_data.get("_meta", {})
    fns = week_data.get("close_ema")
    interpolants = np.array(meta.get("interpolants", []))
    weights = meta.get("weights", [])
    beta_days = meta.get("beta_days", [])
    pearson = meta.get("pearson", 0.0)
    n_days = meta.get("n_days", len(interpolants))

    if n_days == 0:
        print(f"No week data for {week_id}")
        return

    has_spline = fns is not None and len(fns) > 0
    n_panels = 2 if weights else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor('#FAFAFA')
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][:n_days]
    day_x = np.arange(n_days, dtype=float)

    ax = axes[0][0]
    ax.set_facecolor('#FAFAFA')

    # linear interpolation for comparison
    if n_days >= 2:
        lin_x = np.linspace(0, n_days - 1, 300)
        lin_y = np.interp(lin_x, day_x, interpolants)
        ax.plot(lin_x, lin_y, color='#aaa', lw=1.0, ls='--', alpha=0.6, label='Linear interp')

    if has_spline:
        sx, sy = _eval_spline(fns, PTS_PER_SEG)
        ax.plot(sx, sy, color='#D55E00', lw=2.4, alpha=0.95, label='Cubic spline')

    ax.scatter(day_x, interpolants, color='#0072B2', s=80, zorder=5,
               edgecolors='white', linewidths=1.5, label='Interpolant')

    # annotate values
    for i in range(n_days):
        ax.annotate(f'{interpolants[i]:.1f}', (day_x[i], interpolants[i]),
                    textcoords='offset points', xytext=(0, 10),
                    fontsize=7, ha='center', color='#555')

    ax.set_xticks(day_x)
    ax.set_xticklabels(day_labels, fontsize=10)
    ax.set_ylabel('Price (interpolant)')
    ax.set_title(f'{week_id} — ρ={pearson:.3f}, {n_days} days',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if n_panels > 1 and weights:
        ax2 = axes[0][1]
        ax2.set_facecolor('#FAFAFA')
        colors = ['#009E73' if w > 0 else '#D55E00' for w in weights]
        ax2.bar(day_x, weights, color=colors, alpha=0.7, width=0.6, label='Norm weight')

        if beta_days:
            ax2t = ax2.twinx()
            ax2t.plot(day_x, beta_days, 'o-', color='#E69F00', lw=1.5, ms=7, label='β_day')
            ax2t.set_ylabel('β_day', color='#E69F00')
            ax2t.tick_params(axis='y', labelcolor='#E69F00')

        ax2.set_xticks(day_x)
        ax2.set_xticklabels(day_labels, fontsize=10)
        ax2.set_ylabel('Normalized weight')
        ax2.set_title(f'{week_id} — confidence weights', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    fig.suptitle(f'Weekly Cubic Spline — {week_id}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show(block=True)


# grid view: thumbnail spline curves for all days or weeks
def plot_spline_grid(data: dict, mode: str = "daily", n_cols: int = 4,
                     figsize_per: tuple = (4, 3)):
    viz_utils.ensure_backend()
    import matplotlib.pyplot as plt

    keys = sorted(k for k in data.keys() if k != "_meta")
    n = len(keys)
    if n == 0:
        print("  No spline data to plot")
        return

    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor('#FAFAFA')

    for i, key in enumerate(keys):
        r, c = i // n_cols, i % n_cols
        ax = axes[r][c]
        ax.set_facecolor('#FAFAFA')
        entry = data[key]

        if mode == "daily":
            fns = entry.get("close_ema", [])
            if fns:
                kx, ky = _knot_points(fns)
                # linear baseline
                if len(kx) >= 2:
                    ax.plot(kx, ky, 'o--', color='#bbb', lw=0.6, ms=3, alpha=0.5)
                # cubic spline
                sx, sy = _eval_spline(fns, 40)
                ax.plot(sx, sy, color='#D55E00', lw=1.5)
                ax.scatter(kx, ky, color='#0072B2', s=15, zorder=5)
        else:
            meta = entry.get("_meta", {})
            interp = np.array(meta.get("interpolants", []))
            fns = entry.get("close_ema", [])
            if len(interp) > 0:
                ix = np.arange(len(interp))
                # linear baseline
                if len(interp) >= 2:
                    ax.plot(ix, interp, 'o--', color='#bbb', lw=0.6, ms=3, alpha=0.5)
                # cubic spline
                if fns:
                    sx, sy = _eval_spline(fns, 40)
                    ax.plot(sx, sy, color='#D55E00', lw=1.5)
                ax.scatter(ix, interp, color='#0072B2', s=15, zorder=5)

        ax.set_title(key, fontsize=9, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    label = "Daily" if mode == "daily" else "Weekly"
    fig.suptitle(f'{label} Cubic Splines — {n} {mode} curves',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show(block=True)