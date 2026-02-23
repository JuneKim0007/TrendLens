# analysis/weights.py — per-bar weight system (EQ1-19)
#
# 5-min: day_range precomputed → per-bar weights → beta/alpha → EMA
# hourly: aggregated 5-min betas → hourly EMA
# daily:  spline arc length → beta_day → interpolant blend
# weekly: pearson correlation → recency → normalized weights

import numpy as np


EQUATIONS = {
    "EQ1":  "EMA: ema = (1-β)*prev + β*sample",
    "EQ2":  "EMA init: ((1+close/open)/2) * ((close+open)/2)",
    "EQ3":  "Day range: max(High) - min(Low)",
    "EQ4":  "Init weight: (high_0 - low_0) / day_range",
    "EQ5":  "Bar weight: w[i] = (w[i-1] + (high-low)) / day_range",
    "EQ6":  "Beta: clamp(w, 0.001, 0.95)",
    "EQ7":  "Alpha: clamp(β*0.8, 0.001, 0.80)",
    "EQ8":  "Sliding avg: 13-bar centered window",
    "EQ9":  "Hourly β: mean(5min_betas[j*12:(j+1)*12])",
    "EQ10": "Natural cubic spline: S(t) = at³+bt²+ct+d",
    "EQ11": "Arc length: 5-pt Gauss quadrature",
    "EQ12": "Day EMA range: max(ema_hr) - min(ema_hr)",
    "EQ13": "β_day = 1 - range/arc_length",
    "EQ14": "Spline integral mean: (1/N)Σ(a/4+b/3+c/2+d)",
    "EQ15": "Interpolant: (1-β_day)*median + β_day*f_mean",
    "EQ16": "Pearson ρ: corr([0..4], interpolants)",
    "EQ17": "Recency: exp(|ρ|*i/(n-1))",
    "EQ18": "Raw weight: (1-β_day)*recency",
    "EQ19": "Normalized: w_i / Σw_j",
}

BETA_FLOOR  = 0.001
BETA_CAP    = 0.95
ALPHA_RATIO = 0.8
ALPHA_FLOOR = 0.001
ALPHA_CAP   = 0.80


def compute_day_range(highs: np.ndarray, lows: np.ndarray) -> float:
    dr = np.nanmax(highs) - np.nanmin(lows)
    return dr if dr > 1e-10 else 1e-10

def compute_bar_weights(highs: np.ndarray, lows: np.ndarray, day_range: float) -> np.ndarray:
    n = len(highs)
    w = np.empty(n)
    br = highs - lows
    w[0] = br[0] / day_range
    for i in range(1, n):
        w[i] = (w[i - 1] + br[i]) / day_range
    return w

def weights_to_beta(w: np.ndarray) -> np.ndarray:
    return np.clip(w, BETA_FLOOR, BETA_CAP)

def beta_to_alpha(beta: np.ndarray) -> np.ndarray:
    return np.clip(beta * ALPHA_RATIO, ALPHA_FLOOR, ALPHA_CAP)

# EQ9
def compute_hourly_betas(betas_5min: np.ndarray, bars_per_hour: int = 12) -> np.ndarray:
    n = len(betas_5min)
    n_hours = (n + bars_per_hour - 1) // bars_per_hour
    hourly = np.empty(n_hours)
    for j in range(n_hours):
        s = j * bars_per_hour
        chunk = betas_5min[s : min(s + bars_per_hour, n)]
        hourly[j] = chunk.sum() / bars_per_hour
    return np.clip(hourly, BETA_FLOOR, BETA_CAP)

def compute_ema_init(open_p: float, close_p: float) -> float:
    if open_p <= 0 or np.isnan(open_p):
        return close_p if not np.isnan(close_p) else 0.0
    return (1.0 + close_p / open_p) / 2.0 * (close_p + open_p) / 2.0

def compute_ema_variable_beta(samples: np.ndarray, init_val: float, betas: np.ndarray) -> np.ndarray:
    n = len(samples)
    ema = np.empty(n)
    ema[0] = (1.0 - betas[0]) * init_val + betas[0] * samples[0]
    for i in range(1, n):
        ema[i] = (1.0 - betas[i]) * ema[i - 1] + betas[i] * samples[i]
    return ema

#Gauss-Legendre quadrature
_GL_NODES = np.array([
    0.5 - 0.5 * 0.906179845938664,
    0.5 - 0.5 * 0.538469310105683,
    0.5,
    0.5 + 0.5 * 0.538469310105683,
    0.5 + 0.5 * 0.906179845938664,
])
_GL_WEIGHTS = np.array([
    0.5 * 0.236926885056189,
    0.5 * 0.478628670499366,
    0.5 * 0.568888888888888,
    0.5 * 0.478628670499366,
    0.5 * 0.236926885056189,
])

### to find interpolants to build upon weekly splines.
def spline_arc_length(segments: list[list[float]]) -> float:
    total = 0.0
    for a, b, c, d in segments:
        for t, w in zip(_GL_NODES, _GL_WEIGHTS):
            deriv = 3.0 * a * t * t + 2.0 * b * t + c
            total += w * np.sqrt(1.0 + deriv * deriv)
    return total

def spline_integral_mean(segments: list[list[float]]) -> float:
    n = len(segments)
    if n == 0:
        return 0.0
    return sum(a / 4.0 + b / 3.0 + c / 2.0 + d for a, b, c, d in segments) / n

def compute_beta_day(ema_values: np.ndarray, segments: list[list[float]]) -> float:
    rng = float(np.nanmax(ema_values) - np.nanmin(ema_values))
    arc = spline_arc_length(segments)
    if arc < 1e-10:
        return 0.0
    return max(0.0, min(1.0, 1.0 - rng / arc))

def compute_daily_interpolant(ema_values: np.ndarray, segments: list[list[float]], beta_day: float) -> float:
    median_val = float(np.nanmedian(ema_values))
    f_mean = spline_integral_mean(segments)
    return (1.0 - beta_day) * median_val + beta_day * f_mean
#####

def compute_weekly_weights(interpolants: np.ndarray, beta_days: np.ndarray) -> tuple[float, np.ndarray]:
    n = len(interpolants)
    if n < 2:
        return 0.0, np.ones(n) / max(n, 1)

    day_idx = np.arange(n, dtype=float)
    std_i = np.std(interpolants)
    if std_i < 1e-10:
        rho = 0.0
    else:
        rho = float(np.corrcoef(day_idx, interpolants)[0, 1])
        if np.isnan(rho):
            rho = 0.0

    abs_rho = abs(rho)
    recency = np.exp(abs_rho * day_idx / max(n - 1, 1))
    confidence = 1.0 - beta_days
    raw_w = confidence * recency
    total = raw_w.sum()
    weights = raw_w / total if total > 1e-10 else np.ones(n) / n
    return rho, weights