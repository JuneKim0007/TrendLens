
from trendlens.data_loader import DataLoader

dl = DataLoader()

# default: 60d range, hourly EMA + daily cubic spline
dl.visualize("AAPL")

# dl.visualize("AAPL", ema_interval="5min")

# dl.visualize("AAPL", cubic_interval="week")

# dl.visualize("AAPL", start="2026-01-01", end="2026-02-15", cubic_interval="week")

# dl.visualize("META", ema_interval="5min", cubic_interval="day")
dl.visualize("AMZN", ema_interval="1hr", cubic_interval="week")