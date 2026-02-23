# analysis/ — cascade entry point
#
# each sub-module has its own analyze() + load()

from trendlens.db import store
from . import minute, hourly, daily, weekly, utils


def analyze(ticker: str, mode: str = "minute", company_name: str = None,
            auto: bool = True, storage=None) -> dict | None:
    #validate the ticker
    name = utils.validate_ticker(ticker)
    if name is None:
        return None
    
    mode = mode.lower().strip()
    #check the input
    if mode in ("minute", "5min", "min"):
        if auto and store.has_data(name, "analysis_5min"):
            print(f"  {name}: 5min analysis DB already exists")
            return {"5min": minute.load(name)}
        return {"5min": minute.analyze(name, company_name)}

    if mode in ("hour", "1hr", "hourly", "1h"):
        if auto and store.has_data(name, "analysis_5min") and store.has_data(name, "analysis_1hr"):
            print(f"  {name}: hourly analysis DB already exists")
            return {"5min": minute.load(name), "1hr": hourly.load(name)}
        if not store.has_data(name, "analysis_5min"):
            minute.analyze(name, company_name)
        return {"5min": minute.load(name), "1hr": hourly.analyze(name, company_name)}

    if mode in ("day", "daily"):
        if auto:
            #check if db has data first since this is cummulative, to analyze weekly, we need 5-min,hourly, and daily data.
            h5 = store.has_data(name, "analysis_5min")
            hh = store.has_data(name, "analysis_1hr")
            hd = store.has_data(name, "daily_spline_coefficients")
            if h5 and hh and hd:
                print(f"  {name}: daily analysis DB already exists")
                #cache for daily splines since computation heavy but less memory pressure. 
                if storage:
                    daily.populate_cache(name, storage.daily)
                return {"5min": minute.load(name), "1hr": hourly.load(name),
                        "day": daily.load_spline(name, storage)}
            if not h5: minute.analyze(name, company_name)
            if not hh: hourly.analyze(name, company_name)
        else:
            minute.analyze(name, company_name)
            hourly.analyze(name, company_name)
        return {"5min": minute.load(name), "1hr": hourly.load(name),
                "day": daily.analyze(name, company_name, storage)}

    if mode in ("week", "weekly"):
        if auto:
            has_min = store.has_data(name, "analysis_5min")
            has_hr = store.has_data(name, "analysis_1hr")
            has_day = store.has_data(name, "daily_spline_coefficients")
            has_wk = store.has_data(name, "weekly_metrics")
            stale = store.weekly_is_stale(name) if has_wk else True

            if has_min and has_hr and has_day and has_wk and not stale:
                print(f"  {name}: weekly analysis DB already exists")
                if storage:
                    daily.populate_cache(name, storage.daily)
                return {"5min": minute.load(name), "1hr": hourly.load(name),
                        "day": daily.load_spline(name, storage),
                        "long_spline": weekly.build_long_spline(name)}

            if not has_min: minute.analyze(name, company_name)
            if not has_hr:  hourly.analyze(name, company_name)
            if not has_day: daily.analyze(name, company_name, storage)
            if not has_wk or stale:
                weekly.analyze(name, company_name, storage)
        else:
            minute.analyze(name, company_name)
            hourly.analyze(name, company_name)
            daily.analyze(name, company_name, storage)
            weekly.analyze(name, company_name, storage)

        return {"5min": minute.load(name), "1hr": hourly.load(name),
                "day": daily.load_spline(name, storage),
                "long_spline": weekly.build_long_spline(name)}

    print(f" Unknown analyzation interval: {mode}. Use one of [minute, hour, day, or week].")
    return None