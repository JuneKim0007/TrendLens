# cache.py — clock-style LRU cache
#
# clock hand sweeps(LUR) on eviction: clears ref_bit, evicts first 0-ref slot
# DailyCache WeeklyCache = CacheStorage with different defaults
# AnalysisCache holds one of each

import json


MIN_SIZE, MAX_SIZE = 3, 64


class CacheStorage:

    def __init__(self, size: int = 8):
        s = max(MIN_SIZE, min(MAX_SIZE, size))
        self._cap  = s
        self._hand = 0
        self._tickers = [None] * s
        self._keys    = [None] * s
        self._refs    = [0]    * s
        self._data    = [None] * s

    def get(self, ticker: str, key: str):
        for i in range(self._cap):
            if self._tickers[i] == ticker and self._keys[i] == key:
                self._refs[i] = 1
                return self._data[i]
        return None

    def put(self, ticker: str, key: str, data):
        for i in range(self._cap):
            if self._tickers[i] == ticker and self._keys[i] == key:
                self._data[i] = data; self._refs[i] = 1; return

        slot = next((i for i in range(self._cap) if self._tickers[i] is None), None)
        if slot is None:
            slot = self._evict()

        self._tickers[slot] = ticker
        self._keys[slot]    = key
        self._data[slot]    = data
        self._refs[slot]    = 1
    # LRU clock sweep enhanced by our LLM 
    def _evict(self) -> int:
        while True:
            if self._refs[self._hand] == 0:
                slot = self._hand
                self._tickers[slot] = None
                self._keys[slot]    = None
                self._data[slot]    = None
                self._hand = (self._hand + 1) % self._cap
                return slot
            self._refs[self._hand] = 0
            self._hand = (self._hand + 1) % self._cap

    def info(self) -> dict:
        used = sum(1 for t in self._tickers if t is not None)
        return {"capacity": self._cap, "used": used}

    def __repr__(self):
        i = self.info()
        return f"CacheStorage(cap={i['capacity']}, used={i['used']})"


class DailyCache(CacheStorage):
    def __init__(self, size: int = 16):
        super().__init__(size)
    def __repr__(self):
        return f"DailyCache({super().__repr__()})"


class WeeklyCache(CacheStorage):
    def __init__(self, size: int = 8):
        super().__init__(size)
    def __repr__(self):
        return f"WeeklyCache({super().__repr__()})"


class AnalysisCache:
    def __init__(self, daily_size: int = 16, weekly_size: int = 8):
        self.daily  = DailyCache(daily_size)
        self.weekly = WeeklyCache(weekly_size)
    def __repr__(self):
        return f"AnalysisCache(daily={self.daily}, weekly={self.weekly})"

# this is primarly used for spline plotters that it first extract coefficients from the DB to form a cubic spline.
def coefficients_to_lambdas(segments: list[list[float]]) -> list:
    return [lambda t, a=a, b=b, c=c, d=d: a*t**3 + b*t**2 + c*t + d
            for a, b, c, d in segments]

def pack_coefficients(segments: list[list[float]]) -> list[str]:
    return [json.dumps(s) for s in segments]

def unpack_coefficients(json_strs: list[str]) -> list[list[float]]:
    return [json.loads(s) for s in json_strs]