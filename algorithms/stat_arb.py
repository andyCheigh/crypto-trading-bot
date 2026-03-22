"""
Algorithm 1: Statistical Arbitrage / Mean Reversion

Used extensively at prop trading firms for market-neutral alpha generation.
Core idea: prices oscillate around a fair value. When they deviate significantly
(measured by z-score), they tend to revert.

Signals:
- Z-score of price relative to rolling mean (Ornstein-Uhlenbeck process proxy)
- Hurst exponent to confirm mean-reverting regime
- Bollinger Band width for volatility regime detection
- RSI extremes for confirmation

Buy: z-score < -2.0 (oversold), Hurst < 0.5 (mean-reverting), RSI < 30
Sell: z-score > 2.0 (overbought), or mean reversion target hit
"""

import numpy as np
import pandas as pd

from data_fetcher import StockData


class StatArbAlgorithm:
    name = "stat_arb"

    def __init__(self):
        self.lookback = 20   # rolling window
        self.z_buy = -1.8    # z-score threshold to buy
        self.z_sell = 1.8    # z-score threshold to sell
        self.z_exit = 0.0    # exit when reverts to mean

    def _hurst_exponent(self, series: pd.Series, max_lag: int = 20) -> float:
        """Estimate Hurst exponent via R/S analysis.
        H < 0.5: mean-reverting, H = 0.5: random walk, H > 0.5: trending.
        """
        if len(series) < max_lag * 2:
            return 0.5

        lags = range(2, max_lag)
        rs_values = []
        for lag in lags:
            chunks = [series[i:i + lag] for i in range(0, len(series) - lag, lag)]
            rs_for_lag = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean_adj = chunk - chunk.mean()
                cumdev = mean_adj.cumsum()
                r = cumdev.max() - cumdev.min()
                s = chunk.std()
                if s > 0:
                    rs_for_lag.append(r / s)
            if rs_for_lag:
                rs_values.append((np.log(lag), np.log(np.mean(rs_for_lag))))

        if len(rs_values) < 2:
            return 0.5

        x = np.array([v[0] for v in rs_values])
        y = np.array([v[1] for v in rs_values])
        hurst = np.polyfit(x, y, 1)[0]
        return float(np.clip(hurst, 0.0, 1.0))

    def _half_life(self, series: pd.Series) -> float:
        """Mean reversion half-life via OLS on lagged series.
        Shorter half-life = faster reversion = stronger signal.
        """
        if len(series) < 10:
            return float("inf")
        lag = series.shift(1).dropna()
        delta = series.diff().dropna()
        common = min(len(lag), len(delta))
        lag = lag.iloc[-common:]
        delta = delta.iloc[-common:]

        if lag.std() == 0:
            return float("inf")

        beta = np.cov(delta, lag)[0, 1] / np.var(lag)
        if beta >= 0:
            return float("inf")
        return float(-np.log(2) / beta)

    def buy_score(self, data: StockData) -> float:
        """Return score 0-1 indicating buy conviction."""
        if data.ohlcv is None or len(data.ohlcv) < 30:
            return 0.0

        close = data.ohlcv["Close"]
        score = 0.0

        # Z-score component (40% weight)
        rolling_mean = close.rolling(self.lookback).mean()
        rolling_std = close.rolling(self.lookback).std()
        z = float((close.iloc[-1] - rolling_mean.iloc[-1]) / (rolling_std.iloc[-1] + 1e-10))

        if z < self.z_buy:
            # Deeper deviation = stronger signal, cap at -4
            z_score = min(abs(z) / 4.0, 1.0)
            score += 0.40 * z_score

        # Hurst exponent component (25% weight) - want < 0.5
        hurst = self._hurst_exponent(close)
        if hurst < 0.5:
            hurst_score = (0.5 - hurst) / 0.5  # 0 at H=0.5, 1 at H=0.0
            score += 0.25 * hurst_score

        # Half-life component (15% weight) - shorter is better
        hl = self._half_life(close)
        if hl < 20 and hl > 0:
            hl_score = max(0, 1.0 - hl / 20.0)
            score += 0.15 * hl_score

        # RSI confirmation (10% weight)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = float(100 - (100 / (1 + rs.iloc[-1])))
        if rsi < 30:
            score += 0.10 * ((30 - rsi) / 30.0)

        # Options confirmation: high put/call ratio = fear = contrarian buy (10%)
        if data.options.put_call_ratio > 1.2:
            pcr_score = min((data.options.put_call_ratio - 1.0) / 1.0, 1.0)
            score += 0.10 * pcr_score

        return round(min(score, 1.0), 4)

    def sell_score(self, data: StockData, entry_price: float) -> float:
        """Return score 0-1 indicating sell conviction."""
        if data.ohlcv is None or len(data.ohlcv) < 30:
            return 0.0

        close = data.ohlcv["Close"]
        score = 0.0

        # Z-score reversion to mean or overshoot
        rolling_mean = close.rolling(self.lookback).mean()
        rolling_std = close.rolling(self.lookback).std()
        z = float((close.iloc[-1] - rolling_mean.iloc[-1]) / (rolling_std.iloc[-1] + 1e-10))

        if z >= self.z_exit:
            # Reverted to mean or above - signal to take profit
            z_score = min(z / self.z_sell, 1.0)
            score += 0.50 * z_score

        # Hurst now > 0.5 means regime changed to trending (bad for mean reversion)
        hurst = self._hurst_exponent(close)
        if hurst > 0.6:
            score += 0.30 * min((hurst - 0.5) / 0.3, 1.0)

        # P&L based exit
        pnl_pct = (data.price - entry_price) / entry_price
        if pnl_pct > 0.03:  # Take profit at 3%+
            score += 0.20

        return round(min(score, 1.0), 4)
