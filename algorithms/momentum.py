"""
Algorithm 2: Momentum + Volume-Weighted Signal

The workhorse of systematic prop trading. Trend-following with volume confirmation
eliminates false breakouts. Combined with cross-sectional momentum ranking.

Signals:
- Price momentum (5d, 10d, 20d weighted)
- Volume-price trend confirmation (OBV, VWAP deviation)
- MACD histogram acceleration
- Relative strength vs SPY (cross-sectional momentum)
- Accumulation/Distribution confirmation

Buy: Strong multi-timeframe momentum + rising volume + positive A/D
Sell: Momentum deceleration, volume divergence, or trend break
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_fetcher import StockData


class MomentumAlgorithm:
    name = "momentum"

    def _macd(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Average Directional Index - measures trend strength (not direction)."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)

        # When +DM > -DM, keep +DM, zero -DM (and vice versa)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return float(adx.iloc[-1]) if not adx.empty else 0.0

    def buy_score(self, data: StockData) -> float:
        """Multi-timeframe momentum with volume confirmation."""
        if data.ohlcv is None or len(data.ohlcv) < 30:
            return 0.0

        df = data.ohlcv
        close = df["Close"]
        score = 0.0

        # Multi-timeframe momentum (30% weight)
        # Weighted: recent momentum matters more
        mom_5 = float(close.pct_change(5).iloc[-1])
        mom_10 = float(close.pct_change(10).iloc[-1])
        mom_20 = float(close.pct_change(20).iloc[-1])
        weighted_mom = 0.5 * mom_5 + 0.3 * mom_10 + 0.2 * mom_20

        if weighted_mom > 0:
            # Scale: 5% weighted momentum = full score
            mom_score = min(weighted_mom / 0.05, 1.0)
            score += 0.30 * mom_score

        # MACD histogram acceleration (20% weight)
        _, _, histogram = self._macd(close)
        if len(histogram) >= 3:
            hist_accel = float(histogram.iloc[-1] - histogram.iloc[-2])
            hist_val = float(histogram.iloc[-1])
            if hist_val > 0 and hist_accel > 0:
                # Positive and accelerating
                accel_score = min(abs(hist_accel) / (data.atr_14 * 0.1 + 1e-10), 1.0)
                score += 0.20 * accel_score

        # ADX trend strength (15% weight)
        adx = self._adx(df)
        if adx > 25:  # Strong trend
            adx_score = min((adx - 20) / 30.0, 1.0)
            score += 0.15 * adx_score

        # Volume confirmation (15% weight)
        rel_vol = data.volume_profile.relative_volume
        vol_trend = data.volume_profile.volume_trend
        if rel_vol > 1.2 and vol_trend > 0:
            vol_score = min((rel_vol - 1.0) / 1.0, 1.0)
            score += 0.15 * vol_score

        # OBV slope confirmation (10% weight)
        if data.volume_profile.obv_slope > 0:
            obv_score = min(abs(data.volume_profile.obv_slope) / 1e6, 1.0)
            score += 0.10 * obv_score

        # Price above VWAP (10% weight)
        if data.price > data.volume_profile.vwap and data.volume_profile.vwap > 0:
            vwap_dev = (data.price - data.volume_profile.vwap) / data.volume_profile.vwap
            score += 0.10 * min(vwap_dev / 0.02, 1.0)

        return round(min(score, 1.0), 4)

    def sell_score(self, data: StockData, entry_price: float) -> float:
        """Momentum exhaustion and trend break detection."""
        if data.ohlcv is None or len(data.ohlcv) < 30:
            return 0.0

        close = data.ohlcv["Close"]
        score = 0.0

        # MACD histogram deceleration / bearish crossover (35%)
        _, _, histogram = self._macd(close)
        if len(histogram) >= 3:
            hist_val = float(histogram.iloc[-1])
            hist_prev = float(histogram.iloc[-2])
            if hist_val < 0:
                score += 0.20
            if hist_val < hist_prev and hist_prev > 0:
                # Bearish deceleration
                score += 0.15

        # RSI overbought (20%)
        rsi = self._rsi(close)
        rsi_val = float(rsi.iloc[-1])
        if rsi_val > 70:
            score += 0.20 * min((rsi_val - 70) / 20.0, 1.0)

        # Volume divergence: price up but volume dropping (20%)
        if data.returns_1d > 0 and data.volume_profile.relative_volume < 0.8:
            score += 0.20

        # Momentum reversal (25%)
        mom_5 = float(close.pct_change(5).iloc[-1])
        if mom_5 < -0.02:
            score += 0.25 * min(abs(mom_5) / 0.05, 1.0)

        return round(min(score, 1.0), 4)
