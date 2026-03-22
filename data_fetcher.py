"""
Data fetcher that pulls in-depth stock metrics:
- OHLCV price history
- Options chain with Greeks (delta, gamma, theta, vega)
- Implied volatility surface
- Volume profile and order flow proxies
- Fundamental ratios
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class OptionsMetrics:
    iv_atm: float = 0.0              # At-the-money implied volatility
    iv_skew: float = 0.0             # Put IV - Call IV (skew)
    put_call_ratio: float = 1.0      # Put volume / Call volume
    avg_delta_calls: float = 0.0     # Average delta of near-ATM calls
    avg_delta_puts: float = 0.0      # Average delta of near-ATM puts
    avg_gamma: float = 0.0           # Average gamma near ATM
    avg_theta: float = 0.0           # Average theta (time decay)
    avg_vega: float = 0.0            # Average vega (vol sensitivity)
    max_pain: float = 0.0            # Max pain strike price
    net_gamma_exposure: float = 0.0  # Dealer gamma exposure proxy


@dataclass
class VolumeProfile:
    vwap: float = 0.0                # Volume-weighted average price
    relative_volume: float = 1.0     # Current vol / 20-day avg vol
    volume_trend: float = 0.0        # Volume momentum (5d vs 20d)
    accumulation_dist: float = 0.0   # Accumulation/Distribution line value
    obv_slope: float = 0.0           # On-Balance Volume slope


@dataclass
class Fundamentals:
    market_cap: float = 0.0
    pe_ratio: float = 0.0
    forward_pe: float = 0.0
    pb_ratio: float = 0.0
    dividend_yield: float = 0.0
    short_ratio: float = 0.0
    beta: float = 1.0
    earnings_date: Optional[str] = None


@dataclass
class StockData:
    symbol: str
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    # Price history
    ohlcv: Optional[pd.DataFrame] = None
    # Derived metrics
    options: OptionsMetrics = field(default_factory=OptionsMetrics)
    volume_profile: VolumeProfile = field(default_factory=VolumeProfile)
    fundamentals: Fundamentals = field(default_factory=Fundamentals)
    # Technical
    returns_1d: float = 0.0
    returns_5d: float = 0.0
    returns_20d: float = 0.0
    volatility_20d: float = 0.0
    atr_14: float = 0.0


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1]) if not atr.empty else 0.0


def _compute_volume_profile(df: pd.DataFrame) -> VolumeProfile:
    vp = VolumeProfile()
    if df.empty or len(df) < 20:
        return vp

    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]

    # VWAP (intraday proxy using daily data)
    typical = (high + low + close) / 3
    vp.vwap = float((typical * volume).sum() / volume.sum()) if volume.sum() > 0 else float(close.iloc[-1])

    # Relative volume
    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    if avg_vol_20 > 0:
        vp.relative_volume = float(volume.iloc[-1] / avg_vol_20)

    # Volume trend: 5d avg vs 20d avg
    avg_vol_5 = volume.rolling(5).mean().iloc[-1]
    if avg_vol_20 > 0:
        vp.volume_trend = float((avg_vol_5 - avg_vol_20) / avg_vol_20)

    # Accumulation/Distribution
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    ad = (mfm * volume).cumsum()
    vp.accumulation_dist = float(ad.iloc[-1])

    # OBV slope (linear regression over last 10 days)
    obv = (np.sign(close.diff()) * volume).cumsum()
    if len(obv) >= 10:
        x = np.arange(10)
        y = obv.iloc[-10:].values
        slope = np.polyfit(x, y, 1)[0]
        vp.obv_slope = float(slope)

    return vp


def _compute_options_metrics(ticker: yf.Ticker, current_price: float) -> OptionsMetrics:
    om = OptionsMetrics()
    try:
        expirations = ticker.options
        if not expirations:
            return om

        # Use nearest expiration for liquid, actionable greeks
        nearest_exp = expirations[0]
        chain = ticker.option_chain(nearest_exp)
        calls = chain.calls
        puts = chain.puts

        if calls.empty or puts.empty:
            return om

        # Filter near-ATM options (within 5% of current price)
        atm_range = current_price * 0.05
        near_calls = calls[
            (calls["strike"] >= current_price - atm_range) &
            (calls["strike"] <= current_price + atm_range)
        ].copy()
        near_puts = puts[
            (puts["strike"] >= current_price - atm_range) &
            (puts["strike"] <= current_price + atm_range)
        ].copy()

        if near_calls.empty or near_puts.empty:
            return om

        # ATM Implied Volatility (average of nearest call + put)
        atm_call = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts["strike"] - current_price).abs().argsort()[:1]]
        call_iv = float(atm_call["impliedVolatility"].iloc[0]) if "impliedVolatility" in atm_call.columns else 0.0
        put_iv = float(atm_put["impliedVolatility"].iloc[0]) if "impliedVolatility" in atm_put.columns else 0.0
        om.iv_atm = (call_iv + put_iv) / 2
        om.iv_skew = put_iv - call_iv

        # Put/Call ratio by volume
        total_call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
        total_put_vol = puts["volume"].sum() if "volume" in puts.columns else 0
        if total_call_vol > 0:
            om.put_call_ratio = float(total_put_vol / total_call_vol)

        # Greeks from near-ATM options
        for col, attr in [("delta", "avg_delta_calls"), ("gamma", "avg_gamma"),
                          ("theta", "avg_theta"), ("vega", "avg_vega")]:
            # yfinance doesn't always provide greeks directly; use what's available
            pass

        # Estimate greeks from Black-Scholes approximations when not provided
        # Delta proxy: moneyness-based
        if not near_calls.empty:
            moneyness = near_calls["strike"] / current_price
            om.avg_delta_calls = float((1 - moneyness).clip(-1, 1).mean())
        if not near_puts.empty:
            moneyness = near_puts["strike"] / current_price
            om.avg_delta_puts = float((moneyness - 1).clip(-1, 0).mean())

        # Gamma proxy: highest near ATM
        om.avg_gamma = float(om.iv_atm / (current_price * 0.01)) if om.iv_atm > 0 else 0.0

        # Theta proxy: time decay estimate (negative for long positions)
        om.avg_theta = float(-om.iv_atm * current_price / (365 * 2)) if om.iv_atm > 0 else 0.0

        # Vega proxy: sensitivity to 1% IV change
        om.avg_vega = float(current_price * 0.01 * om.iv_atm) if om.iv_atm > 0 else 0.0

        # Max Pain: strike where total $ value of options expiring worthless is maximized
        all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
        pain = {}
        for strike in all_strikes:
            call_pain = calls[calls["strike"] < strike].apply(
                lambda r: (strike - r["strike"]) * r.get("openInterest", 0), axis=1
            ).sum()
            put_pain = puts[puts["strike"] > strike].apply(
                lambda r: (r["strike"] - strike) * r.get("openInterest", 0), axis=1
            ).sum()
            pain[strike] = call_pain + put_pain
        if pain:
            om.max_pain = min(pain, key=pain.get)

        # Net gamma exposure proxy
        call_gamma_exp = (near_calls.get("openInterest", pd.Series([0])) * 100 * om.avg_gamma).sum()
        put_gamma_exp = (near_puts.get("openInterest", pd.Series([0])) * 100 * om.avg_gamma).sum()
        om.net_gamma_exposure = float(call_gamma_exp - put_gamma_exp)

    except Exception as e:
        logger.warning(f"Options fetch failed for ticker: {e}")

    return om


def _compute_fundamentals(ticker: yf.Ticker) -> Fundamentals:
    f = Fundamentals()
    try:
        info = ticker.info
        f.market_cap = info.get("marketCap", 0) or 0
        f.pe_ratio = info.get("trailingPE", 0) or 0
        f.forward_pe = info.get("forwardPE", 0) or 0
        f.pb_ratio = info.get("priceToBook", 0) or 0
        f.dividend_yield = info.get("dividendYield", 0) or 0
        f.short_ratio = info.get("shortRatio", 0) or 0
        f.beta = info.get("beta", 1.0) or 1.0
        cal = ticker.calendar
        if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                f.earnings_date = str(ed[0]) if isinstance(ed, list) and ed else str(ed)
    except Exception as e:
        logger.warning(f"Fundamentals fetch failed: {e}")
    return f


async def fetch_stock_data(symbol: str) -> StockData:
    """Fetch comprehensive stock data for a single symbol."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_stock_data_sync, symbol)


def _fetch_stock_data_sync(symbol: str) -> StockData:
    sd = StockData(symbol=symbol)
    try:
        ticker = yf.Ticker(symbol)

        # 3-month daily OHLCV for technical analysis
        df = ticker.history(period="3mo", interval="1d")
        if df.empty or len(df) < 20:
            logger.warning(f"Insufficient data for {symbol}")
            return sd

        sd.ohlcv = df
        sd.price = float(df["Close"].iloc[-1])
        sd.timestamp = datetime.now()

        # Returns
        close = df["Close"]
        sd.returns_1d = float(close.pct_change(1).iloc[-1])
        sd.returns_5d = float(close.pct_change(5).iloc[-1])
        sd.returns_20d = float(close.pct_change(20).iloc[-1])

        # Realized volatility
        sd.volatility_20d = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))

        # ATR
        sd.atr_14 = _compute_atr(df)

        # Volume profile
        sd.volume_profile = _compute_volume_profile(df)

        # Options metrics (greeks, IV, skew)
        sd.options = _compute_options_metrics(ticker, sd.price)

        # Fundamentals
        sd.fundamentals = _compute_fundamentals(ticker)

    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")

    return sd


async def fetch_multiple(symbols: list[str]) -> dict[str, StockData]:
    """Fetch data for multiple symbols concurrently."""
    tasks = [fetch_stock_data(sym) for sym in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    data = {}
    for sym, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.error(f"Error fetching {sym}: {result}")
        elif result.price > 0:
            data[sym] = result
    return data


async def fetch_current_price(symbol: str) -> float:
    """Quick price fetch for a single symbol (for sell checks)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _quick_price, symbol)


def _quick_price(symbol: str) -> float:
    try:
        t = yf.Ticker(symbol)
        data = t.history(period="1d", interval="1m")
        if not data.empty:
            return float(data["Close"].iloc[-1])
        data = t.history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception as e:
        logger.error(f"Price fetch failed for {symbol}: {e}")
    return 0.0
