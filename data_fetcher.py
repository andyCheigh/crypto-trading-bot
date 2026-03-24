"""
Options-focused data fetcher for prop-desk style trading.
Pulls full options chains across multiple expirations to compute:
- Greeks surface (delta, gamma, theta, vega per strike/expiry)
- Dealer gamma exposure (GEX) with gamma flip level
- Vanna & charm flow predictions
- IV surface: term structure, skew, smile
- Unusual options activity detection
- Volume profile & order flow proxies
- Contract selection: optimal strike/expiry for trade execution
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Black-Scholes Greeks calculator
# ---------------------------------------------------------------------------

def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _bs_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _bs_d1(S, K, T, r, sigma) - sigma * math.sqrt(T) if T > 0 and sigma > 0 else 0.0


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    return float(norm.cdf(d1) if is_call else norm.cdf(d1) - 1)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if is_call:
        return float((term1 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365)
    return float((term1 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * math.sqrt(T) / 100)  # per 1% vol move


def bs_vanna(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """dDelta/dVol = dVega/dSpot. Measures how delta changes with vol."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(S, K, T, r, sigma)
    return float(-norm.pdf(d1) * d2 / sigma)


def bs_charm(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """dDelta/dTime. How delta decays as time passes (delta bleed)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = _bs_d2(S, K, T, r, sigma)
    charm_val = -norm.pdf(d1) * (2 * r * T - d2 * sigma * math.sqrt(T)) / (2 * T * sigma * math.sqrt(T))
    if not is_call:
        charm_val += r * math.exp(-r * T) * norm.cdf(-d2)
    return float(charm_val / 365)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OptionsGreeks:
    """Per-strike greeks computed via Black-Scholes."""
    strike: float = 0.0
    expiry: str = ""
    is_call: bool = True
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    vanna: float = 0.0
    charm: float = 0.0
    open_interest: int = 0
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = 0.0


@dataclass
class ContractRecommendation:
    """Recommended option contract for trade execution."""
    symbol: str
    strike: float
    expiry: str
    option_type: str         # "CALL" or "PUT"
    premium: float           # Mid-price (bid+ask)/2 or last price
    bid: float = 0.0
    ask: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0
    open_interest: int = 0
    volume: int = 0
    dte: int = 0
    score: float = 0.0       # Composite ranking score


@dataclass
class GEXProfile:
    """Dealer Gamma Exposure profile."""
    net_gex: float = 0.0              # Net gamma exposure (positive = dealer long gamma)
    gex_per_strike: dict = field(default_factory=dict)  # strike -> GEX$
    gamma_flip_level: float = 0.0     # Price where GEX flips sign
    call_wall: float = 0.0            # Strike with highest call OI * gamma
    put_wall: float = 0.0             # Strike with highest put OI * gamma
    net_vanna_exposure: float = 0.0   # Aggregate vanna flow
    net_charm_exposure: float = 0.0   # Aggregate charm (delta bleed)


@dataclass
class IVSurface:
    """Implied volatility surface metrics."""
    iv_atm: float = 0.0              # ATM implied vol
    iv_25d_call: float = 0.0         # 25-delta call IV
    iv_25d_put: float = 0.0          # 25-delta put IV
    skew_25d: float = 0.0            # 25d put IV - 25d call IV (risk reversal)
    iv_term_slope: float = 0.0       # Front month IV - back month IV (backwardation = +)
    iv_rv_spread: float = 0.0        # IV - realized vol (vol risk premium)
    iv_percentile_20d: float = 0.5   # Where current IV sits vs 20d range
    rv_5d: float = 0.0               # 5-day realized vol
    rv_10d: float = 0.0              # 10-day realized vol
    rv_20d: float = 0.0              # 20-day realized vol


@dataclass
class OptionsFlow:
    """Options order flow and unusual activity metrics."""
    put_call_vol_ratio: float = 1.0  # Put vol / Call vol
    put_call_oi_ratio: float = 1.0   # Put OI / Call OI
    unusual_calls: int = 0           # Strikes where vol > 2x OI (smart money)
    unusual_puts: int = 0
    total_call_premium: float = 0.0  # $ spent on calls today
    total_put_premium: float = 0.0   # $ spent on puts today
    net_premium_flow: float = 0.0    # Call premium - Put premium (bullish flow)
    max_pain: float = 0.0


@dataclass
class VolumeProfile:
    vwap: float = 0.0
    relative_volume: float = 1.0
    volume_trend: float = 0.0
    accumulation_dist: float = 0.0
    obv_slope: float = 0.0


@dataclass
class StockData:
    symbol: str
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    ohlcv: Optional[pd.DataFrame] = None
    # Options-driven metrics (the core of our signals)
    gex: GEXProfile = field(default_factory=GEXProfile)
    iv_surface: IVSurface = field(default_factory=IVSurface)
    options_flow: OptionsFlow = field(default_factory=OptionsFlow)
    # All computed greeks per strike (for algo use)
    greeks_chain: list = field(default_factory=list)
    # Price-based
    volume_profile: VolumeProfile = field(default_factory=VolumeProfile)
    returns_1d: float = 0.0
    returns_5d: float = 0.0
    returns_20d: float = 0.0
    atr_14: float = 0.0
    # Fundamentals
    market_cap: float = 0.0
    beta: float = 1.0
    short_ratio: float = 0.0
    earnings_date: Optional[str] = None
    days_to_earnings: int = 999


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["High"], df["Low"], df["Close"]
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
    close, volume, high, low = df["Close"], df["Volume"], df["High"], df["Low"]

    typical = (high + low + close) / 3
    vp.vwap = float((typical * volume).sum() / volume.sum()) if volume.sum() > 0 else float(close.iloc[-1])

    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    avg_vol_5 = volume.rolling(5).mean().iloc[-1]
    if avg_vol_20 > 0:
        vp.relative_volume = float(volume.iloc[-1] / avg_vol_20)
        vp.volume_trend = float((avg_vol_5 - avg_vol_20) / avg_vol_20)

    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    vp.accumulation_dist = float((mfm * volume).cumsum().iloc[-1])

    obv = (np.sign(close.diff()) * volume).cumsum()
    if len(obv) >= 10:
        x = np.arange(10)
        vp.obv_slope = float(np.polyfit(x, obv.iloc[-10:].values, 1)[0])
    return vp


def _compute_full_options(ticker: yf.Ticker, S: float, rv_20d: float) -> tuple:
    """Compute full options analytics across multiple expirations.
    Returns (GEXProfile, IVSurface, OptionsFlow, greeks_chain).
    """
    gex = GEXProfile()
    ivs = IVSurface()
    flow = OptionsFlow()
    greeks_chain = []

    r = 0.05  # risk-free rate proxy

    try:
        expirations = ticker.options
        if not expirations:
            return gex, ivs, flow, greeks_chain

        # Use up to 3 nearest expirations for rich surface data
        exps_to_use = expirations[:min(3, len(expirations))]

        all_call_vol = 0
        all_put_vol = 0
        all_call_oi = 0
        all_put_oi = 0
        all_call_premium = 0.0
        all_put_premium = 0.0
        unusual_calls = 0
        unusual_puts = 0

        gex_by_strike = {}
        total_vanna = 0.0
        total_charm = 0.0

        # IV by expiry for term structure
        atm_ivs_by_dte = {}

        for exp_str in exps_to_use:
            try:
                chain = ticker.option_chain(exp_str)
            except Exception:
                continue
            calls, puts = chain.calls, chain.puts
            if calls.empty or puts.empty:
                continue

            # Days to expiry
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = max((exp_date - datetime.now()).days, 1)
            T = dte / 365.0

            # Process calls
            for _, row in calls.iterrows():
                K = float(row["strike"])
                raw_iv = row.get("impliedVolatility", 0)
                iv = float(raw_iv) if pd.notna(raw_iv) else 0.0
                raw_oi = row.get("openInterest", 0)
                oi = int(raw_oi) if pd.notna(raw_oi) else 0
                raw_vol = row.get("volume", 0)
                vol = int(raw_vol) if pd.notna(raw_vol) else 0
                raw_last = row.get("lastPrice", 0)
                last = float(raw_last) if pd.notna(raw_last) else 0.0
                raw_bid = row.get("bid", 0)
                bid = float(raw_bid) if pd.notna(raw_bid) else 0.0
                raw_ask = row.get("ask", 0)
                ask = float(raw_ask) if pd.notna(raw_ask) else 0.0

                if iv <= 0 or iv > 5.0:
                    continue

                delta = bs_delta(S, K, T, r, iv, True)
                gamma = bs_gamma(S, K, T, r, iv)
                theta = bs_theta(S, K, T, r, iv, True)
                vega = bs_vega(S, K, T, r, iv)
                van = bs_vanna(S, K, T, r, iv)
                chrm = bs_charm(S, K, T, r, iv, True)

                greeks_chain.append(OptionsGreeks(
                    strike=K, expiry=exp_str, is_call=True,
                    iv=iv, delta=delta, gamma=gamma, theta=theta,
                    vega=vega, vanna=van, charm=chrm,
                    open_interest=oi, volume=vol,
                    bid=bid, ask=ask, last_price=last,
                ))

                # GEX: dealer is short calls → short gamma on calls
                gex_val = -oi * 100 * gamma * S
                gex_by_strike[K] = gex_by_strike.get(K, 0.0) + gex_val

                total_vanna += -oi * 100 * van
                total_charm += -oi * 100 * chrm

                all_call_vol += vol
                all_call_oi += oi
                all_call_premium += vol * last * 100

                if oi > 0 and vol > 2 * oi:
                    unusual_calls += 1

                # Track ATM IV for term structure
                if abs(K - S) / S < 0.02:
                    atm_ivs_by_dte[dte] = iv

            # Process puts
            for _, row in puts.iterrows():
                K = float(row["strike"])
                raw_iv = row.get("impliedVolatility", 0)
                iv = float(raw_iv) if pd.notna(raw_iv) else 0.0
                raw_oi = row.get("openInterest", 0)
                oi = int(raw_oi) if pd.notna(raw_oi) else 0
                raw_vol = row.get("volume", 0)
                vol = int(raw_vol) if pd.notna(raw_vol) else 0
                raw_last = row.get("lastPrice", 0)
                last = float(raw_last) if pd.notna(raw_last) else 0.0
                raw_bid = row.get("bid", 0)
                bid = float(raw_bid) if pd.notna(raw_bid) else 0.0
                raw_ask = row.get("ask", 0)
                ask = float(raw_ask) if pd.notna(raw_ask) else 0.0

                if iv <= 0 or iv > 5.0:
                    continue

                delta = bs_delta(S, K, T, r, iv, False)
                gamma = bs_gamma(S, K, T, r, iv)
                theta = bs_theta(S, K, T, r, iv, False)
                vega = bs_vega(S, K, T, r, iv)
                van = bs_vanna(S, K, T, r, iv)
                chrm = bs_charm(S, K, T, r, iv, False)

                greeks_chain.append(OptionsGreeks(
                    strike=K, expiry=exp_str, is_call=False,
                    iv=iv, delta=delta, gamma=gamma, theta=theta,
                    vega=vega, vanna=van, charm=chrm,
                    open_interest=oi, volume=vol,
                    bid=bid, ask=ask, last_price=last,
                ))

                # GEX: dealer is short puts → long gamma on puts
                gex_val = oi * 100 * gamma * S
                gex_by_strike[K] = gex_by_strike.get(K, 0.0) + gex_val

                total_vanna += oi * 100 * van
                total_charm += oi * 100 * chrm

                all_put_vol += vol
                all_put_oi += oi
                all_put_premium += vol * last * 100

                if oi > 0 and vol > 2 * oi:
                    unusual_puts += 1

            # Max pain from first expiry
            if exp_str == exps_to_use[0]:
                flow.max_pain = _compute_max_pain(calls, puts)

        # --- Assemble GEX profile ---
        gex.gex_per_strike = gex_by_strike
        gex.net_gex = sum(gex_by_strike.values())
        gex.net_vanna_exposure = total_vanna
        gex.net_charm_exposure = total_charm

        # Gamma flip: find strike where cumulative GEX crosses zero
        if gex_by_strike:
            sorted_strikes = sorted(gex_by_strike.keys())
            cum = 0.0
            for k in sorted_strikes:
                prev_cum = cum
                cum += gex_by_strike[k]
                if prev_cum <= 0 < cum or prev_cum >= 0 > cum:
                    gex.gamma_flip_level = k
                    break

            # Call wall: highest call GEX strike (most positive for calls)
            call_gex = {k: v for k, v in gex_by_strike.items() if v < 0}
            if call_gex:
                gex.call_wall = min(call_gex, key=call_gex.get)  # most negative = biggest call wall
            put_gex = {k: v for k, v in gex_by_strike.items() if v > 0}
            if put_gex:
                gex.put_wall = max(put_gex, key=put_gex.get)

        # --- Assemble IV surface ---
        # ATM IV from nearest expiry
        nearest_greeks = [g for g in greeks_chain if g.expiry == exps_to_use[0]]
        atm_calls = [g for g in nearest_greeks if g.is_call and abs(g.strike - S) / S < 0.02]
        atm_puts = [g for g in nearest_greeks if not g.is_call and abs(g.strike - S) / S < 0.02]
        if atm_calls:
            ivs.iv_atm = np.mean([g.iv for g in atm_calls])
        if atm_calls and atm_puts:
            ivs.iv_atm = (np.mean([g.iv for g in atm_calls]) + np.mean([g.iv for g in atm_puts])) / 2

        # 25-delta skew (risk reversal)
        d25_calls = [g for g in nearest_greeks if g.is_call and 0.20 <= g.delta <= 0.30]
        d25_puts = [g for g in nearest_greeks if not g.is_call and -0.30 <= g.delta <= -0.20]
        if d25_calls:
            ivs.iv_25d_call = np.mean([g.iv for g in d25_calls])
        if d25_puts:
            ivs.iv_25d_put = np.mean([g.iv for g in d25_puts])
        ivs.skew_25d = ivs.iv_25d_put - ivs.iv_25d_call

        # Term structure slope
        if len(atm_ivs_by_dte) >= 2:
            sorted_dte = sorted(atm_ivs_by_dte.keys())
            ivs.iv_term_slope = atm_ivs_by_dte[sorted_dte[0]] - atm_ivs_by_dte[sorted_dte[-1]]

        # IV vs RV spread (vol risk premium)
        ivs.rv_20d = rv_20d
        if rv_20d > 0 and ivs.iv_atm > 0:
            ivs.iv_rv_spread = ivs.iv_atm - rv_20d

        # --- Assemble options flow ---
        flow.put_call_vol_ratio = float(all_put_vol / max(all_call_vol, 1))
        flow.put_call_oi_ratio = float(all_put_oi / max(all_call_oi, 1))
        flow.unusual_calls = unusual_calls
        flow.unusual_puts = unusual_puts
        flow.total_call_premium = all_call_premium
        flow.total_put_premium = all_put_premium
        flow.net_premium_flow = all_call_premium - all_put_premium

    except Exception as e:
        logger.warning(f"Options analysis failed: {e}")

    return gex, ivs, flow, greeks_chain


def _safe_oi(val) -> float:
    if pd.notna(val):
        return float(val)
    return 0.0


def _compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
    if not all_strikes:
        return 0.0
    pain = {}
    for strike in all_strikes:
        call_pain = calls[calls["strike"] < strike].apply(
            lambda r: (strike - r["strike"]) * _safe_oi(r.get("openInterest", 0)), axis=1
        ).sum()
        put_pain = puts[puts["strike"] > strike].apply(
            lambda r: (r["strike"] - strike) * _safe_oi(r.get("openInterest", 0)), axis=1
        ).sum()
        pain[strike] = call_pain + put_pain
    return min(pain, key=pain.get) if pain else 0.0


def _compute_fundamentals_lite(ticker: yf.Ticker) -> tuple:
    """Return (market_cap, beta, short_ratio, earnings_date_str, days_to_earnings)."""
    mcap, beta, sr, ed_str, dte = 0.0, 1.0, 0.0, None, 999
    try:
        info = ticker.info
        mcap = info.get("marketCap", 0) or 0
        beta = info.get("beta", 1.0) or 1.0
        sr = info.get("shortRatio", 0) or 0
        cal = ticker.calendar
        if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                if isinstance(ed, list) and ed:
                    ed_str = str(ed[0])
                    try:
                        ed_dt = datetime.strptime(ed_str[:10], "%Y-%m-%d")
                        dte = (ed_dt - datetime.now()).days
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"Fundamentals fetch failed: {e}")
    return mcap, beta, sr, ed_str, dte


# ---------------------------------------------------------------------------
# Main fetch functions
# ---------------------------------------------------------------------------

async def fetch_stock_data(symbol: str) -> StockData:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_stock_data_sync, symbol)


def _fetch_stock_data_sync(symbol: str) -> StockData:
    sd = StockData(symbol=symbol)
    try:
        ticker = yf.Ticker(symbol)

        # 3-month daily OHLCV
        df = ticker.history(period="3mo", interval="1d")
        if df.empty or len(df) < 20:
            logger.warning(f"Insufficient data for {symbol}")
            return sd

        sd.ohlcv = df
        sd.price = float(df["Close"].iloc[-1])
        sd.timestamp = datetime.now()

        close = df["Close"]
        sd.returns_1d = float(close.pct_change(1).iloc[-1])
        sd.returns_5d = float(close.pct_change(5).iloc[-1])
        sd.returns_20d = float(close.pct_change(20).iloc[-1])
        sd.atr_14 = _compute_atr(df)

        # Realized vols at multiple windows
        daily_ret = close.pct_change()
        rv_5d = float(daily_ret.rolling(5).std().iloc[-1] * np.sqrt(252))
        rv_10d = float(daily_ret.rolling(10).std().iloc[-1] * np.sqrt(252))
        rv_20d = float(daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252))

        sd.volume_profile = _compute_volume_profile(df)

        # Full options analytics
        gex, ivs, flow, greeks = _compute_full_options(ticker, sd.price, rv_20d)
        sd.gex = gex
        sd.iv_surface = ivs
        sd.iv_surface.rv_5d = rv_5d
        sd.iv_surface.rv_10d = rv_10d
        sd.iv_surface.rv_20d = rv_20d
        sd.options_flow = flow
        sd.greeks_chain = greeks

        # Fundamentals (light)
        mcap, beta, sr, ed_str, dte = _compute_fundamentals_lite(ticker)
        sd.market_cap = mcap
        sd.beta = beta
        sd.short_ratio = sr
        sd.earnings_date = ed_str
        sd.days_to_earnings = dte

    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")

    return sd


async def fetch_multiple(symbols: list[str]) -> dict[str, StockData]:
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


async def fetch_vix() -> float:
    """Fetch the current CBOE VIX level directly."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_vix_sync)


def _fetch_vix_sync() -> float:
    """Synchronous VIX fetch via ^VIX ticker."""
    try:
        vix = yf.Ticker("^VIX")
        data = vix.history(period="1d", interval="1m")
        if not data.empty:
            return float(data["Close"].iloc[-1])
        data = vix.history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception as e:
        logger.error(f"VIX fetch failed: {e}")
    return 0.0


# ---------------------------------------------------------------------------
# Contract selection engine
# ---------------------------------------------------------------------------

def select_optimal_contract(
    symbol: str,
    greeks_chain: list[OptionsGreeks],
    direction: str,
    spot_price: float,
    preferred_delta: float = 0.0,
    preferred_dte: int = 0,
) -> Optional[ContractRecommendation]:
    """Select the optimal option contract for a given direction.

    Args:
        symbol: Underlying ticker
        greeks_chain: Full Greeks chain from data fetch
        direction: "CALL" or "PUT"
        spot_price: Current underlying price
        preferred_delta: Algorithm's preferred delta (0 = use config default)
        preferred_dte: Algorithm's preferred DTE (0 = use config default)

    Returns:
        ContractRecommendation or None if no suitable contract found
    """
    is_call = direction == "CALL"
    delta_min, delta_max = config.TARGET_DELTA_RANGE
    dte_min = preferred_dte - 10 if preferred_dte > 0 else config.PREFERRED_DTE_MIN
    dte_max = preferred_dte + 10 if preferred_dte > 0 else config.PREFERRED_DTE_MAX
    target_delta = preferred_delta if preferred_delta > 0 else (delta_min + delta_max) / 2

    candidates = []
    for g in greeks_chain:
        if g.is_call != is_call:
            continue

        # Compute DTE
        try:
            exp_date = datetime.strptime(g.expiry, "%Y-%m-%d")
            dte = max((exp_date - datetime.now()).days, 0)
        except ValueError:
            continue

        # Filter DTE
        if dte < dte_min or dte > dte_max:
            continue

        # Filter delta (use absolute delta for puts)
        abs_delta = abs(g.delta)
        if abs_delta < delta_min or abs_delta > delta_max:
            continue

        # Filter liquidity
        if g.open_interest < config.MIN_OPEN_INTEREST:
            continue

        # Filter bid-ask spread
        mid_price = (g.bid + g.ask) / 2 if (g.bid > 0 and g.ask > 0) else g.last_price
        if mid_price <= 0:
            continue
        if g.bid > 0 and g.ask > 0:
            spread_pct = (g.ask - g.bid) / mid_price
            if spread_pct > config.MAX_BID_ASK_SPREAD_PCT:
                continue

        # Composite ranking score (higher = better)
        # 1. Delta proximity to target (40%)
        delta_score = 1.0 - abs(abs_delta - target_delta) / 0.20
        delta_score = max(delta_score, 0.0)

        # 2. Liquidity: OI + volume (25%)
        liq_score = min((g.open_interest + g.volume) / 2000.0, 1.0)

        # 3. IV value: prefer lower IV (buying cheap vol) (20%)
        iv_score = max(1.0 - g.iv / 0.80, 0.0) if g.iv > 0 else 0.5

        # 4. DTE proximity to target (15%)
        target_dte = preferred_dte if preferred_dte > 0 else 30
        dte_score = 1.0 - abs(dte - target_dte) / 30.0
        dte_score = max(dte_score, 0.0)

        composite = 0.40 * delta_score + 0.25 * liq_score + 0.20 * iv_score + 0.15 * dte_score

        candidates.append(ContractRecommendation(
            symbol=symbol,
            strike=g.strike,
            expiry=g.expiry,
            option_type=direction,
            premium=mid_price,
            bid=g.bid,
            ask=g.ask,
            delta=g.delta,
            gamma=g.gamma,
            theta=g.theta,
            vega=g.vega,
            iv=g.iv,
            open_interest=g.open_interest,
            volume=g.volume,
            dte=dte,
            score=composite,
        ))

    if not candidates:
        # Relax filters: drop OI requirement, widen delta range
        for g in greeks_chain:
            if g.is_call != is_call:
                continue
            try:
                exp_date = datetime.strptime(g.expiry, "%Y-%m-%d")
                dte = max((exp_date - datetime.now()).days, 0)
            except ValueError:
                continue
            if dte < config.PREFERRED_DTE_MIN or dte > config.PREFERRED_DTE_MAX + 15:
                continue
            abs_delta = abs(g.delta)
            if abs_delta < 0.15 or abs_delta > 0.65:
                continue
            mid_price = (g.bid + g.ask) / 2 if (g.bid > 0 and g.ask > 0) else g.last_price
            if mid_price <= 0:
                continue
            candidates.append(ContractRecommendation(
                symbol=symbol, strike=g.strike, expiry=g.expiry,
                option_type=direction, premium=mid_price,
                bid=g.bid, ask=g.ask,
                delta=g.delta, gamma=g.gamma, theta=g.theta, vega=g.vega,
                iv=g.iv, open_interest=g.open_interest, volume=g.volume,
                dte=dte, score=0.3,  # Lower score for relaxed-filter contracts
            ))

    if not candidates:
        logger.warning(f"No suitable {direction} contracts found for {symbol}")
        return None

    # Return best scoring contract
    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]
    logger.info(
        f"Selected {best.symbol} {best.expiry} ${best.strike} {best.option_type} "
        f"@ ${best.premium:.2f} (Δ{best.delta:.2f} IV:{best.iv:.1%} OI:{best.open_interest} DTE:{best.dte})"
    )
    return best


async def fetch_option_price(symbol: str, strike: float, expiry: str, option_type: str) -> Optional[dict]:
    """Fetch current bid/ask/last/greeks for a specific option contract.

    Returns dict with keys: premium, bid, ask, delta, gamma, theta, vega, iv, dte
    or None if fetch fails.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _fetch_option_price_sync, symbol, strike, expiry, option_type
    )


def _fetch_option_price_sync(symbol: str, strike: float, expiry: str, option_type: str) -> Optional[dict]:
    """Synchronous option price fetch for a specific contract."""
    try:
        ticker = yf.Ticker(symbol)

        # Get underlying price for Greeks calculation
        hist = ticker.history(period="1d")
        if hist.empty:
            return None
        S = float(hist["Close"].iloc[-1])

        chain = ticker.option_chain(expiry)
        if option_type == "CALL":
            df = chain.calls
        else:
            df = chain.puts

        if df.empty:
            return None

        # Find the specific strike
        row = df[df["strike"] == strike]
        if row.empty:
            # Find closest strike
            df["dist"] = abs(df["strike"] - strike)
            row = df.loc[[df["dist"].idxmin()]]

        row = row.iloc[0]
        raw_bid = row.get("bid", 0)
        bid = float(raw_bid) if pd.notna(raw_bid) else 0.0
        raw_ask = row.get("ask", 0)
        ask = float(raw_ask) if pd.notna(raw_ask) else 0.0
        raw_last = row.get("lastPrice", 0)
        last = float(raw_last) if pd.notna(raw_last) else 0.0
        raw_iv = row.get("impliedVolatility", 0)
        iv = float(raw_iv) if pd.notna(raw_iv) else 0.0

        premium = (bid + ask) / 2 if (bid > 0 and ask > 0) else last

        # Compute current Greeks
        K = float(row["strike"])
        r = 0.05
        exp_date = datetime.strptime(expiry, "%Y-%m-%d")
        dte = max((exp_date - datetime.now()).days, 1)
        T = dte / 365.0
        is_call = option_type == "CALL"

        delta = bs_delta(S, K, T, r, iv, is_call) if iv > 0 else 0.0
        gamma = bs_gamma(S, K, T, r, iv) if iv > 0 else 0.0
        theta = bs_theta(S, K, T, r, iv, is_call) if iv > 0 else 0.0
        vega = bs_vega(S, K, T, r, iv) if iv > 0 else 0.0

        return {
            "premium": premium,
            "bid": bid,
            "ask": ask,
            "last": last,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "iv": iv,
            "dte": dte,
            "underlying_price": S,
        }

    except Exception as e:
        logger.error(f"Option price fetch failed for {symbol} {expiry} ${strike} {option_type}: {e}")
        return None


async def fetch_option_prices_batch(
    positions: list[tuple[str, float, str, str]],
) -> dict[str, dict]:
    """Fetch current option prices for multiple positions in parallel.

    Args:
        positions: list of (symbol, strike, expiry, option_type) tuples

    Returns:
        dict of option_key -> price_dict
    """
    tasks = [
        fetch_option_price(sym, strike, exp, otype)
        for sym, strike, exp, otype in positions
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    prices = {}
    for (sym, strike, exp, otype), result in zip(positions, results):
        key = f"{sym}_{strike}_{exp}_{otype}"
        if isinstance(result, dict) and result:
            prices[key] = result
    return prices
