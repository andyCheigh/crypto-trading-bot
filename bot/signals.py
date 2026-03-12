"""Technical indicator signal generators."""

import pandas as pd
import ta


def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add all technical indicators to the OHLCV dataframe."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA
    df["ema_fast"] = ta.trend.ema_indicator(close, window=cfg["ema_fast"])
    df["ema_slow"] = ta.trend.ema_indicator(close, window=cfg["ema_slow"])

    # RSI
    df["rsi"] = ta.momentum.rsi(close, window=cfg["rsi_period"])

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=cfg["bb_period"], window_dev=cfg["bb_std_dev"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    # MACD
    macd = ta.trend.MACD(close, window_fast=cfg["macd_fast"], window_slow=cfg["macd_slow"], window_sign=cfg["macd_signal"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    return df


def generate_signals(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Evaluate the latest candle and return individual signal scores.
    Returns dict with signal name -> (+1 = buy, -1 = sell, 0 = neutral).
    """
    if len(df) < 2:
        return {"ema": 0, "rsi": 0, "bb": 0, "macd": 0}

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    signals = {}

    # 1. EMA Crossover
    if prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]:
        signals["ema"] = 1  # Bullish crossover
    elif prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]:
        signals["ema"] = -1  # Bearish crossover
    else:
        signals["ema"] = 0

    # 2. RSI
    if curr["rsi"] < cfg["rsi_oversold"]:
        signals["rsi"] = 1  # Oversold -> buy
    elif curr["rsi"] > cfg["rsi_overbought"]:
        signals["rsi"] = -1  # Overbought -> sell
    else:
        signals["rsi"] = 0

    # 3. Bollinger Bands
    if curr["close"] <= curr["bb_lower"]:
        signals["bb"] = 1  # Price at lower band -> buy
    elif curr["close"] >= curr["bb_upper"]:
        signals["bb"] = -1  # Price at upper band -> sell
    else:
        signals["bb"] = 0

    # 4. MACD Histogram crossover
    if prev["macd_hist"] <= 0 and curr["macd_hist"] > 0:
        signals["macd"] = 1  # Bullish
    elif prev["macd_hist"] >= 0 and curr["macd_hist"] < 0:
        signals["macd"] = -1  # Bearish
    else:
        signals["macd"] = 0

    return signals
