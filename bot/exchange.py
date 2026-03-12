"""Exchange connectivity layer using CCXT."""

import ccxt
import pandas as pd
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)


class ExchangeClient:
    """Wrapper around CCXT for exchange operations."""

    def __init__(self, config: dict):
        load_dotenv()
        self.config = config
        exchange_name = config["exchange"]["name"]
        api_key = os.getenv("EXCHANGE_API_KEY", "")
        api_secret = os.getenv("EXCHANGE_API_SECRET", "")
        passphrase = os.getenv("EXCHANGE_PASSPHRASE", "")

        exchange_class = getattr(ccxt, exchange_name, None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{exchange_name}' not supported by CCXT")

        params = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        }
        if passphrase:
            params["password"] = passphrase
        if config["exchange"].get("sandbox"):
            params["sandbox"] = True

        self.exchange: ccxt.Exchange = exchange_class(params)
        logger.info("Connected to %s", exchange_name)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV candles and return as DataFrame."""
        raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        return self.exchange.fetch_ticker(symbol)

    def fetch_balance(self) -> dict:
        return self.exchange.fetch_balance()

    def create_market_buy(self, symbol: str, amount: float) -> dict:
        return self.exchange.create_market_buy_order(symbol, amount)

    def create_market_sell(self, symbol: str, amount: float) -> dict:
        return self.exchange.create_market_sell_order(symbol, amount)

    def fetch_markets(self) -> list:
        return self.exchange.load_markets()
