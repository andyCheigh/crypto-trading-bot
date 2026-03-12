"""Core trading strategy: multi-signal consensus with risk management."""

import logging

from .signals import compute_indicators, generate_signals

logger = logging.getLogger(__name__)


class MultiSignalStrategy:
    """
    Combines 4 signals (EMA crossover, RSI, Bollinger Bands, MACD) and
    requires a configurable minimum consensus before taking action.
    """

    def __init__(self, config: dict):
        self.cfg = config["strategy"]
        self.min_buy = self.cfg["min_signals_buy"]
        self.min_sell = self.cfg["min_signals_sell"]

    def evaluate(self, df) -> tuple[str, dict]:
        """
        Returns (action, signals_detail).
        action: "buy", "sell", or "hold"
        """
        df = compute_indicators(df, self.cfg)
        signals = generate_signals(df, self.cfg)

        buy_count = sum(1 for v in signals.values() if v == 1)
        sell_count = sum(1 for v in signals.values() if v == -1)

        logger.info(
            "Signals: %s | Buy: %d/%d, Sell: %d/%d",
            signals,
            buy_count,
            self.min_buy,
            sell_count,
            self.min_sell,
        )

        if buy_count >= self.min_buy:
            return "buy", signals
        elif sell_count >= self.min_sell:
            return "sell", signals
        else:
            return "hold", signals
