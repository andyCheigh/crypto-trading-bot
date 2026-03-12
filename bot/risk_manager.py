"""Risk management: position sizing, stop-loss, take-profit, drawdown."""

import logging

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: dict):
        self.cfg = config["risk"]
        self.peak_equity = 0.0

    def check_drawdown(self, current_equity: float) -> bool:
        """Returns True if we should halt trading (circuit breaker)."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.peak_equity == 0:
            return False
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown >= self.cfg["max_drawdown_pct"]:
            logger.warning(
                "CIRCUIT BREAKER: drawdown %.2f%% exceeds max %.2f%%",
                drawdown * 100,
                self.cfg["max_drawdown_pct"] * 100,
            )
            return True
        return False

    def position_size(self, equity: float, price: float) -> float:
        """Calculate position size as quantity of the asset."""
        allocation = equity * self.cfg["max_position_pct"]
        quantity = allocation / price
        return quantity

    def check_stop_loss(self, entry_price: float, current_price: float) -> bool:
        drop = (entry_price - current_price) / entry_price
        return drop >= self.cfg["stop_loss_pct"]

    def check_take_profit(self, entry_price: float, current_price: float) -> bool:
        gain = (current_price - entry_price) / entry_price
        return gain >= self.cfg["take_profit_pct"]

    def can_open_position(self, open_count: int) -> bool:
        return open_count < self.cfg["max_open_positions"]
