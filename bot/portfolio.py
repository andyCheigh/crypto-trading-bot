"""Portfolio tracker for both dry-run and live modes."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    side: str  # "long"
    entry_price: float
    quantity: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Trade:
    symbol: str
    side: str  # "buy" or "sell"
    price: float
    quantity: float
    pnl: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Portfolio:
    """Manages balances and positions for dry-run mode."""

    def __init__(self, starting_balance: dict[str, float]):
        self.balances: dict[str, float] = dict(starting_balance)
        self.positions: dict[str, Position] = {}
        self.trade_history: list[Trade] = []
        self.initial_equity_snapshot: float | None = None

    def get_quote_balance(self, quote: str = "USDT") -> float:
        return self.balances.get(quote, 0.0)

    def equity(self, prices: dict[str, float], quote: str = "USDT") -> float:
        """Total portfolio value in quote currency."""
        total = self.balances.get(quote, 0.0)
        for asset, amount in self.balances.items():
            if asset == quote:
                continue
            pair = f"{asset}/{quote}"
            if pair in prices:
                total += amount * prices[pair]
        return total

    def open_position(self, symbol: str, price: float, quantity: float, quote: str = "USDT"):
        cost = price * quantity
        if self.get_quote_balance(quote) < cost:
            logger.warning("Insufficient %s balance for %s", quote, symbol)
            return False

        base = symbol.split("/")[0]
        self.balances[quote] -= cost
        self.balances[base] = self.balances.get(base, 0.0) + quantity
        self.positions[symbol] = Position(symbol=symbol, side="long", entry_price=price, quantity=quantity)
        self.trade_history.append(Trade(symbol=symbol, side="buy", price=price, quantity=quantity, pnl=0.0))
        logger.info("OPEN %s: %.6f @ %.2f (cost: %.2f %s)", symbol, quantity, price, cost, quote)
        return True

    def close_position(self, symbol: str, price: float, quote: str = "USDT"):
        pos = self.positions.get(symbol)
        if not pos:
            return False

        base = symbol.split("/")[0]
        revenue = price * pos.quantity
        pnl = (price - pos.entry_price) * pos.quantity
        self.balances[quote] = self.balances.get(quote, 0.0) + revenue
        self.balances[base] = self.balances.get(base, 0.0) - pos.quantity
        if self.balances[base] <= 1e-10:
            del self.balances[base]

        self.trade_history.append(Trade(symbol=symbol, side="sell", price=price, quantity=pos.quantity, pnl=pnl))
        del self.positions[symbol]
        logger.info("CLOSE %s: %.6f @ %.2f (PnL: %.2f %s)", symbol, pos.quantity, price, pnl, quote)
        return True

    def summary(self, prices: dict[str, float]) -> str:
        eq = self.equity(prices)
        if self.initial_equity_snapshot is None:
            self.initial_equity_snapshot = eq
        total_pnl = eq - self.initial_equity_snapshot
        pnl_pct = (total_pnl / self.initial_equity_snapshot * 100) if self.initial_equity_snapshot else 0
        lines = [
            f"Equity: {eq:,.2f} USDT | PnL: {total_pnl:+,.2f} ({pnl_pct:+.2f}%)",
            f"Open positions: {len(self.positions)}",
        ]
        for sym, pos in self.positions.items():
            curr = prices.get(sym, pos.entry_price)
            pos_pnl = (curr - pos.entry_price) * pos.quantity
            lines.append(f"  {sym}: {pos.quantity:.6f} @ {pos.entry_price:.2f} -> {curr:.2f} (PnL: {pos_pnl:+,.2f})")
        return "\n".join(lines)
