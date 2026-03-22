"""
Portfolio tracker with $10K initial capital.
Tracks positions, realized/unrealized P&L, per-holding profit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    entry_price: float
    shares: int
    entry_time: datetime = field(default_factory=datetime.now)
    peak_price: float = 0.0  # For trailing stop

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.shares

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price

    def update_peak(self, current_price: float):
        if current_price > self.peak_price:
            self.peak_price = current_price


@dataclass
class Trade:
    symbol: str
    side: str  # "BUY" or "SELL"
    price: float
    shares: int
    timestamp: datetime = field(default_factory=datetime.now)
    pnl: float = 0.0       # realized P&L for sells
    pnl_pct: float = 0.0   # realized P&L % for sells
    reason: str = ""


class Portfolio:
    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_history: list[Trade] = []
        self.realized_pnl = 0.0

    @property
    def num_holdings(self) -> int:
        return len(self.positions)

    def total_equity(self, prices: dict[str, float]) -> float:
        """Total portfolio value = cash + sum of position market values."""
        holdings_value = sum(
            pos.shares * prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + holdings_value

    def total_pnl(self, prices: dict[str, float]) -> float:
        return self.total_equity(prices) - self.initial_capital

    def total_pnl_pct(self, prices: dict[str, float]) -> float:
        return self.total_pnl(prices) / self.initial_capital

    def buy(self, symbol: str, price: float, cash_to_spend: float, reason: str = "") -> Optional[Trade]:
        """Buy shares of a stock. Returns Trade or None if insufficient funds."""
        if symbol in self.positions:
            logger.warning(f"Already holding {symbol}, skipping")
            return None

        shares = int(cash_to_spend / price)
        if shares <= 0:
            logger.warning(f"Cannot afford any shares of {symbol} at ${price:.2f}")
            return None

        cost = shares * price
        if cost > self.cash:
            shares = int(self.cash / price)
            if shares <= 0:
                return None
            cost = shares * price

        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
            entry_price=price,
            shares=shares,
        )

        trade = Trade(
            symbol=symbol,
            side="BUY",
            price=price,
            shares=shares,
            reason=reason,
        )
        self.trade_history.append(trade)
        logger.info(f"BUY {shares} x {symbol} @ ${price:.2f} = ${cost:.2f}")
        return trade

    def sell(self, symbol: str, price: float, reason: str = "") -> Optional[Trade]:
        """Sell entire position. Returns Trade or None if not holding."""
        if symbol not in self.positions:
            logger.warning(f"Not holding {symbol}, cannot sell")
            return None

        pos = self.positions[symbol]
        revenue = pos.shares * price
        pnl = (price - pos.entry_price) * pos.shares
        pnl_pct = (price - pos.entry_price) / pos.entry_price

        self.cash += revenue
        self.realized_pnl += pnl
        del self.positions[symbol]

        trade = Trade(
            symbol=symbol,
            side="SELL",
            price=price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )
        self.trade_history.append(trade)
        logger.info(f"SELL {pos.shares} x {symbol} @ ${price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2%})")
        return trade

    def format_status(self, prices: dict[str, float]) -> str:
        """Format full portfolio status for Telegram."""
        equity = self.total_equity(prices)
        total_pnl = self.total_pnl(prices)
        total_pnl_pct = self.total_pnl_pct(prices)

        arrow = "+" if total_pnl >= 0 else ""
        lines = [
            "━━━ PORTFOLIO STATUS ━━━",
            f"Equity: ${equity:,.2f}  ({arrow}{total_pnl_pct:.2%})",
            f"P&L: {arrow}${total_pnl:,.2f}",
            f"Cash: ${self.cash:,.2f}",
            f"Holdings: {self.num_holdings}/{config.MAX_HOLDINGS}",
            "",
        ]

        if self.positions:
            lines.append("── Holdings ──")
            for sym, pos in self.positions.items():
                current = prices.get(sym, pos.entry_price)
                upnl = pos.unrealized_pnl(current)
                upnl_pct = pos.unrealized_pnl_pct(current)
                arrow_h = "+" if upnl >= 0 else ""
                lines.append(
                    f"  {sym}: {pos.shares} shares @ ${pos.entry_price:.2f}"
                    f"  now ${current:.2f}"
                    f"  {arrow_h}${upnl:.2f} ({arrow_h}{upnl_pct:.2%})"
                )
            lines.append("")

        lines.append(f"Realized P&L: ${self.realized_pnl:+,.2f}")
        lines.append(f"Total trades: {len(self.trade_history)}")

        return "\n".join(lines)
