"""
Options portfolio tracker with $10K initial capital.
Tracks option positions (strike, expiry, call/put, contracts, premium, Greeks)
and realized/unrealized P&L.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class OptionPosition:
    """A single option contract position."""
    symbol: str              # Underlying ticker (e.g., "AAPL")
    strike: float            # Strike price
    expiry: str              # Expiration date "YYYY-MM-DD"
    option_type: str         # "CALL" or "PUT"
    contracts: int           # Number of contracts
    entry_premium: float     # Premium per share at entry
    entry_time: datetime = field(default_factory=datetime.now)
    entry_iv: float = 0.0    # IV at entry (for crush detection)
    entry_delta: float = 0.0 # Delta at entry
    peak_premium: float = 0.0  # For trailing stop on premium

    def __post_init__(self):
        if self.peak_premium == 0.0:
            self.peak_premium = self.entry_premium

    @property
    def cost_basis(self) -> float:
        """Total cost = premium × contracts × 100."""
        return self.entry_premium * self.contracts * config.CONTRACT_MULTIPLIER

    def unrealized_pnl(self, current_premium: float) -> float:
        return (current_premium - self.entry_premium) * self.contracts * config.CONTRACT_MULTIPLIER

    def unrealized_pnl_pct(self, current_premium: float) -> float:
        if self.entry_premium <= 0:
            return 0.0
        return (current_premium - self.entry_premium) / self.entry_premium

    def update_peak(self, current_premium: float):
        if current_premium > self.peak_premium:
            self.peak_premium = current_premium

    @property
    def option_key(self) -> str:
        """Unique key for this option position."""
        return f"{self.symbol}_{self.strike}_{self.expiry}_{self.option_type}"

    @property
    def display_name(self) -> str:
        """Human-readable contract name: AAPL APR 17 $185 CALL."""
        try:
            dt = datetime.strptime(self.expiry, "%Y-%m-%d")
            date_str = dt.strftime("%b %d").upper()
        except ValueError:
            date_str = self.expiry
        return f"{self.symbol} {date_str} ${self.strike:.0f} {self.option_type}"


@dataclass
class Trade:
    """Record of an executed option trade."""
    symbol: str
    strike: float
    expiry: str
    option_type: str         # "CALL" or "PUT"
    side: str                # "BUY" or "SELL"
    premium: float           # Premium per share
    contracts: int
    timestamp: datetime = field(default_factory=datetime.now)
    pnl: float = 0.0        # Realized P&L for sells
    pnl_pct: float = 0.0    # Realized P&L % for sells
    reason: str = ""
    entry_delta: float = 0.0
    entry_iv: float = 0.0

    @property
    def option_key(self) -> str:
        return f"{self.symbol}_{self.strike}_{self.expiry}_{self.option_type}"

    @property
    def display_name(self) -> str:
        try:
            dt = datetime.strptime(self.expiry, "%Y-%m-%d")
            date_str = dt.strftime("%b %d").upper()
        except ValueError:
            date_str = self.expiry
        return f"{self.symbol} {date_str} ${self.strike:.0f} {self.option_type}"

    @property
    def total_cost(self) -> float:
        return self.premium * self.contracts * config.CONTRACT_MULTIPLIER


class Portfolio:
    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, OptionPosition] = {}  # keyed by option_key
        self.trade_history: list[Trade] = []
        self.realized_pnl = 0.0

    @property
    def num_holdings(self) -> int:
        return len(self.positions)

    def total_equity(self, premiums: dict[str, float]) -> float:
        """Total portfolio value = cash + sum of option position market values.
        premiums: dict of option_key -> current premium per share
        """
        holdings_value = sum(
            pos.contracts * config.CONTRACT_MULTIPLIER * premiums.get(pos.option_key, pos.entry_premium)
            for pos in self.positions.values()
        )
        return self.cash + holdings_value

    def total_pnl(self, premiums: dict[str, float]) -> float:
        return self.total_equity(premiums) - self.initial_capital

    def total_pnl_pct(self, premiums: dict[str, float]) -> float:
        return self.total_pnl(premiums) / self.initial_capital

    def buy(
        self,
        symbol: str,
        strike: float,
        expiry: str,
        option_type: str,
        premium: float,
        contracts: int,
        entry_iv: float = 0.0,
        entry_delta: float = 0.0,
        reason: str = "",
    ) -> Optional[Trade]:
        """Buy option contracts. Returns Trade or None if insufficient funds."""
        key = f"{symbol}_{strike}_{expiry}_{option_type}"
        if key in self.positions:
            logger.warning(f"Already holding {key}, skipping")
            return None

        total_cost = premium * contracts * config.CONTRACT_MULTIPLIER
        if total_cost > self.cash:
            # Reduce contracts to fit budget
            contracts = int(self.cash / (premium * config.CONTRACT_MULTIPLIER))
            if contracts <= 0:
                logger.warning(f"Cannot afford any contracts of {symbol} {strike} {option_type}")
                return None
            total_cost = premium * contracts * config.CONTRACT_MULTIPLIER

        self.cash -= total_cost
        self.positions[key] = OptionPosition(
            symbol=symbol,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            contracts=contracts,
            entry_premium=premium,
            entry_iv=entry_iv,
            entry_delta=entry_delta,
        )

        trade = Trade(
            symbol=symbol,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            side="BUY",
            premium=premium,
            contracts=contracts,
            entry_delta=entry_delta,
            entry_iv=entry_iv,
            reason=reason,
        )
        self.trade_history.append(trade)
        logger.info(
            f"BUY {contracts}x {symbol} {expiry} ${strike} {option_type} "
            f"@ ${premium:.2f} = ${total_cost:.2f} "
            f"(Δ{entry_delta:.2f} IV:{entry_iv:.1%})"
        )
        return trade

    def sell(self, option_key: str, current_premium: float, reason: str = "") -> Optional[Trade]:
        """Sell entire option position. Returns Trade or None if not holding."""
        if option_key not in self.positions:
            logger.warning(f"Not holding {option_key}, cannot sell")
            return None

        pos = self.positions[option_key]
        revenue = current_premium * pos.contracts * config.CONTRACT_MULTIPLIER
        pnl = (current_premium - pos.entry_premium) * pos.contracts * config.CONTRACT_MULTIPLIER
        pnl_pct = (current_premium - pos.entry_premium) / pos.entry_premium if pos.entry_premium > 0 else 0.0

        self.cash += revenue
        self.realized_pnl += pnl
        del self.positions[option_key]

        trade = Trade(
            symbol=pos.symbol,
            strike=pos.strike,
            expiry=pos.expiry,
            option_type=pos.option_type,
            side="SELL",
            premium=current_premium,
            contracts=pos.contracts,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )
        self.trade_history.append(trade)
        logger.info(
            f"SELL {pos.contracts}x {pos.display_name} "
            f"@ ${current_premium:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2%}) | {reason}"
        )
        return trade

    def format_status(self, premiums: dict[str, float]) -> str:
        """Format full portfolio status for Telegram."""
        equity = self.total_equity(premiums)
        total_pnl = self.total_pnl(premiums)
        total_pnl_pct = self.total_pnl_pct(premiums)

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
            lines.append("── Option Positions ──")
            for key, pos in self.positions.items():
                current = premiums.get(key, pos.entry_premium)
                upnl = pos.unrealized_pnl(current)
                upnl_pct = pos.unrealized_pnl_pct(current)
                arrow_h = "+" if upnl >= 0 else ""
                lines.append(
                    f"  {pos.display_name}"
                    f"  {pos.contracts}x @ ${pos.entry_premium:.2f}"
                    f"  now ${current:.2f}"
                    f"  {arrow_h}${upnl:.2f} ({arrow_h}{upnl_pct:.2%})"
                )
            lines.append("")

        lines.append(f"Realized P&L: ${self.realized_pnl:+,.2f}")
        lines.append(f"Total trades: {len(self.trade_history)}")

        return "\n".join(lines)
