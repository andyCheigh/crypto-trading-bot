"""
Telegram bot for sending options trade signals and portfolio updates.
Shows full contract details: strike, expiry, call/put, premium, Greeks.
"""

from __future__ import annotations

import logging

from telegram import Bot
from telegram.constants import ParseMode

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id

    async def send(self, message: str):
        try:
            if len(message) > 4000:
                message = message[:4000] + "\n..."
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"```\n{message}\n```",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=message)
            except Exception as e2:
                logger.error(f"Telegram fallback send also failed: {e2}")

    async def send_buy_signal(
        self,
        display_name: str,
        premium: float,
        contracts: int,
        scores: dict[str, str],
        ensemble_direction: str,
        ensemble_conviction: float,
        greeks: dict,
        portfolio_status: str,
    ):
        total_cost = premium * contracts * 100
        algo_lines = "\n".join(
            f"  {name.replace('_', ' ').title()}: {score_str}"
            for name, score_str in scores.items()
        )

        delta = greeks.get("delta", 0)
        gamma = greeks.get("gamma", 0)
        theta = greeks.get("theta", 0)
        vega = greeks.get("vega", 0)
        iv = greeks.get("iv", 0)
        dte = greeks.get("dte", 0)

        msg = (
            f"BUY {ensemble_direction}: {display_name}\n"
            f"Premium: ${premium:.2f} | Contracts: {contracts} | Cost: ${total_cost:,.2f}\n"
            f"Greeks: D{delta:.2f} G{gamma:.4f} T{theta:.3f} V{vega:.3f} IV:{iv:.1%}\n"
            f"DTE: {dte}\n"
            f"\n"
            f"Algorithm Signals:\n{algo_lines}\n"
            f"Ensemble: {ensemble_conviction:.4f} ({ensemble_direction})\n"
            f"\n"
            f"{portfolio_status}"
        )
        await self.send(msg)

    async def send_sell_signal(
        self,
        display_name: str,
        premium: float,
        contracts: int,
        pnl: float,
        pnl_pct: float,
        reason: str,
        greeks: dict,
        portfolio_status: str,
    ):
        arrow = "+" if pnl >= 0 else ""
        delta = greeks.get("delta", 0)
        iv = greeks.get("iv", 0)
        theta = greeks.get("theta", 0)
        dte = greeks.get("dte", 0)

        msg = (
            f"SELL: {display_name}\n"
            f"Premium: ${premium:.2f} | Contracts: {contracts}\n"
            f"P&L: {arrow}${pnl:.2f} ({arrow}{pnl_pct:.2%})\n"
            f"Greeks at exit: D{delta:.2f} T{theta:.3f} IV:{iv:.1%} DTE:{dte}\n"
            f"Reason: {reason}\n"
            f"\n"
            f"{portfolio_status}"
        )
        await self.send(msg)

    async def send_startup(self, portfolio_status: str):
        msg = (
            f"OPTIONS TRADING BOT STARTED\n"
            f"Instruments: CALLS & PUTS\n"
            f"Sell check: every 15s (Greeks + premium stops)\n"
            f"Buy scan: every 60s (ensemble direction signals)\n"
            f"EOD close: 3:45 PM ET\n"
            f"Max holdings: 10\n"
            f"\n"
            f"{portfolio_status}"
        )
        await self.send(msg)

    async def send_heartbeat(self, portfolio_status: str):
        msg = f"HEARTBEAT\n\n{portfolio_status}"
        await self.send(msg)
