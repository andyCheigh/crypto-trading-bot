"""
Telegram bot for sending trade signals and portfolio updates.
"""

from __future__ import annotations

import logging
from typing import Optional

from telegram import Bot
from telegram.constants import ParseMode

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id

    async def send(self, message: str):
        """Send a message to the configured chat."""
        try:
            # Telegram max message length is 4096
            if len(message) > 4000:
                message = message[:4000] + "\n..."
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"```\n{message}\n```",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            # Fallback without markdown
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                )
            except Exception as e2:
                logger.error(f"Telegram fallback send also failed: {e2}")

    async def send_buy_signal(
        self,
        symbol: str,
        price: float,
        shares: int,
        scores: dict[str, float],
        ensemble_score: float,
        portfolio_status: str,
    ):
        algo_lines = "\n".join(f"  {name}: {score:.4f}" for name, score in scores.items())
        msg = (
            f"BUY SIGNAL: {symbol}\n"
            f"Price: ${price:.2f}\n"
            f"Shares: {shares}\n"
            f"Cost: ${price * shares:,.2f}\n"
            f"\n"
            f"Algorithm Scores:\n{algo_lines}\n"
            f"Ensemble: {ensemble_score:.4f}\n"
            f"\n"
            f"{portfolio_status}"
        )
        await self.send(msg)

    async def send_sell_signal(
        self,
        symbol: str,
        price: float,
        shares: int,
        pnl: float,
        pnl_pct: float,
        reason: str,
        portfolio_status: str,
    ):
        arrow = "+" if pnl >= 0 else ""
        msg = (
            f"SELL SIGNAL: {symbol}\n"
            f"Price: ${price:.2f}\n"
            f"Shares: {shares}\n"
            f"P&L: {arrow}${pnl:.2f} ({arrow}{pnl_pct:.2%})\n"
            f"Reason: {reason}\n"
            f"\n"
            f"{portfolio_status}"
        )
        await self.send(msg)

    async def send_startup(self, portfolio_status: str):
        msg = (
            f"TRADING BOT STARTED\n"
            f"Sell check: every 15s\n"
            f"Buy scan: every 60s\n"
            f"Max holdings: 5\n"
            f"\n"
            f"{portfolio_status}"
        )
        await self.send(msg)

    async def send_heartbeat(self, portfolio_status: str):
        msg = f"HEARTBEAT\n\n{portfolio_status}"
        await self.send(msg)
