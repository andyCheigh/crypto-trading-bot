"""
Entry point for the stock trading signal bot.

Usage:
    1. Copy .env.example to .env and fill in your Telegram credentials
    2. pip install -r requirements.txt
    3. python main.py
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from engine import TradingEngine
from telegram_bot import TelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")
        sys.exit(1)

    notifier = TelegramNotifier(token=token, chat_id=chat_id)
    engine = TradingEngine(notifier=notifier)

    logger.info("Launching trading bot...")
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
