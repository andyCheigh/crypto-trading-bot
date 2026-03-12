"""Main trading bot loop."""

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from colorama import Fore, Style, init as colorama_init

from .exchange import ExchangeClient
from .portfolio import Portfolio
from .risk_manager import RiskManager
from .strategy import MultiSignalStrategy

colorama_init()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    # Suppress noisy ccxt logs
    logging.getLogger("ccxt").setLevel(logging.WARNING)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def print_banner(config: dict):
    mode = config["trading"]["mode"]
    color = Fore.YELLOW if mode == "dry_run" else Fore.RED
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  Crypto Trading Bot v1.0")
    print(f"  Exchange: {config['exchange']['name'].upper()}")
    print(f"  Mode: {color}{mode.upper()}{Fore.CYAN}")
    print(f"  Symbols: {', '.join(config['trading']['symbols'])}")
    print(f"  Timeframe: {config['trading']['timeframe']}")
    print(f"{'='*60}{Style.RESET_ALL}\n")


def run_cycle(
    exchange: ExchangeClient,
    strategy: MultiSignalStrategy,
    risk: RiskManager,
    portfolio: Portfolio,
    config: dict,
    is_dry_run: bool,
):
    """Execute one full trading cycle across all symbols."""
    symbols = config["trading"]["symbols"]
    timeframe = config["trading"]["timeframe"]
    limit = config["trading"]["candle_limit"]

    prices: dict[str, float] = {}

    for symbol in symbols:
        try:
            df = exchange.fetch_ohlcv(symbol, timeframe, limit)
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker["last"]
            prices[symbol] = current_price

            action, signals = strategy.evaluate(df)

            if is_dry_run:
                _handle_dry_run(action, symbol, current_price, portfolio, risk, prices)
            else:
                _handle_live(action, symbol, current_price, exchange, risk, portfolio, prices)

        except Exception as e:
            logging.getLogger(__name__).error("Error processing %s: %s", symbol, e)

    # Print portfolio summary
    summary = portfolio.summary(prices)
    print(f"\n{Fore.GREEN}{summary}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] Cycle complete.{Style.RESET_ALL}\n")


def _handle_dry_run(
    action: str,
    symbol: str,
    price: float,
    portfolio: Portfolio,
    risk: RiskManager,
    prices: dict[str, float],
):
    equity = portfolio.equity(prices)

    if risk.check_drawdown(equity):
        return

    has_position = symbol in portfolio.positions

    # Check stop-loss / take-profit on open positions
    if has_position:
        pos = portfolio.positions[symbol]
        if risk.check_stop_loss(pos.entry_price, price):
            print(f"{Fore.RED}[STOP-LOSS] {symbol} @ {price:.2f}{Style.RESET_ALL}")
            portfolio.close_position(symbol, price)
            return
        if risk.check_take_profit(pos.entry_price, price):
            print(f"{Fore.GREEN}[TAKE-PROFIT] {symbol} @ {price:.2f}{Style.RESET_ALL}")
            portfolio.close_position(symbol, price)
            return

    if action == "buy" and not has_position:
        if not risk.can_open_position(len(portfolio.positions)):
            return
        qty = risk.position_size(equity, price)
        if qty > 0:
            print(f"{Fore.GREEN}[DRY BUY] {symbol}: {qty:.6f} @ {price:.2f}{Style.RESET_ALL}")
            portfolio.open_position(symbol, price, qty)

    elif action == "sell" and has_position:
        print(f"{Fore.RED}[DRY SELL] {symbol} @ {price:.2f}{Style.RESET_ALL}")
        portfolio.close_position(symbol, price)


def _handle_live(
    action: str,
    symbol: str,
    price: float,
    exchange: ExchangeClient,
    risk: RiskManager,
    portfolio: Portfolio,
    prices: dict[str, float],
):
    equity = portfolio.equity(prices)

    if risk.check_drawdown(equity):
        return

    has_position = symbol in portfolio.positions

    # Check stop-loss / take-profit
    if has_position:
        pos = portfolio.positions[symbol]
        if risk.check_stop_loss(pos.entry_price, price):
            print(f"{Fore.RED}[STOP-LOSS] {symbol} @ {price:.2f}{Style.RESET_ALL}")
            order = exchange.create_market_sell(symbol, pos.quantity)
            portfolio.close_position(symbol, order.get("average", price))
            return
        if risk.check_take_profit(pos.entry_price, price):
            print(f"{Fore.GREEN}[TAKE-PROFIT] {symbol} @ {price:.2f}{Style.RESET_ALL}")
            order = exchange.create_market_sell(symbol, pos.quantity)
            portfolio.close_position(symbol, order.get("average", price))
            return

    if action == "buy" and not has_position:
        if not risk.can_open_position(len(portfolio.positions)):
            return
        qty = risk.position_size(equity, price)
        if qty > 0:
            print(f"{Fore.GREEN}[LIVE BUY] {symbol}: {qty:.6f} @ {price:.2f}{Style.RESET_ALL}")
            order = exchange.create_market_buy(symbol, qty)
            fill_price = order.get("average", price)
            portfolio.open_position(symbol, fill_price, qty)

    elif action == "sell" and has_position:
        pos = portfolio.positions[symbol]
        print(f"{Fore.RED}[LIVE SELL] {symbol} @ {price:.2f}{Style.RESET_ALL}")
        order = exchange.create_market_sell(symbol, pos.quantity)
        fill_price = order.get("average", price)
        portfolio.close_position(symbol, fill_price)


def main():
    setup_logging()
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    print_banner(config)

    is_dry_run = config["trading"]["mode"] == "dry_run"

    exchange = ExchangeClient(config)
    strategy = MultiSignalStrategy(config)
    risk = RiskManager(config)

    if is_dry_run:
        balances = config["dry_run"]["starting_balance"]
        portfolio = Portfolio(balances)
        print(f"{Fore.YELLOW}[DRY RUN] Starting with: {balances}{Style.RESET_ALL}")
    else:
        # Fetch real balances from exchange
        raw = exchange.fetch_balance()
        balances = {k: v for k, v in raw.get("free", {}).items() if v and v > 0}
        portfolio = Portfolio(balances)
        print(f"{Fore.RED}[LIVE MODE] Trading with real funds!{Style.RESET_ALL}")

    poll = config["trading"]["poll_interval_seconds"]

    print(f"Starting trading loop (poll every {poll}s). Press Ctrl+C to stop.\n")

    try:
        while True:
            run_cycle(exchange, strategy, risk, portfolio, config, is_dry_run)
            time.sleep(poll)
    except KeyboardInterrupt:
        print(f"\n{Fore.CYAN}Bot stopped. Final state:{Style.RESET_ALL}")
        # Get last known prices
        prices = {}
        for sym in config["trading"]["symbols"]:
            try:
                prices[sym] = exchange.fetch_ticker(sym)["last"]
            except Exception:
                pass
        print(portfolio.summary(prices))

        if portfolio.trade_history:
            print(f"\n{Fore.CYAN}Trade History ({len(portfolio.trade_history)} trades):{Style.RESET_ALL}")
            for t in portfolio.trade_history:
                color = Fore.GREEN if t.side == "buy" else Fore.RED
                print(f"  {color}{t.timestamp.strftime('%H:%M:%S')} {t.side.upper()} {t.symbol}: "
                      f"{t.quantity:.6f} @ {t.price:.2f} (PnL: {t.pnl:+,.2f}){Style.RESET_ALL}")


if __name__ == "__main__":
    main()
