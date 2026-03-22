"""
Main trading engine with parallel buy/sell loops.

- Sell check: every 15 seconds on all held positions
- Buy scan: every 60 seconds if < 5 holdings
- Both loops run concurrently via asyncio
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone, timedelta

import config
from algorithms import StatArbAlgorithm, MomentumAlgorithm, VolSurfaceAlgorithm
from data_fetcher import fetch_current_price, fetch_multiple, StockData
from portfolio import Portfolio
from telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)

# US Eastern timezone (UTC-5 / UTC-4 during DST)
ET = timezone(timedelta(hours=-4))  # EDT; adjust to -5 for EST

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
WEEKDAYS = range(0, 5)  # Mon=0 through Fri=4


def is_market_open() -> bool:
    """Check if US stock market is currently open (Mon-Fri 9:30-16:00 ET)."""
    now_et = datetime.now(ET)
    if now_et.weekday() not in WEEKDAYS:
        return False
    return MARKET_OPEN <= now_et.time() < MARKET_CLOSE


def seconds_until_market_open() -> float:
    """Seconds until next market open. Returns 0 if market is open."""
    if is_market_open():
        return 0.0
    now_et = datetime.now(ET)
    # Next market open
    target = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et.time() >= MARKET_CLOSE or now_et.weekday() >= 5:
        # Move to next weekday
        days_ahead = 1
        candidate = now_et + timedelta(days=days_ahead)
        while candidate.weekday() not in WEEKDAYS:
            days_ahead += 1
            candidate = now_et + timedelta(days=days_ahead)
        target = candidate.replace(hour=9, minute=30, second=0, microsecond=0)
    elif now_et.time() < MARKET_OPEN:
        pass  # target is today at 9:30
    return max((target - now_et).total_seconds(), 0.0)


class TradingEngine:
    def __init__(self, notifier: TelegramNotifier):
        self.portfolio = Portfolio(initial_capital=config.INITIAL_CAPITAL)
        self.notifier = notifier

        # Initialize algorithms
        self.algorithms = [
            StatArbAlgorithm(),
            MomentumAlgorithm(),
            VolSurfaceAlgorithm(),
        ]
        self.algo_weights = config.ALGO_WEIGHTS

        # Cache of latest stock data for held positions
        self._held_data_cache: dict[str, StockData] = {}
        # Last known prices for off-hours display
        self._last_known_prices: dict[str, float] = {}

        # Lock to prevent buy/sell race conditions on portfolio
        self._portfolio_lock = asyncio.Lock()

        # Track market state for open/close notifications
        self._was_market_open = False

    def _ensemble_buy_score(self, data: StockData) -> tuple[float, dict[str, float]]:
        """Compute weighted ensemble buy score across all algorithms."""
        scores = {}
        ensemble = 0.0
        for algo in self.algorithms:
            s = algo.buy_score(data)
            scores[algo.name] = s
            ensemble += s * self.algo_weights.get(algo.name, 0.0)
        return round(ensemble, 4), scores

    def _ensemble_sell_score(self, data: StockData, entry_price: float) -> tuple[float, dict[str, float]]:
        """Compute weighted ensemble sell score across all algorithms."""
        scores = {}
        ensemble = 0.0
        for algo in self.algorithms:
            s = algo.sell_score(data, entry_price)
            scores[algo.name] = s
            ensemble += s * self.algo_weights.get(algo.name, 0.0)
        return round(ensemble, 4), scores

    async def _get_current_prices(self, allow_fetch: bool = True) -> dict[str, float]:
        """Get current prices for all held symbols.
        If allow_fetch=False or market is closed, return last known prices.
        """
        symbols = list(self.portfolio.positions.keys())
        if not symbols:
            return {}

        if allow_fetch and is_market_open():
            prices = {}
            tasks = [fetch_current_price(sym) for sym in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, result in zip(symbols, results):
                if isinstance(result, (int, float)) and result > 0:
                    prices[sym] = result
                    self._last_known_prices[sym] = result
                else:
                    prices[sym] = self._last_known_prices.get(
                        sym, self.portfolio.positions[sym].entry_price
                    )
            return prices

        # Off-hours: use last known or entry price
        return {
            sym: self._last_known_prices.get(sym, self.portfolio.positions[sym].entry_price)
            for sym in symbols
        }

    async def sell_loop(self):
        """Check every 15 seconds whether any held stock should be sold."""
        logger.info("Sell loop started (every 15s)")
        while True:
            if is_market_open():
                try:
                    async with self._portfolio_lock:
                        if self.portfolio.positions:
                            await self._check_sells()
                except Exception as e:
                    logger.error(f"Sell loop error: {e}", exc_info=True)

            await asyncio.sleep(config.SELL_CHECK_INTERVAL)

    async def _check_sells(self):
        """Evaluate all positions for sell signals."""
        prices = await self._get_current_prices()

        for symbol in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions.get(symbol)
            if not pos:
                continue

            current_price = prices.get(symbol, pos.entry_price)
            pos.update_peak(current_price)
            pnl_pct = pos.unrealized_pnl_pct(current_price)

            sell_reason = None

            # Hard stop loss
            if pnl_pct <= config.STOP_LOSS_PCT:
                sell_reason = f"Stop loss hit ({pnl_pct:.2%})"

            # Take profit
            elif pnl_pct >= config.TAKE_PROFIT_PCT:
                sell_reason = f"Take profit hit ({pnl_pct:.2%})"

            # Trailing stop
            elif pos.peak_price > pos.entry_price:
                trailing_drop = (current_price - pos.peak_price) / pos.peak_price
                if trailing_drop <= -config.TRAILING_STOP_PCT:
                    sell_reason = f"Trailing stop ({trailing_drop:.2%} from peak ${pos.peak_price:.2f})"

            # Algorithm-based sell signal (check less frequently - use cached data)
            if sell_reason is None and symbol in self._held_data_cache:
                cached = self._held_data_cache[symbol]
                cached_price_attr = cached.price
                # Update the price in cached data for scoring
                cached.price = current_price
                ensemble_sell, _ = self._ensemble_sell_score(cached, pos.entry_price)
                cached.price = cached_price_attr  # restore
                if ensemble_sell > 0.55:
                    sell_reason = f"Algo sell signal (score: {ensemble_sell:.4f})"

            if sell_reason:
                trade = self.portfolio.sell(symbol, current_price, reason=sell_reason)
                if trade:
                    # Remove from cache
                    self._held_data_cache.pop(symbol, None)
                    # Send notification
                    all_prices = await self._get_current_prices()
                    all_prices[symbol] = current_price  # include sold symbol for display
                    status = self.portfolio.format_status(all_prices)
                    await self.notifier.send_sell_signal(
                        symbol=symbol,
                        price=current_price,
                        shares=trade.shares,
                        pnl=trade.pnl,
                        pnl_pct=trade.pnl_pct,
                        reason=sell_reason,
                        portfolio_status=status,
                    )

    async def buy_loop(self):
        """Every 60 seconds, scan universe for buy opportunities if < 5 holdings."""
        logger.info("Buy loop started (every 60s)")
        await asyncio.sleep(5)
        while True:
            if is_market_open():
                try:
                    async with self._portfolio_lock:
                        if self.portfolio.num_holdings < config.MAX_HOLDINGS:
                            await self._scan_for_buys()
                except Exception as e:
                    logger.error(f"Buy loop error: {e}", exc_info=True)

            await asyncio.sleep(config.BUY_SCAN_INTERVAL)

    async def _scan_for_buys(self):
        """Scan stock universe and buy the best candidates."""
        slots_available = config.MAX_HOLDINGS - self.portfolio.num_holdings
        if slots_available <= 0:
            return

        # Exclude already-held stocks
        held = set(self.portfolio.positions.keys())
        candidates = [s for s in config.STOCK_UNIVERSE if s not in held]

        logger.info(f"Scanning {len(candidates)} stocks for buy signals ({slots_available} slots open)")

        # Fetch detailed data for all candidates
        all_data = await fetch_multiple(candidates)

        if not all_data:
            logger.info("No data fetched, skipping buy scan")
            return

        # Score all candidates
        scored = []
        for symbol, data in all_data.items():
            ensemble, scores = self._ensemble_buy_score(data)
            if ensemble >= config.BUY_THRESHOLD:
                scored.append((symbol, data, ensemble, scores))

        # Sort by ensemble score descending
        scored.sort(key=lambda x: x[2], reverse=True)

        # Buy top candidates up to available slots
        for symbol, data, ensemble, scores in scored[:slots_available]:
            available_cash = self.portfolio.cash
            cash_to_spend = available_cash * config.POSITION_SIZE_PCT

            if cash_to_spend < 50:  # minimum trade size
                logger.info("Insufficient cash for more buys")
                break

            trade = self.portfolio.buy(
                symbol=symbol,
                price=data.price,
                cash_to_spend=cash_to_spend,
                reason=f"Ensemble score: {ensemble:.4f}",
            )

            if trade:
                # Cache data for sell scoring
                self._held_data_cache[symbol] = data

                # Send notification
                prices = await self._get_current_prices()
                status = self.portfolio.format_status(prices)
                await self.notifier.send_buy_signal(
                    symbol=symbol,
                    price=data.price,
                    shares=trade.shares,
                    scores=scores,
                    ensemble_score=ensemble,
                    portfolio_status=status,
                )

        # Also refresh cache for existing holdings
        held_symbols = [s for s in self.portfolio.positions if s in all_data]
        for sym in held_symbols:
            self._held_data_cache[sym] = all_data[sym]

    async def heartbeat_loop(self):
        """Send portfolio status every 15 minutes. No API calls off-hours."""
        while True:
            await asyncio.sleep(900)  # 15 minutes
            try:
                prices = await self._get_current_prices(allow_fetch=is_market_open())
                status = self.portfolio.format_status(prices)
                market_tag = "MARKET OPEN" if is_market_open() else "MARKET CLOSED (last known prices)"
                await self.notifier.send_heartbeat(f"[{market_tag}]\n\n{status}")
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def market_monitor_loop(self):
        """Monitor market open/close transitions and notify."""
        while True:
            currently_open = is_market_open()
            if currently_open and not self._was_market_open:
                logger.info("Market just opened")
                await self.notifier.send("MARKET OPENED - Trading active")
            elif not currently_open and self._was_market_open:
                logger.info("Market just closed")
                prices = await self._get_current_prices(allow_fetch=False)
                status = self.portfolio.format_status(prices)
                await self.notifier.send(f"MARKET CLOSED - Trading paused\n\n{status}")
            self._was_market_open = currently_open

            if not currently_open:
                # Sleep longer when market is closed to avoid busy-waiting
                wait = seconds_until_market_open()
                if wait > 60:
                    logger.info(f"Market closed. Next open in {wait/3600:.1f}h. Sleeping...")
                    # Wake every 5 min to keep heartbeat alive, or until market opens
                    await asyncio.sleep(min(wait, 300))
                else:
                    await asyncio.sleep(10)
            else:
                await asyncio.sleep(30)

    async def run(self):
        """Start the engine with parallel buy and sell loops."""
        logger.info("Starting trading engine...")
        logger.info(f"Initial capital: ${config.INITIAL_CAPITAL:,.2f}")
        logger.info(f"Stock universe: {len(config.STOCK_UNIVERSE)} symbols")
        logger.info(f"Algorithms: {[a.name for a in self.algorithms]}")

        market_status = "OPEN" if is_market_open() else "CLOSED"
        self._was_market_open = is_market_open()
        logger.info(f"Market status: {market_status}")

        # Send startup notification (no API call if market closed)
        prices = await self._get_current_prices(allow_fetch=is_market_open())
        status = self.portfolio.format_status(prices)
        await self.notifier.send_startup(f"[Market: {market_status}]\n\n{status}")

        # Run all loops concurrently
        await asyncio.gather(
            self.sell_loop(),
            self.buy_loop(),
            self.heartbeat_loop(),
            self.market_monitor_loop(),
        )
