"""
Main trading engine with parallel buy/sell loops and EOD position management.

- Sell check: every 15 seconds on all held positions
- Buy scan: every 60 seconds if < 10 holdings
- EOD close: at 3:45 PM ET, close positions unless strong swing signal
- All loops run concurrently via asyncio
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone, timedelta

import config
from algorithms import VolArbAlgorithm, GammaExposureAlgorithm, OptionsFlowAlgorithm
from data_fetcher import fetch_current_price, fetch_multiple, StockData
from portfolio import Portfolio
from telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)

# US Eastern timezone
ET = timezone(timedelta(hours=-4))  # EDT; adjust to -5 for EST

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
EOD_CLOSE_TIME = time(15, 45)  # 3:45 PM ET — start closing positions
WEEKDAYS = range(0, 5)


def is_market_open() -> bool:
    now_et = datetime.now(ET)
    if now_et.weekday() not in WEEKDAYS:
        return False
    return MARKET_OPEN <= now_et.time() < MARKET_CLOSE


def is_eod_window() -> bool:
    """Are we in the end-of-day closing window (3:45 PM - 4:00 PM ET)?"""
    now_et = datetime.now(ET)
    if now_et.weekday() not in WEEKDAYS:
        return False
    return EOD_CLOSE_TIME <= now_et.time() < MARKET_CLOSE


def seconds_until_market_open() -> float:
    if is_market_open():
        return 0.0
    now_et = datetime.now(ET)
    target = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et.time() >= MARKET_CLOSE or now_et.weekday() >= 5:
        days_ahead = 1
        candidate = now_et + timedelta(days=days_ahead)
        while candidate.weekday() not in WEEKDAYS:
            days_ahead += 1
            candidate = now_et + timedelta(days=days_ahead)
        target = candidate.replace(hour=9, minute=30, second=0, microsecond=0)
    return max((target - now_et).total_seconds(), 0.0)


class TradingEngine:
    def __init__(self, notifier: TelegramNotifier):
        self.portfolio = Portfolio(initial_capital=config.INITIAL_CAPITAL)
        self.notifier = notifier

        self.algorithms = [
            VolArbAlgorithm(),
            GammaExposureAlgorithm(),
            OptionsFlowAlgorithm(),
        ]
        self.algo_weights = config.ALGO_WEIGHTS

        self._held_data_cache: dict[str, StockData] = {}
        self._last_known_prices: dict[str, float] = {}
        self._portfolio_lock = asyncio.Lock()
        self._was_market_open = False
        self._eod_closed_today = False  # Track if we already ran EOD close

    def _ensemble_buy_score(self, data: StockData) -> tuple[float, dict[str, float]]:
        scores = {}
        ensemble = 0.0
        for algo in self.algorithms:
            s = algo.buy_score(data)
            scores[algo.name] = s
            ensemble += s * self.algo_weights.get(algo.name, 0.0)
        return round(ensemble, 4), scores

    def _ensemble_sell_score(self, data: StockData, entry_price: float) -> tuple[float, dict[str, float]]:
        scores = {}
        ensemble = 0.0
        for algo in self.algorithms:
            s = algo.sell_score(data, entry_price)
            scores[algo.name] = s
            ensemble += s * self.algo_weights.get(algo.name, 0.0)
        return round(ensemble, 4), scores

    def _ensemble_swing_score(self, data: StockData) -> float:
        """Aggregate swing score: should we hold this overnight?"""
        total = 0.0
        weights = {"vol_arb": 0.35, "gamma_exposure": 0.35, "options_flow": 0.30}
        for algo in self.algorithms:
            s = algo.swing_score(data)
            total += s * weights.get(algo.name, 0.0)
        return round(total, 4)

    async def _get_current_prices(self, allow_fetch: bool = True) -> dict[str, float]:
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
        return {
            sym: self._last_known_prices.get(sym, self.portfolio.positions[sym].entry_price)
            for sym in symbols
        }

    # ------------------------------------------------------------------
    # Sell loop: every 15 seconds
    # ------------------------------------------------------------------

    async def sell_loop(self):
        logger.info("Sell loop started (every 15s)")
        while True:
            if is_market_open() and not is_eod_window():
                try:
                    async with self._portfolio_lock:
                        if self.portfolio.positions:
                            await self._check_sells()
                except Exception as e:
                    logger.error(f"Sell loop error: {e}", exc_info=True)
            await asyncio.sleep(config.SELL_CHECK_INTERVAL)

    async def _check_sells(self):
        prices = await self._get_current_prices()
        for symbol in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions.get(symbol)
            if not pos:
                continue
            current_price = prices.get(symbol, pos.entry_price)
            pos.update_peak(current_price)
            pnl_pct = pos.unrealized_pnl_pct(current_price)
            sell_reason = None

            if pnl_pct <= config.STOP_LOSS_PCT:
                sell_reason = f"STOP LOSS ({pnl_pct:.2%})"
            elif pnl_pct >= config.TAKE_PROFIT_PCT:
                sell_reason = f"TAKE PROFIT ({pnl_pct:.2%})"
            elif pos.peak_price > pos.entry_price:
                trailing_drop = (current_price - pos.peak_price) / pos.peak_price
                if trailing_drop <= -config.TRAILING_STOP_PCT:
                    sell_reason = f"TRAILING STOP ({trailing_drop:.2%} from peak ${pos.peak_price:.2f})"

            # Algo-based sell signal
            if sell_reason is None and symbol in self._held_data_cache:
                cached = self._held_data_cache[symbol]
                orig_price = cached.price
                cached.price = current_price
                ensemble_sell, _ = self._ensemble_sell_score(cached, pos.entry_price)
                cached.price = orig_price
                if ensemble_sell > 0.55:
                    sell_reason = f"ALGO SELL (score: {ensemble_sell:.4f})"

            if sell_reason:
                trade = self.portfolio.sell(symbol, current_price, reason=sell_reason)
                if trade:
                    self._held_data_cache.pop(symbol, None)
                    all_prices = await self._get_current_prices()
                    status = self.portfolio.format_status(all_prices)
                    await self.notifier.send_sell_signal(
                        symbol=symbol, price=current_price,
                        shares=trade.shares, pnl=trade.pnl,
                        pnl_pct=trade.pnl_pct, reason=sell_reason,
                        portfolio_status=status,
                    )

    # ------------------------------------------------------------------
    # Buy loop: every 60 seconds
    # ------------------------------------------------------------------

    async def buy_loop(self):
        logger.info("Buy loop started (every 60s)")
        await asyncio.sleep(5)
        while True:
            # Don't buy during EOD window — we're closing, not opening
            if is_market_open() and not is_eod_window():
                try:
                    async with self._portfolio_lock:
                        if self.portfolio.num_holdings < config.MAX_HOLDINGS:
                            await self._scan_for_buys()
                except Exception as e:
                    logger.error(f"Buy loop error: {e}", exc_info=True)
            await asyncio.sleep(config.BUY_SCAN_INTERVAL)

    async def _scan_for_buys(self):
        slots_available = config.MAX_HOLDINGS - self.portfolio.num_holdings
        if slots_available <= 0:
            return
        held = set(self.portfolio.positions.keys())
        candidates = [s for s in config.STOCK_UNIVERSE if s not in held]
        logger.info(f"Scanning {len(candidates)} stocks ({slots_available} slots open)")

        all_data = await fetch_multiple(candidates)
        if not all_data:
            logger.info("No data fetched, skipping buy scan")
            return

        scored = []
        for symbol, data in all_data.items():
            ensemble, scores = self._ensemble_buy_score(data)
            if ensemble >= config.BUY_THRESHOLD:
                scored.append((symbol, data, ensemble, scores))
        scored.sort(key=lambda x: x[2], reverse=True)

        for symbol, data, ensemble, scores in scored[:slots_available]:
            available_cash = self.portfolio.cash
            cash_to_spend = available_cash * config.POSITION_SIZE_PCT
            if cash_to_spend < 50:
                logger.info("Insufficient cash for more buys")
                break

            trade = self.portfolio.buy(
                symbol=symbol, price=data.price,
                cash_to_spend=cash_to_spend,
                reason=f"Ensemble: {ensemble:.4f}",
            )
            if trade:
                self._held_data_cache[symbol] = data
                prices = await self._get_current_prices()
                status = self.portfolio.format_status(prices)
                await self.notifier.send_buy_signal(
                    symbol=symbol, price=data.price,
                    shares=trade.shares, scores=scores,
                    ensemble_score=ensemble, portfolio_status=status,
                )

        # Refresh cache for existing holdings
        for sym in list(self.portfolio.positions.keys()):
            if sym in all_data:
                self._held_data_cache[sym] = all_data[sym]

    # ------------------------------------------------------------------
    # EOD closing loop: close positions at 3:45 PM unless swing signal
    # ------------------------------------------------------------------

    async def eod_close_loop(self):
        """At 3:45 PM ET, evaluate all positions for overnight hold viability.
        Close positions that don't have a strong swing signal."""
        logger.info("EOD close loop active (triggers at 3:45 PM ET)")
        while True:
            now_et = datetime.now(ET)

            # Reset the daily flag at market open
            if now_et.time() < EOD_CLOSE_TIME:
                self._eod_closed_today = False

            if is_eod_window() and not self._eod_closed_today:
                try:
                    async with self._portfolio_lock:
                        await self._run_eod_close()
                    self._eod_closed_today = True
                except Exception as e:
                    logger.error(f"EOD close error: {e}", exc_info=True)

            await asyncio.sleep(30)

    async def _run_eod_close(self):
        """Evaluate each position: close unless strong swing signal."""
        if not self.portfolio.positions:
            return

        logger.info("=== EOD POSITION REVIEW ===")
        prices = await self._get_current_prices()
        swing_holds = []
        closes = []

        for symbol in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions.get(symbol)
            if not pos:
                continue

            current_price = prices.get(symbol, pos.entry_price)

            # Check swing viability using cached data
            swing = 0.0
            if symbol in self._held_data_cache:
                cached = self._held_data_cache[symbol]
                cached.price = current_price
                swing = self._ensemble_swing_score(cached)

            if swing >= config.SWING_HOLD_THRESHOLD:
                swing_holds.append((symbol, swing, pos.unrealized_pnl_pct(current_price)))
                logger.info(f"SWING HOLD: {symbol} (swing score: {swing:.4f})")
            else:
                # Close this position
                pnl_pct = pos.unrealized_pnl_pct(current_price)
                trade = self.portfolio.sell(
                    symbol, current_price,
                    reason=f"EOD CLOSE (swing score: {swing:.4f} < {config.SWING_HOLD_THRESHOLD})"
                )
                if trade:
                    self._held_data_cache.pop(symbol, None)
                    closes.append((symbol, trade.pnl, trade.pnl_pct))
                    logger.info(f"EOD SELL: {symbol} P&L: ${trade.pnl:+.2f} ({trade.pnl_pct:+.2%})")

        # Send EOD summary
        all_prices = await self._get_current_prices()
        status = self.portfolio.format_status(all_prices)

        hold_lines = "\n".join(
            f"  {sym} (swing: {sw:.4f}, P&L: {pnl:+.2%})"
            for sym, sw, pnl in swing_holds
        ) or "  None"
        close_lines = "\n".join(
            f"  {sym}: ${pnl:+.2f} ({pnl_pct:+.2%})"
            for sym, pnl, pnl_pct in closes
        ) or "  None"

        msg = (
            f"=== EOD POSITION REVIEW (3:45 PM ET) ===\n\n"
            f"HOLDING OVERNIGHT:\n{hold_lines}\n\n"
            f"CLOSED:\n{close_lines}\n\n"
            f"{status}"
        )
        await self.notifier.send(msg)

    # ------------------------------------------------------------------
    # Heartbeat & market monitor
    # ------------------------------------------------------------------

    async def heartbeat_loop(self):
        while True:
            await asyncio.sleep(3600)
            try:
                prices = await self._get_current_prices(allow_fetch=is_market_open())
                status = self.portfolio.format_status(prices)
                tag = "MARKET OPEN" if is_market_open() else "MARKET CLOSED (last known prices)"
                await self.notifier.send_heartbeat(f"[{tag}]\n\n{status}")
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def market_monitor_loop(self):
        while True:
            currently_open = is_market_open()
            if currently_open and not self._was_market_open:
                logger.info("Market just opened")
                self._eod_closed_today = False
                await self.notifier.send("MARKET OPENED - Trading active\nAlgos: Vol Arb | GEX/Dealer Flow | Options Flow")
            elif not currently_open and self._was_market_open:
                logger.info("Market just closed")
                prices = await self._get_current_prices(allow_fetch=False)
                status = self.portfolio.format_status(prices)
                await self.notifier.send(f"MARKET CLOSED - Trading paused\n\n{status}")
            self._was_market_open = currently_open

            if not currently_open:
                wait = seconds_until_market_open()
                if wait > 60:
                    logger.info(f"Market closed. Next open in {wait/3600:.1f}h")
                    await asyncio.sleep(min(wait, 300))
                else:
                    await asyncio.sleep(10)
            else:
                await asyncio.sleep(30)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def run(self):
        logger.info("Starting options-driven trading engine...")
        logger.info(f"Initial capital: ${config.INITIAL_CAPITAL:,.2f}")
        logger.info(f"Max holdings: {config.MAX_HOLDINGS}")
        logger.info(f"Stock universe: {len(config.STOCK_UNIVERSE)} symbols")
        logger.info(f"Algorithms: {[a.name for a in self.algorithms]}")
        logger.info(f"EOD close at 3:45 PM ET, swing threshold: {config.SWING_HOLD_THRESHOLD}")

        market_status = "OPEN" if is_market_open() else "CLOSED"
        self._was_market_open = is_market_open()

        prices = await self._get_current_prices(allow_fetch=is_market_open())
        status = self.portfolio.format_status(prices)
        await self.notifier.send_startup(
            f"[Market: {market_status}]\n"
            f"Algos: Vol Arb | GEX/Dealer Flow | Options Flow\n"
            f"EOD close: 3:45 PM ET (swing override at {config.SWING_HOLD_THRESHOLD})\n\n"
            f"{status}"
        )

        await asyncio.gather(
            self.sell_loop(),
            self.buy_loop(),
            self.eod_close_loop(),
            self.heartbeat_loop(),
            self.market_monitor_loop(),
        )
