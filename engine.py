"""
Options trading engine with parallel buy/sell loops and EOD position management.

Trades actual option contracts (calls and puts) based on ensemble signals.
- Buy scan: every 60 seconds — get direction signals, select contracts, buy
- Sell check: every 15 seconds — Greeks-based exits + premium stops
- EOD close: at 3:45 PM ET, close near-expiry + weak swing positions
- All loops run concurrently via asyncio

Risk overlays:
- Vol regime detection via VIX — adapts sizing, thresholds, algo weights
- Sector correlation management — max 3 per sector, portfolio beta cap
- Kelly criterion position sizing — conviction-proportional allocation
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone, timedelta

import config
from algorithms import VolArbAlgorithm, GammaExposureAlgorithm, OptionsFlowAlgorithm, OptionSignal
from correlation import CorrelationManager
from data_fetcher import (
    fetch_current_price, fetch_multiple, fetch_option_prices_batch,
    fetch_vix, select_optimal_contract, StockData,
)
from portfolio import Portfolio
from position_sizer import PositionSizer
from regime import VolRegimeDetector, RegimeParams
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

        # Risk overlays
        self.regime_detector = VolRegimeDetector()
        self.correlation_mgr = CorrelationManager()
        self.position_sizer = PositionSizer()
        self.current_regime = RegimeParams()  # defaults to NORMAL

        self._held_data_cache: dict[str, StockData] = {}  # symbol -> StockData
        self._portfolio_lock = asyncio.Lock()
        self._was_market_open = False
        self._eod_closed_today = False

    @property
    def _active_weights(self) -> dict[str, float]:
        """Return regime-adjusted algo weights."""
        return self.current_regime.algo_weights

    def _ensemble_signal(self, data: StockData) -> tuple[str, float, dict[str, OptionSignal]]:
        """Compute ensemble direction and conviction from all algorithms.

        Returns:
            (direction, conviction, algo_signals)
            direction: "CALL", "PUT", or "NEUTRAL"
            conviction: 0-1 weighted conviction
            algo_signals: per-algorithm OptionSignal objects
        """
        weights = self._active_weights
        algo_signals = {}
        call_conviction = 0.0
        put_conviction = 0.0

        for algo in self.algorithms:
            sig = algo.signal(data)
            algo_signals[algo.name] = sig
            w = weights.get(algo.name, 0.0)

            if sig.direction == "CALL":
                call_conviction += sig.conviction * w
            elif sig.direction == "PUT":
                put_conviction += sig.conviction * w

        # Direction by weighted conviction
        if call_conviction > put_conviction and call_conviction > 0.05:
            return "CALL", round(call_conviction, 4), algo_signals
        elif put_conviction > call_conviction and put_conviction > 0.05:
            return "PUT", round(put_conviction, 4), algo_signals
        else:
            return "NEUTRAL", 0.0, algo_signals

    def _ensemble_sell_score(self, data: StockData, position_direction: str) -> float:
        """Compute ensemble exit score for a held position."""
        weights = self._active_weights
        total = 0.0
        for algo in self.algorithms:
            s = algo.sell_score(data, position_direction)
            total += s * weights.get(algo.name, 0.0)
        return round(total, 4)

    def _ensemble_swing_score(self, data: StockData) -> float:
        """Aggregate swing score: should we hold overnight?"""
        weights = self._active_weights
        total = 0.0
        for algo in self.algorithms:
            s = algo.swing_score(data)
            total += s * weights.get(algo.name, 0.0)
        return round(total, 4)

    def _weighted_preferred_params(self, algo_signals: dict[str, OptionSignal]) -> tuple[float, int]:
        """Compute weighted preferred delta and DTE from algorithm signals."""
        weights = self._active_weights
        total_delta = 0.0
        total_dte = 0.0
        total_weight = 0.0
        for name, sig in algo_signals.items():
            if sig.direction == "NEUTRAL":
                continue
            w = weights.get(name, 0.0) * sig.conviction
            total_delta += sig.preferred_delta * w
            total_dte += sig.preferred_dte * w
            total_weight += w
        if total_weight > 0:
            return total_delta / total_weight, int(total_dte / total_weight)
        return 0.40, 30

    async def _get_current_premiums(self) -> dict[str, float]:
        """Fetch current premiums for all held option positions."""
        if not self.portfolio.positions:
            return {}

        position_specs = [
            (pos.symbol, pos.strike, pos.expiry, pos.option_type)
            for pos in self.portfolio.positions.values()
        ]
        price_data = await fetch_option_prices_batch(position_specs)

        premiums = {}
        for key, pos in self.portfolio.positions.items():
            if key in price_data and price_data[key]:
                premiums[key] = price_data[key]["premium"]
            else:
                premiums[key] = pos.entry_premium  # fallback
        return premiums

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
        # Fetch current option prices + Greeks for all positions
        position_specs = [
            (pos.symbol, pos.strike, pos.expiry, pos.option_type)
            for pos in self.portfolio.positions.values()
        ]
        price_data = await fetch_option_prices_batch(position_specs)

        # Regime-adjusted stop levels
        regime_stop = abs(config.PREMIUM_STOP_LOSS_PCT) * self.current_regime.stop_loss_mult
        regime_trailing = config.PREMIUM_TRAILING_STOP_PCT * self.current_regime.trailing_stop_mult

        for key in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions.get(key)
            if not pos:
                continue

            option_data = price_data.get(key)
            if not option_data:
                continue

            current_premium = option_data["premium"]
            current_delta = option_data.get("delta", 0.0)
            current_iv = option_data.get("iv", 0.0)
            current_theta = option_data.get("theta", 0.0)
            current_dte = option_data.get("dte", 999)

            pos.update_peak(current_premium)
            pnl_pct = pos.unrealized_pnl_pct(current_premium)
            sell_reason = None

            # Tiered exit logic:
            # - Hard stops (premium stop/take-profit/trailing): always active
            # - Soft/technical exits (Greeks, algo): gated behind min hold time

            # === HARD STOPS — always active, regime-adjusted ===

            # 1. Premium stop loss (regime widens in high vol)
            if pnl_pct <= -regime_stop:
                sell_reason = f"PREMIUM STOP LOSS ({pnl_pct:+.2%})"

            # 2. Premium take profit: up 100%
            elif pnl_pct >= config.PREMIUM_TAKE_PROFIT_PCT:
                sell_reason = f"PREMIUM TAKE PROFIT ({pnl_pct:+.2%})"

            # 3. Trailing stop (regime widens in high vol)
            elif pos.peak_premium > pos.entry_premium:
                trailing_drop = (current_premium - pos.peak_premium) / pos.peak_premium
                if trailing_drop <= -regime_trailing:
                    sell_reason = (
                        f"TRAILING STOP ({trailing_drop:+.2%} from peak "
                        f"${pos.peak_premium:.2f})"
                    )

            # === SOFT/TECHNICAL EXITS — need time to develop ===
            hold_seconds = (datetime.now() - pos.entry_time).total_seconds()
            past_cooldown = hold_seconds >= config.MIN_HOLD_SECONDS

            # 4. Delta decay: option going deep OTM
            if sell_reason is None and past_cooldown and abs(current_delta) < config.DELTA_STOP_LOSS:
                sell_reason = f"DELTA DECAY (Δ{current_delta:.3f} < {config.DELTA_STOP_LOSS})"

            # 5. IV crush: IV dropped significantly from entry
            if sell_reason is None and past_cooldown and pos.entry_iv > 0 and current_iv > 0:
                iv_change = (current_iv - pos.entry_iv) / pos.entry_iv
                if iv_change < -config.IV_CRUSH_EXIT_PCT:
                    sell_reason = f"IV CRUSH (IV: {pos.entry_iv:.1%} → {current_iv:.1%}, {iv_change:+.1%})"

            # 6. Theta bleed: daily decay exceeds threshold
            if sell_reason is None and past_cooldown and current_premium > 0:
                daily_theta_pct = abs(current_theta) / current_premium
                if daily_theta_pct > config.MAX_THETA_DECAY_PCT:
                    sell_reason = f"THETA BLEED (Θ{current_theta:.3f}, {daily_theta_pct:.1%}/day)"

            # 7. Near expiry: force close within 3 DTE (gamma risk)
            if sell_reason is None and current_dte <= config.NEAR_EXPIRY_DTE:
                sell_reason = f"NEAR EXPIRY ({current_dte} DTE)"

            # 8. Algo-based sell signal
            if sell_reason is None and past_cooldown and pos.symbol in self._held_data_cache:
                cached = self._held_data_cache[pos.symbol]
                ensemble_sell = self._ensemble_sell_score(cached, pos.option_type)
                if ensemble_sell > 0.55:
                    sell_reason = f"ALGO SELL (score: {ensemble_sell:.4f})"

            if sell_reason:
                trade = self.portfolio.sell(key, current_premium, reason=sell_reason)
                if trade:
                    premiums = await self._get_current_premiums()
                    status = self.portfolio.format_status(premiums)
                    await self.notifier.send_sell_signal(
                        display_name=trade.display_name,
                        premium=current_premium,
                        contracts=trade.contracts,
                        pnl=trade.pnl,
                        pnl_pct=trade.pnl_pct,
                        reason=sell_reason,
                        greeks={
                            "delta": current_delta,
                            "iv": current_iv,
                            "theta": current_theta,
                            "dte": current_dte,
                        },
                        portfolio_status=status,
                    )

    # ------------------------------------------------------------------
    # Buy loop: every 60 seconds
    # ------------------------------------------------------------------

    async def buy_loop(self):
        logger.info("Buy loop started (every 60s)")
        await asyncio.sleep(5)
        while True:
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

        # --- Regime detection: fetch VIX every buy cycle ---
        vix_level = await fetch_vix()
        self.current_regime = self.regime_detector.detect(vix_level)
        regime = self.current_regime

        # Regime-adjusted buy threshold
        buy_threshold = config.BUY_THRESHOLD + regime.buy_threshold_adj

        # Exclude underlyings we already have positions on
        held_symbols = {pos.symbol for pos in self.portfolio.positions.values()}
        candidates = [s for s in config.STOCK_UNIVERSE if s not in held_symbols]
        logger.info(
            f"Scanning {len(candidates)} stocks ({slots_available} slots open) "
            f"[{regime.regime.value} | VIX: {regime.vix_level:.1f} | "
            f"threshold: {buy_threshold:.2f}]"
        )

        all_data = await fetch_multiple(candidates)
        if not all_data:
            logger.info("No data fetched, skipping buy scan")
            return

        # Score all candidates with ensemble signal
        scored = []
        for symbol, data in all_data.items():
            direction, conviction, algo_signals = self._ensemble_signal(data)
            if direction in ("CALL", "PUT") and conviction >= buy_threshold:
                # Apply diversification penalty to conviction for ranking
                div_mult = self.correlation_mgr.diversification_multiplier(
                    symbol, self.portfolio.positions
                )
                if div_mult <= 0:
                    logger.info(
                        f"Skipping {symbol}: sector "
                        f"'{self.correlation_mgr.sector_of(symbol)}' at max"
                    )
                    continue
                adj_conviction = conviction * div_mult
                scored.append((symbol, data, direction, conviction, adj_conviction, algo_signals))

        # Sort by diversification-adjusted conviction descending
        scored.sort(key=lambda x: x[4], reverse=True)

        for symbol, data, direction, conviction, adj_conviction, algo_signals in scored[:slots_available]:
            # Correlation gate: check sector and beta limits
            premiums = await self._get_current_premiums()
            equity = self.portfolio.total_equity(premiums)
            allowed, reason = self.correlation_mgr.can_add_position(
                symbol=symbol,
                positions=self.portfolio.positions,
                data_cache=self._held_data_cache,
                candidate_beta=data.beta,
                total_equity=equity,
            )
            if not allowed:
                logger.info(f"Correlation block: {symbol} — {reason}")
                continue

            # Get weighted preferred params from algorithm signals
            pref_delta, pref_dte = self._weighted_preferred_params(algo_signals)

            # Select optimal contract
            contract = select_optimal_contract(
                symbol=symbol,
                greeks_chain=data.greeks_chain,
                direction=direction,
                spot_price=data.price,
                preferred_delta=pref_delta,
                preferred_dte=pref_dte,
            )
            if not contract:
                logger.info(f"No suitable {direction} contract for {symbol}, skipping")
                continue

            # Kelly criterion position sizing
            cost_per_contract = contract.premium * config.CONTRACT_MULTIPLIER
            if cost_per_contract <= 0:
                continue

            num_contracts = self.position_sizer.compute_size(
                conviction=conviction,
                regime=regime,
                available_cash=self.portfolio.cash,
                contract_cost=cost_per_contract,
            )
            if num_contracts <= 0:
                logger.info(f"Kelly says skip {symbol} (insufficient edge or cash)")
                continue

            trade = self.portfolio.buy(
                symbol=symbol,
                strike=contract.strike,
                expiry=contract.expiry,
                option_type=contract.option_type,
                premium=contract.premium,
                contracts=num_contracts,
                entry_iv=contract.iv,
                entry_delta=contract.delta,
                reason=f"Ensemble: {conviction:.4f} {direction}",
            )

            if trade:
                self._held_data_cache[symbol] = data
                premiums = await self._get_current_premiums()
                status = self.portfolio.format_status(premiums)

                # Build scores summary for notification
                scores_summary = {}
                for name, sig in algo_signals.items():
                    scores_summary[name] = f"{sig.direction} ({sig.conviction:.4f})"

                sector = self.correlation_mgr.sector_of(symbol)
                await self.notifier.send_buy_signal(
                    display_name=trade.display_name,
                    premium=contract.premium,
                    contracts=num_contracts,
                    scores=scores_summary,
                    ensemble_direction=direction,
                    ensemble_conviction=conviction,
                    greeks={
                        "delta": contract.delta,
                        "gamma": contract.gamma,
                        "theta": contract.theta,
                        "vega": contract.vega,
                        "iv": contract.iv,
                        "dte": contract.dte,
                    },
                    portfolio_status=(
                        f"Regime: {regime.regime.value} (VIX: {regime.vix_level:.1f})\n"
                        f"Sector: {sector}\n"
                        f"Kelly size: {num_contracts} contracts\n\n"
                        f"{status}"
                    ),
                )

        # Refresh cache for existing holdings
        for pos in self.portfolio.positions.values():
            if pos.symbol in all_data:
                self._held_data_cache[pos.symbol] = all_data[pos.symbol]

    # ------------------------------------------------------------------
    # EOD closing loop
    # ------------------------------------------------------------------

    async def eod_close_loop(self):
        logger.info("EOD close loop active (triggers at 3:45 PM ET)")
        while True:
            now_et = datetime.now(ET)
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
        if not self.portfolio.positions:
            return

        logger.info("=== EOD POSITION REVIEW ===")
        premiums = await self._get_current_premiums()
        swing_holds = []
        closes = []

        for key in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions.get(key)
            if not pos:
                continue

            current_premium = premiums.get(key, pos.entry_premium)

            # Force close near-expiry options (gamma risk)
            try:
                exp_date = datetime.strptime(pos.expiry, "%Y-%m-%d")
                dte = max((exp_date - datetime.now()).days, 0)
            except ValueError:
                dte = 999

            if dte <= config.NEAR_EXPIRY_DTE:
                trade = self.portfolio.sell(
                    key, current_premium,
                    reason=f"EOD NEAR EXPIRY ({dte} DTE)"
                )
                if trade:
                    closes.append((pos.display_name, trade.pnl, trade.pnl_pct))
                continue

            # Check swing viability
            swing = 0.0
            if pos.symbol in self._held_data_cache:
                cached = self._held_data_cache[pos.symbol]
                cached.price = current_premium  # Not ideal but maintains interface
                swing = self._ensemble_swing_score(cached)

            if swing >= config.SWING_HOLD_THRESHOLD:
                swing_holds.append((pos.display_name, swing, pos.unrealized_pnl_pct(current_premium)))
                logger.info(f"SWING HOLD: {pos.display_name} (score: {swing:.4f})")
            else:
                trade = self.portfolio.sell(
                    key, current_premium,
                    reason=f"EOD CLOSE (swing: {swing:.4f} < {config.SWING_HOLD_THRESHOLD})"
                )
                if trade:
                    closes.append((pos.display_name, trade.pnl, trade.pnl_pct))

        premiums = await self._get_current_premiums()
        status = self.portfolio.format_status(premiums)

        hold_lines = "\n".join(
            f"  {name} (swing: {sw:.4f}, P&L: {pnl:+.2%})"
            for name, sw, pnl in swing_holds
        ) or "  None"
        close_lines = "\n".join(
            f"  {name}: ${pnl:+.2f} ({pnl_pct:+.2%})"
            for name, pnl, pnl_pct in closes
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
                premiums = await self._get_current_premiums()
                status = self.portfolio.format_status(premiums)
                regime = self.current_regime
                tag = "MARKET OPEN" if is_market_open() else "MARKET CLOSED"
                await self.notifier.send_heartbeat(
                    f"[{tag}] Regime: {regime.regime.value} (VIX: {regime.vix_level:.1f})\n\n{status}"
                )
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def market_monitor_loop(self):
        while True:
            currently_open = is_market_open()
            if currently_open and not self._was_market_open:
                logger.info("Market just opened")
                self._eod_closed_today = False
                await self.notifier.send(
                    "MARKET OPENED - Options Trading Active\n"
                    "Algos: Vol Arb | GEX/Dealer Flow | Options Flow\n"
                    "Trading: CALLS & PUTS\n"
                    "Risk: VIX Regime | Sector Limits | Kelly Sizing"
                )
            elif not currently_open and self._was_market_open:
                logger.info("Market just closed")
                premiums = await self._get_current_premiums()
                status = self.portfolio.format_status(premiums)
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
        logger.info("Starting options trading engine...")
        logger.info(f"Initial capital: ${config.INITIAL_CAPITAL:,.2f}")
        logger.info(f"Max holdings: {config.MAX_HOLDINGS}")
        logger.info(f"Stock universe: {len(config.STOCK_UNIVERSE)} symbols")
        logger.info(f"Algorithms: {[a.name for a in self.algorithms]}")
        logger.info(f"Target delta: {config.TARGET_DELTA_RANGE}, DTE: {config.PREFERRED_DTE_MIN}-{config.PREFERRED_DTE_MAX}")
        logger.info(f"Risk overlays: VIX regime | Sector limits ({config.MAX_PER_SECTOR}/sector) | Kelly sizing ({config.KELLY_FRACTION}x)")
        logger.info(f"EOD close at 3:45 PM ET, swing threshold: {config.SWING_HOLD_THRESHOLD}")

        market_status = "OPEN" if is_market_open() else "CLOSED"
        self._was_market_open = is_market_open()

        # Initial VIX check
        vix_level = await fetch_vix()
        self.current_regime = self.regime_detector.detect(vix_level)

        premiums = await self._get_current_premiums()
        status = self.portfolio.format_status(premiums)
        regime = self.current_regime
        await self.notifier.send_startup(
            f"[Market: {market_status}]\n"
            f"Regime: {regime.regime.value} (VIX: {regime.vix_level:.1f})\n"
            f"Algos: Vol Arb | GEX/Dealer Flow | Options Flow\n"
            f"Trading: CALLS & PUTS\n"
            f"Delta: {config.TARGET_DELTA_RANGE} | DTE: {config.PREFERRED_DTE_MIN}-{config.PREFERRED_DTE_MAX}\n"
            f"Risk: {config.MAX_PER_SECTOR}/sector | Beta cap: {config.MAX_PORTFOLIO_BETA} | Kelly: {config.KELLY_FRACTION}x\n"
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
