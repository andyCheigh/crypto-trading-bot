#!/usr/bin/env python3
"""
30-day backtest for the options trading system.

Trades actual option contracts (calls and puts) with Greeks-based P&L simulation.
- Walks through each trading day in the last 30 calendar days
- Generates CALL/PUT signals via ensemble of 3 algorithms
- Selects optimal contracts by delta/DTE/liquidity
- Simulates premium changes: ΔPremium ≈ δ×ΔS + ½γ×ΔS² + θ×Δt + ν×ΔIV
- Greeks-based exits: delta decay, IV crush, theta bleed, near-expiry

Limitation: Options data (GEX, IV, flow) is a current snapshot, not historical.
Price-based metrics (returns, ATR, RV) are properly point-in-time.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

import config
from algorithms import VolArbAlgorithm, GammaExposureAlgorithm, OptionsFlowAlgorithm, OptionSignal
from data_fetcher import (
    StockData, GEXProfile, IVSurface, OptionsFlow, VolumeProfile,
    ContractRecommendation, select_optimal_contract,
    _compute_atr, _compute_volume_profile, _compute_full_options,
    _compute_fundamentals_lite,
)
from portfolio import Portfolio
from telegram_bot import TelegramNotifier

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# Algorithms
ALGOS = [VolArbAlgorithm(), GammaExposureAlgorithm(), OptionsFlowAlgorithm()]


def ensemble_signal(data: StockData) -> tuple[str, float, dict[str, OptionSignal]]:
    """Compute ensemble direction and conviction."""
    algo_signals = {}
    call_conviction = 0.0
    put_conviction = 0.0

    for algo in ALGOS:
        sig = algo.signal(data)
        algo_signals[algo.name] = sig
        w = config.ALGO_WEIGHTS.get(algo.name, 0.0)
        if sig.direction == "CALL":
            call_conviction += sig.conviction * w
        elif sig.direction == "PUT":
            put_conviction += sig.conviction * w

    if call_conviction > put_conviction and call_conviction > 0.05:
        return "CALL", round(call_conviction, 4), algo_signals
    elif put_conviction > call_conviction and put_conviction > 0.05:
        return "PUT", round(put_conviction, 4), algo_signals
    return "NEUTRAL", 0.0, algo_signals


def ensemble_sell_score(data: StockData, position_direction: str) -> float:
    """Compute ensemble exit score."""
    total = 0.0
    for algo in ALGOS:
        s = algo.sell_score(data, position_direction)
        total += s * config.ALGO_WEIGHTS.get(algo.name, 0.0)
    return round(total, 4)


def ensemble_swing_score(data: StockData) -> float:
    total = 0.0
    for algo in ALGOS:
        s = algo.swing_score(data)
        total += s * config.ALGO_WEIGHTS.get(algo.name, 0.0)
    return round(total, 4)


def weighted_preferred_params(algo_signals: dict[str, OptionSignal]) -> tuple[float, int]:
    """Compute weighted preferred delta and DTE."""
    total_delta = 0.0
    total_dte = 0.0
    total_weight = 0.0
    for name, sig in algo_signals.items():
        if sig.direction == "NEUTRAL":
            continue
        w = config.ALGO_WEIGHTS.get(name, 0.0) * sig.conviction
        total_delta += sig.preferred_delta * w
        total_dte += sig.preferred_dte * w
        total_weight += w
    if total_weight > 0:
        return total_delta / total_weight, int(total_dte / total_weight)
    return 0.40, 30


def simulate_premium_change(
    entry_premium: float,
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    price_change: float,
    days: float = 1.0,
    iv_change: float = 0.0,
) -> float:
    """Simulate option premium change using Greeks (Taylor expansion).

    ΔPremium ≈ δ×ΔS + ½γ×ΔS² + θ×Δt + ν×ΔIV

    Returns new premium (floored at 0.01).
    """
    delta_pnl = delta * price_change
    gamma_pnl = 0.5 * gamma * price_change ** 2
    theta_pnl = theta * days  # theta is already per-day
    vega_pnl = vega * iv_change * 100  # vega is per 1% vol move

    new_premium = entry_premium + delta_pnl + gamma_pnl + theta_pnl + vega_pnl
    return max(new_premium, 0.01)


def build_stock_data_for_day(symbol: str, full_df: pd.DataFrame, day_idx: int,
                              options_cache: dict) -> StockData:
    """Build a StockData object using data up to day_idx (inclusive)."""
    sd = StockData(symbol=symbol)
    df = full_df.iloc[:day_idx + 1].copy()

    if len(df) < 20:
        return sd

    sd.ohlcv = df
    sd.price = float(df["Close"].iloc[-1])
    sd.timestamp = df.index[-1].to_pydatetime() if hasattr(df.index[-1], 'to_pydatetime') else datetime.now()

    close = df["Close"]
    sd.returns_1d = float(close.pct_change(1).iloc[-1]) if len(close) > 1 else 0.0
    sd.returns_5d = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0
    sd.returns_20d = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
    sd.atr_14 = _compute_atr(df) if len(df) >= 14 else 0.0

    daily_ret = close.pct_change()
    rv_5 = daily_ret.rolling(5).std().iloc[-1] * np.sqrt(252) if len(daily_ret) > 5 else 0.0
    rv_10 = daily_ret.rolling(10).std().iloc[-1] * np.sqrt(252) if len(daily_ret) > 10 else 0.0
    rv_20 = daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) if len(daily_ret) > 20 else 0.0

    sd.volume_profile = _compute_volume_profile(df)

    if symbol in options_cache:
        cached = options_cache[symbol]
        sd.gex = cached.get("gex", GEXProfile())
        sd.iv_surface = cached.get("iv_surface", IVSurface())
        sd.options_flow = cached.get("options_flow", OptionsFlow())
        sd.greeks_chain = cached.get("greeks_chain", [])
        sd.market_cap = cached.get("market_cap", 0.0)
        sd.beta = cached.get("beta", 1.0)
        sd.short_ratio = cached.get("short_ratio", 0.0)
        sd.earnings_date = cached.get("earnings_date", None)
        sd.days_to_earnings = cached.get("days_to_earnings", 999)
    else:
        sd.iv_surface = IVSurface()

    sd.iv_surface.rv_5d = float(rv_5) if pd.notna(rv_5) else 0.0
    sd.iv_surface.rv_10d = float(rv_10) if pd.notna(rv_10) else 0.0
    sd.iv_surface.rv_20d = float(rv_20) if pd.notna(rv_20) else 0.0

    if sd.iv_surface.iv_atm > 0 and sd.iv_surface.rv_20d > 0:
        sd.iv_surface.iv_rv_spread = sd.iv_surface.iv_atm - sd.iv_surface.rv_20d

    return sd


def fetch_options_snapshot(symbol: str, price: float, rv_20d: float) -> dict:
    """Fetch current options data for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        gex, ivs, flow, greeks = _compute_full_options(ticker, price, rv_20d)
        mcap, beta, sr, ed_str, dte = _compute_fundamentals_lite(ticker)
        return {
            "gex": gex, "iv_surface": ivs, "options_flow": flow,
            "greeks_chain": greeks, "market_cap": mcap, "beta": beta,
            "short_ratio": sr, "earnings_date": ed_str, "days_to_earnings": dte,
        }
    except Exception as e:
        logger.warning(f"Options fetch failed for {symbol}: {e}")
        return {}


# Track simulated Greeks per position for premium simulation
class SimulatedPosition:
    """Track Greeks state for premium simulation during backtest."""
    def __init__(self, contract: ContractRecommendation, underlying_price: float):
        self.symbol = contract.symbol
        self.strike = contract.strike
        self.expiry = contract.expiry
        self.option_type = contract.option_type
        self.delta = contract.delta
        self.gamma = contract.gamma
        self.theta = contract.theta
        self.vega = contract.vega
        self.iv = contract.iv
        self.dte = contract.dte
        self.current_premium = contract.premium
        self.entry_premium = contract.premium
        self.entry_iv = contract.iv
        self.peak_premium = contract.premium
        self.underlying_price = underlying_price

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

    def update_for_new_day(self, new_underlying_price: float, iv_noise: float = 0.0):
        """Simulate premium change for a new trading day."""
        price_change = new_underlying_price - self.underlying_price

        # For puts, delta is negative — the math handles direction naturally
        new_premium = simulate_premium_change(
            entry_premium=self.current_premium,
            delta=self.delta,
            gamma=self.gamma,
            theta=self.theta,
            vega=self.vega,
            price_change=price_change,
            days=1.0,
            iv_change=iv_noise,
        )

        self.current_premium = new_premium
        self.underlying_price = new_underlying_price
        self.dte = max(self.dte - 1, 0)

        # Update peak
        if new_premium > self.peak_premium:
            self.peak_premium = new_premium

        # Update delta approximation (shift by gamma)
        self.delta = self.delta + self.gamma * price_change
        # Clamp delta
        if self.option_type == "CALL":
            self.delta = max(min(self.delta, 1.0), 0.0)
        else:
            self.delta = max(min(self.delta, 0.0), -1.0)


async def run_backtest():
    # Initialize Telegram notifier
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    notifier = TelegramNotifier(token=token, chat_id=chat_id) if token and chat_id else None

    if notifier:
        print("Telegram notifier enabled")
    else:
        print("WARNING: No Telegram credentials. Signals will not be sent.")

    print("=" * 70)
    print("  OPTIONS TRADING SYSTEM — 30-DAY BACKTEST (CALLS & PUTS)")
    print("=" * 70)
    print(f"\nConfig: ${config.INITIAL_CAPITAL:,.0f} capital | "
          f"Max {config.MAX_HOLDINGS} positions | "
          f"Buy threshold: {config.BUY_THRESHOLD}")
    print(f"Contract selection: Delta {config.TARGET_DELTA_RANGE} | "
          f"DTE {config.PREFERRED_DTE_MIN}-{config.PREFERRED_DTE_MAX}")
    print(f"Exits: {config.PREMIUM_STOP_LOSS_PCT:.0%} SL | "
          f"{config.PREMIUM_TAKE_PROFIT_PCT:.0%} TP | "
          f"{config.PREMIUM_TRAILING_STOP_PCT:.0%} trailing | "
          f"Delta<{config.DELTA_STOP_LOSS} | "
          f"IV crush {config.IV_CRUSH_EXIT_PCT:.0%} | "
          f"Theta>{config.MAX_THETA_DECAY_PCT:.0%}/day")
    print(f"Universe: {len(config.STOCK_UNIVERSE)} stocks")

    # Step 1: Fetch price data
    print(f"\n[1/3] Fetching 3-month daily data for {len(config.STOCK_UNIVERSE)} stocks...")
    price_data = {}
    for i, sym in enumerate(config.STOCK_UNIVERSE):
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="3mo", interval="1d")
            if df is not None and len(df) >= 30:
                price_data[sym] = df
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  ... fetched {i + 1}/{len(config.STOCK_UNIVERSE)}")
    print(f"  Got data for {len(price_data)} stocks")

    # Step 2: Fetch options snapshots
    symbols_with_data = list(price_data.keys())
    print(f"\n[2/3] Fetching options snapshots for {len(symbols_with_data)} stocks...")
    options_cache = {}
    for i, sym in enumerate(symbols_with_data):
        df = price_data[sym]
        price = float(df["Close"].iloc[-1])
        daily_ret = df["Close"].pct_change()
        rv_20d = float(daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252))
        if pd.isna(rv_20d):
            rv_20d = 0.0
        snap = fetch_options_snapshot(sym, price, rv_20d)
        if snap:
            options_cache[sym] = snap
        if (i + 1) % 50 == 0:
            print(f"  ... fetched {i + 1}/{len(symbols_with_data)}")
    print(f"  Got options data for {len(options_cache)} stocks")

    # Step 3: Determine backtest dates
    ref_sym = "AAPL" if "AAPL" in price_data else list(price_data.keys())[0]
    ref_df = price_data[ref_sym]
    all_dates = ref_df.index
    cutoff = datetime.now() - timedelta(days=30)
    bt_dates = [d for d in all_dates if d.to_pydatetime().replace(tzinfo=None) >= cutoff]

    if not bt_dates:
        print("ERROR: No trading dates in the last 30 days!")
        return

    print(f"\n[3/3] Simulating {len(bt_dates)} trading days "
          f"({bt_dates[0].strftime('%Y-%m-%d')} to {bt_dates[-1].strftime('%Y-%m-%d')})")
    print("-" * 70)

    # Portfolio + simulated positions
    portfolio = Portfolio(initial_capital=config.INITIAL_CAPITAL)
    sim_positions: dict[str, SimulatedPosition] = {}  # option_key -> SimulatedPosition
    data_cache = {}
    all_trades = []

    # Send startup
    if notifier:
        await notifier.send(
            f"BACKTEST STARTED (OPTIONS)\n"
            f"Period: {bt_dates[0].strftime('%Y-%m-%d')} to {bt_dates[-1].strftime('%Y-%m-%d')}\n"
            f"Capital: ${config.INITIAL_CAPITAL:,.0f}\n"
            f"Delta: {config.TARGET_DELTA_RANGE} | DTE: {config.PREFERRED_DTE_MIN}-{config.PREFERRED_DTE_MAX}\n"
            f"Universe: {len(price_data)} stocks"
        )

    for day_num, date in enumerate(bt_dates):
        day_buys = []
        day_sells = []

        # --- UPDATE SIMULATED PREMIUMS for held positions ---
        for key in list(sim_positions.keys()):
            sim = sim_positions[key]
            if sim.symbol not in price_data:
                continue
            df = price_data[sim.symbol]
            if date not in df.index:
                continue
            day_idx = df.index.get_loc(date)
            new_price = float(df["Close"].iloc[day_idx])

            # Add small random IV noise to simulate vol movement
            iv_noise = np.random.normal(0, 0.005)  # ~0.5% daily IV noise
            sim.update_for_new_day(new_price, iv_noise)

        # --- SELL CHECK for held positions ---
        for key in list(portfolio.positions.keys()):
            pos = portfolio.positions.get(key)
            if not pos:
                continue
            sim = sim_positions.get(key)
            if not sim:
                continue

            current_premium = sim.current_premium
            sell_reason = None

            # 1. Premium stop loss
            pnl_pct = (current_premium - pos.entry_premium) / pos.entry_premium if pos.entry_premium > 0 else 0
            if pnl_pct <= config.PREMIUM_STOP_LOSS_PCT:
                sell_reason = f"PREMIUM STOP LOSS ({pnl_pct:+.2%})"

            # 2. Premium take profit
            elif pnl_pct >= config.PREMIUM_TAKE_PROFIT_PCT:
                sell_reason = f"PREMIUM TAKE PROFIT ({pnl_pct:+.2%})"

            # 3. Trailing stop
            elif sim.peak_premium > pos.entry_premium:
                trailing_drop = (current_premium - sim.peak_premium) / sim.peak_premium
                if trailing_drop <= -config.PREMIUM_TRAILING_STOP_PCT:
                    sell_reason = f"TRAILING STOP ({trailing_drop:+.2%} from ${sim.peak_premium:.2f})"

            # 4. Delta decay
            if sell_reason is None and abs(sim.delta) < config.DELTA_STOP_LOSS:
                sell_reason = f"DELTA DECAY (Δ{sim.delta:.3f})"

            # 5. IV crush
            if sell_reason is None and sim.entry_iv > 0 and sim.iv > 0:
                iv_change = (sim.iv - sim.entry_iv) / sim.entry_iv
                if iv_change < -config.IV_CRUSH_EXIT_PCT:
                    sell_reason = f"IV CRUSH ({sim.entry_iv:.1%} → {sim.iv:.1%})"

            # 6. Theta bleed
            if sell_reason is None and current_premium > 0:
                daily_theta_pct = abs(sim.theta) / current_premium
                if daily_theta_pct > config.MAX_THETA_DECAY_PCT:
                    sell_reason = f"THETA BLEED ({daily_theta_pct:.1%}/day)"

            # 7. Near expiry
            if sell_reason is None and sim.dte <= config.NEAR_EXPIRY_DTE:
                sell_reason = f"NEAR EXPIRY ({sim.dte} DTE)"

            # 8. Algo sell signal
            if sell_reason is None and pos.symbol in data_cache:
                sd = data_cache[pos.symbol]
                ens_sell = ensemble_sell_score(sd, pos.option_type)
                if ens_sell > 0.55:
                    sell_reason = f"ALGO SELL ({ens_sell:.4f})"

            # 9. EOD swing check
            if sell_reason is None and pos.symbol in data_cache:
                sd = data_cache[pos.symbol]
                swing = ensemble_swing_score(sd)
                if swing < config.SWING_HOLD_THRESHOLD:
                    sell_reason = f"EOD CLOSE (swing: {swing:.4f})"

            if sell_reason:
                trade = portfolio.sell(key, current_premium, reason=sell_reason)
                if trade:
                    day_sells.append(trade)
                    all_trades.append((date, trade))
                    sim_positions.pop(key, None)

                    if notifier:
                        premiums = {k: sp.current_premium for k, sp in sim_positions.items()}
                        status = portfolio.format_status(premiums)
                        await notifier.send_sell_signal(
                            display_name=trade.display_name,
                            premium=current_premium,
                            contracts=trade.contracts,
                            pnl=trade.pnl,
                            pnl_pct=trade.pnl_pct,
                            reason=f"[BT {date.strftime('%m/%d')}] {sell_reason}",
                            greeks={"delta": sim.delta, "iv": sim.iv, "theta": sim.theta, "dte": sim.dte},
                            portfolio_status=status,
                        )

        # --- BUY SCAN ---
        if portfolio.num_holdings < config.MAX_HOLDINGS:
            held_symbols = {pos.symbol for pos in portfolio.positions.values()}
            candidates = [s for s in symbols_with_data if s not in held_symbols]

            scored = []
            for sym in candidates:
                if sym not in price_data:
                    continue
                df = price_data[sym]
                if date not in df.index:
                    continue
                day_idx = df.index.get_loc(date)
                if day_idx < 20:
                    continue

                sd = build_stock_data_for_day(sym, df, day_idx, options_cache)
                if sd.price <= 0:
                    continue

                direction, conviction, algo_signals = ensemble_signal(sd)
                if direction in ("CALL", "PUT") and conviction >= config.BUY_THRESHOLD:
                    scored.append((sym, sd, direction, conviction, algo_signals))
                    data_cache[sym] = sd

            scored.sort(key=lambda x: x[3], reverse=True)
            slots = config.MAX_HOLDINGS - portfolio.num_holdings

            for sym, sd, direction, conviction, algo_signals in scored[:slots]:
                max_budget = portfolio.cash * config.POSITION_SIZE_PCT
                if max_budget < 50:
                    break

                pref_delta, pref_dte = weighted_preferred_params(algo_signals)

                # Select optimal contract
                contract = select_optimal_contract(
                    symbol=sym,
                    greeks_chain=sd.greeks_chain,
                    direction=direction,
                    spot_price=sd.price,
                    preferred_delta=pref_delta,
                    preferred_dte=pref_dte,
                )
                if not contract:
                    continue

                # Determine contracts
                cost_per = contract.premium * config.CONTRACT_MULTIPLIER
                if cost_per <= 0:
                    continue
                num_contracts = min(int(max_budget / cost_per), 5)
                if num_contracts <= 0:
                    continue

                trade = portfolio.buy(
                    symbol=sym,
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
                    # Create simulated position for premium tracking
                    sim_positions[trade.option_key] = SimulatedPosition(contract, sd.price)
                    data_cache[sym] = sd
                    day_buys.append(trade)
                    all_trades.append((date, trade))

                    if notifier:
                        scores_summary = {
                            name: f"{sig.direction} ({sig.conviction:.4f})"
                            for name, sig in algo_signals.items()
                        }
                        premiums = {k: sp.current_premium for k, sp in sim_positions.items()}
                        status = portfolio.format_status(premiums)
                        await notifier.send_buy_signal(
                            display_name=trade.display_name,
                            premium=contract.premium,
                            contracts=num_contracts,
                            scores=scores_summary,
                            ensemble_direction=direction,
                            ensemble_conviction=conviction,
                            greeks={
                                "delta": contract.delta, "gamma": contract.gamma,
                                "theta": contract.theta, "vega": contract.vega,
                                "iv": contract.iv, "dte": contract.dte,
                            },
                            portfolio_status=f"[BT {date.strftime('%m/%d')}]\n{status}",
                        )

        # Daily summary
        if day_buys or day_sells:
            date_str = date.strftime("%Y-%m-%d")
            premiums = {k: sp.current_premium for k, sp in sim_positions.items()}
            equity = portfolio.total_equity(premiums)
            pnl = portfolio.total_pnl(premiums)
            pnl_pct = portfolio.total_pnl_pct(premiums)

            buys_str = ", ".join(
                f"{t.display_name}@${t.premium:.2f}" for t in day_buys
            )
            sells_str = ", ".join(
                f"{t.display_name}@${t.premium:.2f} ({t.pnl:+.2f})" for t in day_sells
            )
            print(f"  {date_str} | Equity: ${equity:,.2f} ({pnl:+.2f}, {pnl_pct:+.2%}) | "
                  f"Holdings: {portfolio.num_holdings}")
            if buys_str:
                print(f"    BUY:  {buys_str}")
            if sells_str:
                print(f"    SELL: {sells_str}")

    # --- FINAL RESULTS ---
    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS — OPTIONS TRADING (CALLS & PUTS)")
    print("=" * 70)

    final_premiums = {k: sp.current_premium for k, sp in sim_positions.items()}
    final_equity = portfolio.total_equity(final_premiums)
    total_pnl = portfolio.total_pnl(final_premiums)
    total_pnl_pct = portfolio.total_pnl_pct(final_premiums)

    buy_trades = [t for _, t in all_trades if t.side == "BUY"]
    sell_trades = [t for _, t in all_trades if t.side == "SELL"]
    call_buys = [t for t in buy_trades if t.option_type == "CALL"]
    put_buys = [t for t in buy_trades if t.option_type == "PUT"]
    winning = [t for t in sell_trades if t.pnl > 0]
    losing = [t for t in sell_trades if t.pnl <= 0]

    print(f"\n  Period: {bt_dates[0].strftime('%Y-%m-%d')} to {bt_dates[-1].strftime('%Y-%m-%d')} "
          f"({len(bt_dates)} trading days)")
    print(f"\n  Initial Capital:  ${config.INITIAL_CAPITAL:>12,.2f}")
    print(f"  Final Equity:     ${final_equity:>12,.2f}")
    print(f"  Total P&L:        ${total_pnl:>12,.2f}  ({total_pnl_pct:+.2%})")
    print(f"  Realized P&L:     ${portfolio.realized_pnl:>12,.2f}")
    print(f"  Unrealized P&L:   ${total_pnl - portfolio.realized_pnl:>12,.2f}")

    print(f"\n  Total Buys:       {len(buy_trades):>6}  (CALL: {len(call_buys)}, PUT: {len(put_buys)})")
    print(f"  Total Sells:      {len(sell_trades):>6}")
    print(f"  Winning Trades:   {len(winning):>6}")
    print(f"  Losing Trades:    {len(losing):>6}")

    if sell_trades:
        win_rate = len(winning) / len(sell_trades)
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
        best = max(sell_trades, key=lambda t: t.pnl)
        worst = min(sell_trades, key=lambda t: t.pnl)

        print(f"  Win Rate:         {win_rate:>6.1%}")
        print(f"  Avg Win:          ${avg_win:>12,.2f}")
        print(f"  Avg Loss:         ${avg_loss:>12,.2f}")
        if losing and sum(t.pnl for t in losing) != 0:
            print(f"  Profit Factor:    {abs(sum(t.pnl for t in winning) / sum(t.pnl for t in losing)):>10.2f}x")
        print(f"  Best Trade:       {best.display_name} ${best.pnl:+,.2f} ({best.pnl_pct:+.2%})")
        print(f"  Worst Trade:      {worst.display_name} ${worst.pnl:+,.2f} ({worst.pnl_pct:+.2%})")

    # Exit reason breakdown
    if sell_trades:
        print(f"\n  ── Exit Reason Breakdown ──")
        reasons = {}
        for t in sell_trades:
            # Extract reason category
            reason_cat = t.reason.split("(")[0].strip().split("[")[-1].strip()
            if reason_cat.startswith("BT "):
                reason_cat = reason_cat.split("] ")[-1] if "]" in reason_cat else reason_cat
            reasons[reason_cat] = reasons.get(reason_cat, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # Open positions
    if portfolio.positions:
        print(f"\n  ── Open Positions ──")
        for key, pos in portfolio.positions.items():
            sim = sim_positions.get(key)
            curr_prem = sim.current_premium if sim else pos.entry_premium
            upnl = pos.unrealized_pnl(curr_prem)
            upnl_pct = pos.unrealized_pnl_pct(curr_prem)
            delta_str = f"Δ{sim.delta:.2f}" if sim else ""
            dte_str = f"{sim.dte}DTE" if sim else ""
            print(f"    {pos.display_name}: {pos.contracts}x @ ${pos.entry_premium:.2f} → ${curr_prem:.2f} "
                  f"({upnl:+.2f}, {upnl_pct:+.2%}) {delta_str} {dte_str}")

    # Trade log
    if all_trades:
        print(f"\n  ── Trade Log (last 25) ──")
        for date, t in all_trades[-25:]:
            date_str = date.strftime("%m/%d")
            if t.side == "BUY":
                print(f"    {date_str} BUY  {t.display_name:>30} {t.contracts:>2}x @ ${t.premium:>6.2f} "
                      f"(Δ{t.entry_delta:.2f} IV:{t.entry_iv:.1%}) | {t.reason}")
            else:
                print(f"    {date_str} SELL {t.display_name:>30} {t.contracts:>2}x @ ${t.premium:>6.2f} "
                      f"| P&L: ${t.pnl:>+8.2f} ({t.pnl_pct:>+6.2%}) | {t.reason}")

    print("\n" + "=" * 70)
    print("  Note: Options data is a current snapshot (not historical).")
    print("  Premium changes simulated via Greeks (δ×ΔS + ½γ×ΔS² + θ×Δt + ν×ΔIV)")
    print("=" * 70)

    # Send final summary to Telegram
    if notifier:
        summary_lines = [
            "BACKTEST COMPLETE (OPTIONS)",
            f"Period: {bt_dates[0].strftime('%Y-%m-%d')} to {bt_dates[-1].strftime('%Y-%m-%d')}",
            "",
            f"Initial: ${config.INITIAL_CAPITAL:,.2f}",
            f"Final:   ${final_equity:,.2f}",
            f"P&L:     ${total_pnl:+,.2f} ({total_pnl_pct:+.2%})",
            f"Realized: ${portfolio.realized_pnl:+,.2f}",
            "",
            f"Buys: {len(buy_trades)} (CALL:{len(call_buys)} PUT:{len(put_buys)})",
            f"Sells: {len(sell_trades)} | Wins: {len(winning)} | Losses: {len(losing)}",
        ]
        if sell_trades:
            win_rate = len(winning) / len(sell_trades)
            summary_lines.append(f"Win Rate: {win_rate:.1%}")
            best = max(sell_trades, key=lambda t: t.pnl)
            worst = min(sell_trades, key=lambda t: t.pnl)
            summary_lines.append(f"Best:  {best.display_name} ${best.pnl:+,.2f}")
            summary_lines.append(f"Worst: {worst.display_name} ${worst.pnl:+,.2f}")
        if portfolio.positions:
            summary_lines.append("")
            summary_lines.append("Open positions:")
            for key, pos in portfolio.positions.items():
                sim = sim_positions.get(key)
                curr = sim.current_premium if sim else pos.entry_premium
                upnl_pct = pos.unrealized_pnl_pct(curr)
                summary_lines.append(f"  {pos.display_name} {pos.contracts}x ({upnl_pct:+.2%})")
        await notifier.send("\n".join(summary_lines))


if __name__ == "__main__":
    asyncio.run(run_backtest())
