"""
Algorithm 1: Volatility Arbitrage (The Jane Street Edge)

Exploits the Variance Risk Premium: IV systematically overprices realized vol.
Now outputs directional CALL/PUT signals based on vol regime analysis.

BUY CALL when:
  - IV cheap relative to RV → vol expansion expected → stock moves up
  - RV compressing (coiled spring) → breakout imminent
  - Term structure in contango → calm, supportive of longs
  - Elevated put skew → fear = contrarian bullish

BUY PUT when:
  - IV extremely expensive → mean reversion in vol = price decline
  - RV expanding rapidly → exhaustion, reversal imminent
  - Term structure in steep backwardation → near-term fear/event
  - Skew collapsing → complacency, no protection = vulnerable to selloff
  - GARCH forecast << IV → vol will contract, overpriced fear

EXIT (high sell score) when:
  - Signal reversal (bullish→bearish or vice versa)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from algorithms.signal import OptionSignal
from data_fetcher import StockData


class VolArbAlgorithm:
    name = "vol_arb"

    def _rv_cone_percentile(self, data: StockData) -> float:
        """Where current RV sits vs its own 60-day rolling distribution."""
        if data.ohlcv is None or len(data.ohlcv) < 60:
            return 0.5
        daily_ret = data.ohlcv["Close"].pct_change()
        rv_series = daily_ret.rolling(20).std() * np.sqrt(252)
        rv_series = rv_series.dropna()
        if len(rv_series) < 10:
            return 0.5
        current_rv = float(rv_series.iloc[-1])
        percentile = float((rv_series < current_rv).sum() / len(rv_series))
        return percentile

    def _vol_regime(self, data: StockData) -> str:
        """Detect vol regime: compressing, expanding, or stable."""
        ivs = data.iv_surface
        if ivs.rv_5d == 0 or ivs.rv_20d == 0:
            return "stable"
        rv_ratio = ivs.rv_5d / ivs.rv_20d
        if rv_ratio > 1.3:
            return "expanding"
        elif rv_ratio < 0.7:
            return "compressing"
        return "stable"

    def _garch_simple_forecast(self, data: StockData) -> float:
        """EWMA vol forecast (GARCH(1,1) proxy)."""
        if data.ohlcv is None or len(data.ohlcv) < 30:
            return 0.0
        daily_ret = data.ohlcv["Close"].pct_change().dropna()
        lam = 0.94
        var_t = daily_ret.var()
        for r in daily_ret.iloc[-20:]:
            var_t = lam * var_t + (1 - lam) * r ** 2
        return float(np.sqrt(var_t * 252))

    def signal(self, data: StockData) -> OptionSignal:
        """Produce directional signal based on vol regime analysis."""
        ivs = data.iv_surface
        bullish_score = 0.0
        bearish_score = 0.0
        component_scores = {}

        # --- BULLISH COMPONENTS (favor CALL) ---

        # 1. IV-RV spread: IV cheap relative to RV (18%)
        #    VRP calibration: IV normally trades 15-25% above RV (variance risk premium).
        #    Only flag as "cheap" when spread < 0.25 (i.e., IV barely above RV).
        iv_rv_bull = 0.0
        if ivs.iv_atm > 0 and ivs.rv_20d > 0:
            spread_ratio = ivs.iv_rv_spread / ivs.rv_20d
            if spread_ratio < 0.25:
                iv_rv_bull = min((0.25 - spread_ratio) / 0.50, 1.0)
        component_scores["iv_rv_cheap"] = iv_rv_bull
        bullish_score += 0.18 * iv_rv_bull

        # 2. GARCH forecast > IV: vol will expand (12%)
        garch_bull = 0.0
        garch_vol = self._garch_simple_forecast(data)
        if garch_vol > 0 and ivs.iv_atm > 0:
            vol_discount = (garch_vol - ivs.iv_atm) / ivs.iv_atm
            if vol_discount > 0:
                garch_bull = min(vol_discount / 0.15, 1.0)
        component_scores["garch_underpriced"] = garch_bull
        bullish_score += 0.12 * garch_bull

        # 3. RV cone low percentile: room to run up (8%)
        rv_pct = self._rv_cone_percentile(data)
        rv_bull = max((0.5 - rv_pct) / 0.5, 0.0) if rv_pct < 0.5 else 0.0
        component_scores["rv_cone_low"] = rv_bull
        bullish_score += 0.08 * rv_bull

        # 4. Vol compressing: coiled spring → breakout up (12%)
        regime = self._vol_regime(data)
        compress_bull = 1.0 if regime == "compressing" else 0.0
        component_scores["vol_compressing"] = compress_bull
        bullish_score += 0.12 * compress_bull

        # 5. Term structure contango: calm market (8%)
        contango_bull = 0.0
        if ivs.iv_term_slope < 0.01:
            contango_bull = min((0.01 - ivs.iv_term_slope) / 0.06, 1.0)
        component_scores["term_contango"] = contango_bull
        bullish_score += 0.08 * contango_bull

        # 6. Elevated put skew: fear = contrarian buy (8%)
        skew_bull = 0.0
        if ivs.skew_25d > 0.01:
            skew_bull = min(ivs.skew_25d / 0.06, 1.0)
        component_scores["skew_elevated"] = skew_bull
        bullish_score += 0.08 * skew_bull

        # 7. Price trend: price above 20-day SMA = bullish momentum (12%)
        #    Every real desk uses trend as a baseline directional signal.
        trend_bull = 0.0
        if data.ohlcv is not None and len(data.ohlcv) >= 20:
            sma_20 = float(data.ohlcv["Close"].rolling(20).mean().iloc[-1])
            if sma_20 > 0 and data.price > sma_20:
                trend_bull = min((data.price - sma_20) / (sma_20 * 0.03), 1.0)
        component_scores["trend_above_sma"] = trend_bull
        bullish_score += 0.12 * trend_bull

        # --- BEARISH COMPONENTS (favor PUT) ---

        # 8. IV spike: IV >> RV, genuinely expensive premium (18%)
        #    VRP calibration: only flag when spread > 0.35 (well above normal VRP).
        iv_spike_bear = 0.0
        if ivs.iv_atm > 0 and ivs.rv_20d > 0:
            spread = ivs.iv_rv_spread / ivs.rv_20d
            if spread > 0.35:
                iv_spike_bear = min((spread - 0.10) / 0.50, 1.0)
        component_scores["iv_spike"] = iv_spike_bear
        bearish_score += 0.18 * iv_spike_bear

        # 9. RV expanding: exhaustion → reversal (12%)
        expand_bear = 1.0 if regime == "expanding" else 0.0
        component_scores["rv_expanding"] = expand_bear
        bearish_score += 0.12 * expand_bear

        # 10. Term structure backwardation: fear/event premium (12%)
        backw_bear = 0.0
        if ivs.iv_term_slope > 0.03:
            backw_bear = min(ivs.iv_term_slope / 0.08, 1.0)
        component_scores["term_backwardation"] = backw_bear
        bearish_score += 0.12 * backw_bear

        # 11. Skew collapse: puts getting cheap → complacency (8%)
        skew_bear = 0.0
        if ivs.skew_25d < -0.01:
            skew_bear = min(abs(ivs.skew_25d) / 0.05, 1.0)
        component_scores["skew_collapse"] = skew_bear
        bearish_score += 0.08 * skew_bear

        # 12. GARCH forecast << IV: vol overpriced → crush coming (8%)
        #     VRP calibration: require larger premium (>0.25) before flagging.
        garch_bear = 0.0
        if garch_vol > 0 and ivs.iv_atm > 0:
            vol_premium = (ivs.iv_atm - garch_vol) / ivs.iv_atm
            if vol_premium > 0.25:
                garch_bear = min(vol_premium / 0.40, 1.0)
        component_scores["garch_overpriced"] = garch_bear
        bearish_score += 0.08 * garch_bear

        # 13. RV cone high percentile: extended move, exhaustion (8%)
        rv_bear = max((rv_pct - 0.7) / 0.3, 0.0) if rv_pct > 0.7 else 0.0
        component_scores["rv_cone_high"] = rv_bear
        bearish_score += 0.08 * rv_bear

        # 14. Price trend: price below 20-day SMA = bearish momentum (12%)
        trend_bear = 0.0
        if data.ohlcv is not None and len(data.ohlcv) >= 20:
            sma_20 = float(data.ohlcv["Close"].rolling(20).mean().iloc[-1])
            if sma_20 > 0 and data.price < sma_20:
                trend_bear = min((sma_20 - data.price) / (sma_20 * 0.03), 1.0)
        component_scores["trend_below_sma"] = trend_bear
        bearish_score += 0.12 * trend_bear

        # --- Conviction calibration ---
        # Raw component sums produce scores in 0.10-0.30 range in typical conditions.
        # Scale by 2.0x so moderate conviction maps to the ensemble threshold range.
        bullish_score = min(bullish_score * 2.0, 1.0)
        bearish_score = min(bearish_score * 2.0, 1.0)
        component_scores["bullish_total"] = round(bullish_score, 4)
        component_scores["bearish_total"] = round(bearish_score, 4)

        # Net direction: need meaningful edge over the other side
        # Threshold 0.05: with conviction scaling, this still requires genuine
        # directional agreement across components — prevents NEUTRAL over-classification.
        net = bullish_score - bearish_score
        if net > 0.05:
            direction = "CALL"
            conviction = bullish_score
            # Vol arb prefers slightly higher delta (more directional)
            # and moderate DTE (enough theta runway but not too far)
            preferred_delta = 0.40 + 0.10 * min(bullish_score, 1.0)  # 0.40-0.50
            preferred_dte = 30 if regime != "compressing" else 21  # Shorter for coiled spring
        elif net < -0.05:
            direction = "PUT"
            conviction = bearish_score
            # For puts, prefer slightly lower delta (more OTM for leverage)
            preferred_delta = 0.35 + 0.10 * min(bearish_score, 1.0)  # 0.35-0.45
            preferred_dte = 30 if regime != "expanding" else 21
        else:
            direction = "NEUTRAL"
            conviction = 0.0
            preferred_delta = 0.40
            preferred_dte = 30

        return OptionSignal(
            direction=direction,
            conviction=round(conviction, 4),
            scores=component_scores,
            preferred_delta=round(preferred_delta, 2),
            preferred_dte=preferred_dte,
        )

    def sell_score(self, data: StockData, position_direction: str) -> float:
        """Exit score: how strongly should we close this position?
        High score = strong reason to exit.
        """
        sig = self.signal(data)
        # If holding a CALL but signal says PUT → strong exit
        if position_direction == "CALL" and sig.direction == "PUT":
            return min(0.50 + sig.conviction * 0.50, 1.0)
        # If holding a PUT but signal says CALL → strong exit
        if position_direction == "PUT" and sig.direction == "CALL":
            return min(0.50 + sig.conviction * 0.50, 1.0)
        # If signal is NEUTRAL → mild exit pressure
        if sig.direction == "NEUTRAL":
            return 0.30
        # Signal agrees with position → no exit pressure
        return 0.0

    def swing_score(self, data: StockData) -> float:
        """Overnight hold viability."""
        ivs = data.iv_surface
        score = 0.0
        if ivs.iv_atm > 0 and ivs.rv_20d > 0:
            if ivs.iv_rv_spread < 0:
                score += 0.3
        if self._vol_regime(data) == "compressing":
            score += 0.3
        if ivs.iv_term_slope < -0.02:
            score += 0.2
        if data.days_to_earnings > 7:
            score += 0.2
        return round(min(score, 1.0), 4)
