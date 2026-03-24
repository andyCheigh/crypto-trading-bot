"""
Algorithm 3: Options Order Flow & Smart Money Sentiment (The Jump Edge)

Reads institutional order flow to detect smart money positioning.
Options flow reveals intent because of leverage and anonymity.

BUY CALL when:
  - Unusual call activity > puts: smart money bullish
  - Net premium flow into calls: institutions spending on upside
  - Extreme high PCR (>1.3): contrarian buy, fear overdone
  - Max pain above price: gravitational pull upward near expiry
  - High put OI ratio: heavy institutional hedging = they own it

BUY PUT when:
  - Unusual put activity > calls: smart money bearish
  - Net premium flow into puts: institutions spending on downside
  - Extreme low PCR (<0.5): complacency, no protection
  - Max pain below price: gravitational pull downward
  - Heavy call selling (premium flowing out of calls)

EARNINGS GUARD:
  - Within 3 days: halve conviction (binary event risk)
  - Never hold overnight within 3 days of earnings
"""

from __future__ import annotations

import numpy as np

from algorithms.signal import OptionSignal
from data_fetcher import StockData


class OptionsFlowAlgorithm:
    name = "options_flow"

    def _uoa_signal(self, data: StockData) -> float:
        """Unusual Options Activity: net direction of smart money.
        Positive = more unusual calls (bullish), negative = more unusual puts (bearish).
        """
        uc = data.options_flow.unusual_calls
        up = data.options_flow.unusual_puts
        total = uc + up
        if total == 0:
            return 0.0
        return float((uc - up) / total)

    def _premium_flow_signal(self, data: StockData) -> float:
        """Net premium flow. Positive = call-heavy (bullish), negative = put-heavy."""
        net = data.options_flow.net_premium_flow
        total = data.options_flow.total_call_premium + data.options_flow.total_put_premium
        if total == 0:
            return 0.0
        return float(np.clip(net / total, -1.0, 1.0))

    def _pcr_contrarian(self, data: StockData) -> float:
        """Put/Call ratio contrarian: high PCR = fear (buy), low PCR = complacency (sell)."""
        pcr = data.options_flow.put_call_vol_ratio
        if pcr > 1.3:
            return min((pcr - 1.0) / 0.8, 1.0)
        elif pcr < 0.5:
            return -min((0.7 - pcr) / 0.4, 1.0)
        return 0.0

    def _max_pain_signal(self, data: StockData) -> float:
        """Max pain magnet. Positive = pain above (pull up), negative = pain below (pull down)."""
        mp = data.options_flow.max_pain
        if mp <= 0 or data.price <= 0:
            return 0.0
        dev = (mp - data.price) / data.price
        return float(np.clip(dev / 0.03, -1.0, 1.0))

    def _oi_pcr_trend(self, data: StockData) -> float:
        """OI-based PCR: high = institutional hedging = support."""
        oi_pcr = data.options_flow.put_call_oi_ratio
        if oi_pcr > 1.0:
            return min((oi_pcr - 1.0) / 0.5, 1.0)
        elif oi_pcr < 0.6:
            # Very low put OI = institutions not hedging = vulnerable
            return -min((0.6 - oi_pcr) / 0.3, 1.0)
        return 0.0

    def signal(self, data: StockData) -> OptionSignal:
        """Produce CALL/PUT signal based on options order flow analysis."""
        bullish_score = 0.0
        bearish_score = 0.0
        component_scores = {}

        uoa = self._uoa_signal(data)
        pf = self._premium_flow_signal(data)
        pcr_sig = self._pcr_contrarian(data)
        mp_sig = self._max_pain_signal(data)
        oi_sig = self._oi_pcr_trend(data)

        # --- BULLISH COMPONENTS (favor CALL) ---

        # 1. Unusual call activity dominant (20%)
        uoa_bull = max(uoa, 0.0)
        component_scores["uoa_calls"] = uoa_bull
        bullish_score += 0.20 * uoa_bull

        # 2. Net premium flow bullish (15%)
        pf_bull = max(pf, 0.0)
        component_scores["premium_bullish"] = pf_bull
        bullish_score += 0.15 * pf_bull

        # 3. PCR contrarian buy at extremes (12%)
        pcr_bull = max(pcr_sig, 0.0)
        component_scores["pcr_fear_buy"] = pcr_bull
        bullish_score += 0.12 * pcr_bull

        # 4. Max pain above (pull up) (10%)
        mp_bull = max(mp_sig, 0.0)
        component_scores["max_pain_above"] = mp_bull
        bullish_score += 0.10 * mp_bull

        # 5. High put OI = institutional hedging (10%)
        oi_bull = max(oi_sig, 0.0)
        component_scores["put_oi_support"] = oi_bull
        bullish_score += 0.10 * oi_bull

        # --- BEARISH COMPONENTS (favor PUT) ---

        # 6. Unusual put activity dominant (20%)
        uoa_bear = max(-uoa, 0.0)
        component_scores["uoa_puts"] = uoa_bear
        bearish_score += 0.20 * uoa_bear

        # 7. Net premium flow bearish (15%)
        pf_bear = max(-pf, 0.0)
        component_scores["premium_bearish"] = pf_bear
        bearish_score += 0.15 * pf_bear

        # 8. PCR low = complacency = contrarian sell (12%)
        pcr_bear = max(-pcr_sig, 0.0)
        component_scores["pcr_complacency"] = pcr_bear
        bearish_score += 0.12 * pcr_bear

        # 9. Max pain below (pull down) (10%)
        mp_bear = max(-mp_sig, 0.0)
        component_scores["max_pain_below"] = mp_bear
        bearish_score += 0.10 * mp_bear

        # 10. Low put OI = no hedge = vulnerable (10%)
        oi_bear = max(-oi_sig, 0.0)
        component_scores["no_put_hedge"] = oi_bear
        bearish_score += 0.10 * oi_bear

        # --- Earnings adjustment ---
        earnings_multiplier = 1.0
        if data.days_to_earnings <= 3:
            earnings_multiplier = 0.50  # Halve conviction near earnings
        component_scores["earnings_multiplier"] = earnings_multiplier

        bullish_score = min(bullish_score * earnings_multiplier, 1.0)
        bearish_score = min(bearish_score * earnings_multiplier, 1.0)
        component_scores["bullish_total"] = round(bullish_score, 4)
        component_scores["bearish_total"] = round(bearish_score, 4)

        # --- Direction ---
        net = bullish_score - bearish_score
        if net > 0.06:
            direction = "CALL"
            conviction = bullish_score
            # Flow signals are near-term: moderate DTE
            preferred_delta = 0.40
            preferred_dte = 28
        elif net < -0.06:
            direction = "PUT"
            conviction = bearish_score
            preferred_delta = 0.38
            preferred_dte = 28
        else:
            direction = "NEUTRAL"
            conviction = 0.0
            preferred_delta = 0.40
            preferred_dte = 30

        return OptionSignal(
            direction=direction,
            conviction=round(conviction, 4),
            scores=component_scores,
            preferred_delta=preferred_delta,
            preferred_dte=preferred_dte,
        )

    def sell_score(self, data: StockData, position_direction: str) -> float:
        """Exit score based on flow reversal."""
        sig = self.signal(data)
        if position_direction == "CALL" and sig.direction == "PUT":
            return min(0.50 + sig.conviction * 0.50, 1.0)
        if position_direction == "PUT" and sig.direction == "CALL":
            return min(0.50 + sig.conviction * 0.50, 1.0)
        if sig.direction == "NEUTRAL":
            return 0.25
        # Approaching earnings → close
        if data.days_to_earnings <= 2:
            return 0.60
        return 0.0

    def swing_score(self, data: StockData) -> float:
        """Overnight hold viability based on flow signals."""
        score = 0.0
        pf = self._premium_flow_signal(data)
        if abs(pf) > 0.2:
            score += 0.30
        uoa = self._uoa_signal(data)
        if abs(uoa) > 0.3:
            score += 0.25
        oi_sig = self._oi_pcr_trend(data)
        if abs(oi_sig) > 0:
            score += 0.20 * abs(oi_sig)
        if data.days_to_earnings > 7:
            score += 0.25
        elif data.days_to_earnings <= 3:
            score = 0.0  # Never hold overnight near earnings
        return round(min(score, 1.0), 4)
