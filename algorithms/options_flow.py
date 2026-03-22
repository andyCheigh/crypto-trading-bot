"""
Algorithm 3: Options Order Flow & Smart Money Sentiment (The Jump Edge)

Jump Trading's edge comes from reading order flow before the move happens.
Options flow reveals institutional intent because:
1. Options provide leverage → big directional bets show up here first
2. Unusual activity (volume >> open interest) signals new positions
3. Premium flow ($ into calls vs puts) reveals conviction level
4. Skew changes reflect informed views on tail risk

Key signals:
- Unusual options activity (UOA): vol > 2x OI on a strike = new position
- Net premium flow: $ into calls vs puts (bullish vs bearish conviction)
- Put/Call ratios (contrarian at extremes, confirming in middle)
- Max pain convergence: price pulled toward max pain near expiry
- Earnings proximity: avoid overnight holds within 3 days of earnings
- Smart money flow: large block trades on OTM options
"""

from __future__ import annotations

import numpy as np

from data_fetcher import StockData


class OptionsFlowAlgorithm:
    name = "options_flow"

    def _uoa_signal(self, data: StockData) -> float:
        """Unusual Options Activity: net direction of smart money bets.
        More unusual calls than puts = bullish institutional positioning.
        """
        uc = data.options_flow.unusual_calls
        up = data.options_flow.unusual_puts
        total = uc + up
        if total == 0:
            return 0.0
        # Net ratio: positive = more unusual calls (bullish)
        return float((uc - up) / total)

    def _premium_flow_signal(self, data: StockData) -> float:
        """Net premium flow: are institutions spending more on calls or puts?
        Normalized to [-1, 1].
        """
        net = data.options_flow.net_premium_flow
        total = data.options_flow.total_call_premium + data.options_flow.total_put_premium
        if total == 0:
            return 0.0
        return float(np.clip(net / total, -1.0, 1.0))

    def _pcr_contrarian(self, data: StockData) -> float:
        """Put/Call ratio as contrarian indicator.
        Very high PCR (>1.3) = extreme fear → contrarian buy.
        Very low PCR (<0.5) = complacency → contrarian sell.
        Middle range is noise.
        """
        pcr = data.options_flow.put_call_vol_ratio
        if pcr > 1.3:
            return min((pcr - 1.0) / 0.8, 1.0)  # Positive = contrarian buy
        elif pcr < 0.5:
            return -min((0.7 - pcr) / 0.4, 1.0)  # Negative = contrarian sell
        return 0.0

    def _max_pain_signal(self, data: StockData) -> float:
        """Max pain magnet effect.
        Price below max pain → expect upward pull.
        Price above max pain → expect downward pull.
        """
        mp = data.options_flow.max_pain
        if mp <= 0 or data.price <= 0:
            return 0.0
        dev = (mp - data.price) / data.price
        return float(np.clip(dev / 0.03, -1.0, 1.0))

    def _oi_pcr_trend(self, data: StockData) -> float:
        """OI-based put/call ratio: more structural than volume-based.
        High OI PCR = heavy put hedging = support underneath.
        """
        oi_pcr = data.options_flow.put_call_oi_ratio
        if oi_pcr > 1.0:
            # High put OI = institutional hedging = they own the stock (bullish)
            return min((oi_pcr - 1.0) / 0.5, 1.0)
        return 0.0

    def buy_score(self, data: StockData) -> float:
        score = 0.0

        # 1. Unusual call activity > put activity (25%)
        uoa = self._uoa_signal(data)
        if uoa > 0:
            score += 0.25 * uoa

        # 2. Net premium flow bullish (20%)
        pf = self._premium_flow_signal(data)
        if pf > 0:
            score += 0.20 * pf

        # 3. Put/Call ratio contrarian buy (extreme fear) (20%)
        pcr_sig = self._pcr_contrarian(data)
        if pcr_sig > 0:
            score += 0.20 * pcr_sig

        # 4. Max pain above current price (pull up) (15%)
        mp_sig = self._max_pain_signal(data)
        if mp_sig > 0:
            score += 0.15 * mp_sig

        # 5. High put OI ratio (institutional hedging = they own it) (10%)
        oi_sig = self._oi_pcr_trend(data)
        if oi_sig > 0:
            score += 0.10 * oi_sig

        # 6. Avoid buying near earnings (penalty) (10%)
        if data.days_to_earnings <= 3:
            score *= 0.5  # Heavy penalty for earnings proximity

        return round(min(score, 1.0), 4)

    def sell_score(self, data: StockData, entry_price: float) -> float:
        score = 0.0

        # 1. Unusual put activity dominant (25%)
        uoa = self._uoa_signal(data)
        if uoa < 0:
            score += 0.25 * abs(uoa)

        # 2. Net premium flow bearish (20%)
        pf = self._premium_flow_signal(data)
        if pf < 0:
            score += 0.20 * abs(pf)

        # 3. PCR low = complacency (contrarian sell) (20%)
        pcr_sig = self._pcr_contrarian(data)
        if pcr_sig < 0:
            score += 0.20 * abs(pcr_sig)

        # 4. Max pain below price (pull down) (15%)
        mp_sig = self._max_pain_signal(data)
        if mp_sig < 0:
            score += 0.15 * abs(mp_sig)

        # 5. Approaching earnings → close position (20%)
        if data.days_to_earnings <= 2:
            score += 0.20

        return round(min(score, 1.0), 4)

    def swing_score(self, data: StockData) -> float:
        """Overnight hold viability based on flow signals."""
        score = 0.0

        # Bullish premium flow = institutions expect upside
        pf = self._premium_flow_signal(data)
        if pf > 0.2:
            score += 0.30

        # Unusual call activity = smart money bullish
        uoa = self._uoa_signal(data)
        if uoa > 0.3:
            score += 0.25

        # High put OI = institutional hedging = supportive
        oi_sig = self._oi_pcr_trend(data)
        if oi_sig > 0:
            score += 0.20 * oi_sig

        # NOT near earnings (critical for overnight)
        if data.days_to_earnings > 7:
            score += 0.25
        elif data.days_to_earnings <= 3:
            score = 0.0  # Never hold overnight near earnings

        return round(min(score, 1.0), 4)
