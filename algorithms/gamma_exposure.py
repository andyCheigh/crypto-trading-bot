"""
Algorithm 2: Gamma Exposure & Dealer Flow (The HRT/Citadel Edge)

This is the most powerful options-derived signal for short-term equity
direction. Market makers (dealers) must delta-hedge their options books.
Their hedging flow creates predictable price dynamics.

Key concept: Dealers are structurally short calls and long puts.
- Short calls → short gamma on calls → when stock rises, dealer must BUY
  (amplifying the move). When stock falls, dealer must SELL (amplifying down).
- Long puts → long gamma on puts → when stock falls, dealer buys (cushioning).

Net Gamma Exposure (GEX) = aggregate dealer gamma across all strikes.

Positive GEX (dealer long gamma overall):
  → Dealer sells rallies, buys dips → STABILIZING → low vol, mean reversion
  → Price gravitates toward high-gamma strike clusters → "pinning"

Negative GEX (dealer short gamma overall):
  → Dealer amplifies moves in both directions → DESTABILIZING → high vol
  → Breakouts are real, trends persist

Additional signals:
  - Gamma flip level: price where GEX changes sign (key level)
  - Call wall: highest call gamma strike → resistance / magnet
  - Put wall: highest put gamma strike → support / magnet
  - Vanna flow: as vol drops, dealer hedging adjusts → bullish above gamma flip
  - Charm flow: delta decay over time → predictable hedging at EOD
"""

from __future__ import annotations

import numpy as np

from data_fetcher import StockData


class GammaExposureAlgorithm:
    name = "gamma_exposure"

    def _price_vs_gamma_flip(self, data: StockData) -> float:
        """Signed distance from current price to gamma flip level.
        Positive = above flip (positive GEX territory).
        Negative = below flip (negative GEX territory).
        """
        flip = data.gex.gamma_flip_level
        if flip <= 0 or data.price <= 0:
            return 0.0
        return (data.price - flip) / data.price

    def _pin_risk_score(self, data: StockData) -> float:
        """How close is price to a high-gamma strike cluster?
        Closer = stronger pin → mean-reverting → good for swing holds.
        """
        if not data.gex.gex_per_strike:
            return 0.0
        # Find strike with highest absolute GEX
        max_gex_strike = max(data.gex.gex_per_strike, key=lambda k: abs(data.gex.gex_per_strike[k]))
        dist = abs(data.price - max_gex_strike) / data.price
        if dist < 0.01:
            return 1.0  # Very close to pin
        elif dist < 0.03:
            return 0.5
        return 0.0

    def _vanna_signal(self, data: StockData) -> float:
        """Vanna flow: dDelta/dVol.
        When vol drops (common during rallies), positive vanna means dealers
        need to buy more stock → bullish.
        Normalized to [-1, 1] scale.
        """
        vanna = data.gex.net_vanna_exposure
        if vanna == 0 or data.market_cap == 0:
            return 0.0
        # Normalize by market cap for cross-stock comparison
        norm = vanna / (data.market_cap / 1e6)
        return float(np.clip(norm * 100, -1.0, 1.0))

    def _charm_signal(self, data: StockData) -> float:
        """Charm flow: dDelta/dTime.
        As time passes, OTM options lose delta → dealers must adjust hedges.
        Positive charm = dealers need to sell → headwind.
        Negative charm = dealers need to buy → tailwind.
        """
        charm = data.gex.net_charm_exposure
        if charm == 0 or data.market_cap == 0:
            return 0.0
        norm = charm / (data.market_cap / 1e6)
        return float(np.clip(-norm * 100, -1.0, 1.0))  # flip sign: negative charm = bullish

    def _wall_magnet_signal(self, data: StockData) -> float:
        """Price tends to be pulled toward call/put walls.
        If call wall is above price → bullish magnet.
        If put wall is below price → support.
        """
        cw = data.gex.call_wall
        pw = data.gex.put_wall
        if cw <= 0 and pw <= 0:
            return 0.0

        signal = 0.0
        if cw > 0 and cw > data.price:
            # Call wall above → upward magnet
            dist = (cw - data.price) / data.price
            signal += min(dist / 0.03, 1.0) * 0.5
        if pw > 0 and pw < data.price:
            # Put wall below → support
            dist = (data.price - pw) / data.price
            if dist < 0.05:  # Close support
                signal += 0.5
        return float(np.clip(signal, 0.0, 1.0))

    def buy_score(self, data: StockData) -> float:
        score = 0.0

        # 1. Positive GEX environment: stabilizing, mean-reverting (25%)
        #    Good for buying dips because dealers cushion the downside
        if data.gex.net_gex > 0:
            # In positive gamma, buy when price dipped toward support
            if data.returns_1d < -0.005:  # Small dip
                score += 0.25

        # 2. Price below gamma flip but GEX flipping positive (20%)
        #    Transitioning from unstable to stable = inflection buy
        flip_dist = self._price_vs_gamma_flip(data)
        if -0.02 < flip_dist < 0.01:
            score += 0.20

        # 3. Vanna flow bullish (20%)
        vanna_sig = self._vanna_signal(data)
        if vanna_sig > 0:
            score += 0.20 * vanna_sig

        # 4. Charm flow bullish (15%)
        charm_sig = self._charm_signal(data)
        if charm_sig > 0:
            score += 0.15 * charm_sig

        # 5. Call wall magnet above price (20%)
        wall_sig = self._wall_magnet_signal(data)
        if wall_sig > 0:
            score += 0.20 * wall_sig

        return round(min(score, 1.0), 4)

    def sell_score(self, data: StockData, entry_price: float) -> float:
        score = 0.0

        # 1. Negative GEX: destabilizing, sells amplified (30%)
        if data.gex.net_gex < 0:
            score += 0.30

        # 2. Price approaching call wall from below (resistance) (20%)
        cw = data.gex.call_wall
        if cw > 0 and data.price > 0:
            dist_to_wall = (cw - data.price) / data.price
            if 0 < dist_to_wall < 0.01:  # Very close to resistance
                score += 0.20

        # 3. Vanna flow bearish (20%)
        vanna_sig = self._vanna_signal(data)
        if vanna_sig < 0:
            score += 0.20 * abs(vanna_sig)

        # 4. Charm flow bearish (15%)
        charm_sig = self._charm_signal(data)
        if charm_sig < 0:
            score += 0.15 * abs(charm_sig)

        # 5. Price above gamma flip and falling (losing positive GEX support) (15%)
        flip_dist = self._price_vs_gamma_flip(data)
        if flip_dist > 0 and data.returns_1d < -0.01:
            score += 0.15

        return round(min(score, 1.0), 4)

    def swing_score(self, data: StockData) -> float:
        """Overnight hold viability based on gamma profile."""
        score = 0.0

        # Positive GEX = stabilizing overnight, safe to hold
        if data.gex.net_gex > 0:
            score += 0.35

        # Pin risk high = price likely to stay here overnight
        pin = self._pin_risk_score(data)
        score += 0.25 * pin

        # Positive vanna = vol compression helps longs overnight
        vanna_sig = self._vanna_signal(data)
        if vanna_sig > 0:
            score += 0.20 * vanna_sig

        # Price above gamma flip = stable zone
        if self._price_vs_gamma_flip(data) > 0.01:
            score += 0.20

        return round(min(score, 1.0), 4)
