"""
Algorithm 2: Gamma Exposure & Dealer Flow (The HRT/Citadel Edge)

Market makers delta-hedge their options books. Their hedging creates
predictable price dynamics based on their aggregate gamma position.

BUY CALL when:
  - Positive GEX + dip: dealers cushion downside, mean reversion up
  - Price near gamma flip from below: transitioning to positive gamma territory
  - Vanna bullish: vol dropping → dealers buy → upward pressure
  - Charm bullish: time decay → dealers buy → tailwind
  - Call wall above + put wall below: price pinned, pulled toward call wall

BUY PUT when:
  - Negative GEX: dealer flow destabilizing, amplifies selloffs
  - Negative GEX + rally: unsustainable, reversal setup
  - Price above call wall: hit resistance, likely rejection
  - Vanna bearish: vol rising → dealers sell → downward pressure
  - Charm bearish: time decay → dealers sell → headwind
  - Price below gamma flip and falling: negative gamma territory, accelerating
"""

from __future__ import annotations

import numpy as np

from algorithms.signal import OptionSignal
from data_fetcher import StockData


class GammaExposureAlgorithm:
    name = "gamma_exposure"

    def _price_vs_gamma_flip(self, data: StockData) -> float:
        """Signed distance from price to gamma flip level."""
        flip = data.gex.gamma_flip_level
        if flip <= 0 or data.price <= 0:
            return 0.0
        return (data.price - flip) / data.price

    def _pin_risk_score(self, data: StockData) -> float:
        """How close is price to a high-gamma strike cluster?"""
        if not data.gex.gex_per_strike:
            return 0.0
        max_gex_strike = max(data.gex.gex_per_strike, key=lambda k: abs(data.gex.gex_per_strike[k]))
        dist = abs(data.price - max_gex_strike) / data.price
        if dist < 0.01:
            return 1.0
        elif dist < 0.03:
            return 0.5
        return 0.0

    def _vanna_signal(self, data: StockData) -> float:
        """Vanna flow: positive vanna + vol drop = dealers buy = bullish."""
        vanna = data.gex.net_vanna_exposure
        if vanna == 0 or data.market_cap == 0:
            return 0.0
        norm = vanna / (data.market_cap / 1e6)
        return float(np.clip(norm * 100, -1.0, 1.0))

    def _charm_signal(self, data: StockData) -> float:
        """Charm flow: negative charm = dealers buy = bullish."""
        charm = data.gex.net_charm_exposure
        if charm == 0 or data.market_cap == 0:
            return 0.0
        norm = charm / (data.market_cap / 1e6)
        return float(np.clip(-norm * 100, -1.0, 1.0))

    def _wall_magnet_signal(self, data: StockData) -> float:
        """Price tends to be pulled toward call/put walls.
        Returns positive for upward pull, negative for downward pull.
        """
        cw = data.gex.call_wall
        pw = data.gex.put_wall
        if cw <= 0 and pw <= 0:
            return 0.0

        signal = 0.0
        if cw > 0 and cw > data.price:
            dist = (cw - data.price) / data.price
            signal += min(dist / 0.03, 1.0) * 0.5
        if pw > 0 and pw < data.price:
            dist = (data.price - pw) / data.price
            if dist < 0.05:
                signal += 0.5
        # Negative if price is above call wall (resistance rejection)
        if cw > 0 and data.price > cw:
            dist = (data.price - cw) / data.price
            signal -= min(dist / 0.02, 1.0) * 0.5
        # Negative if price is below put wall (support broken)
        if pw > 0 and data.price < pw:
            dist = (pw - data.price) / data.price
            signal -= min(dist / 0.02, 1.0) * 0.5

        return float(np.clip(signal, -1.0, 1.0))

    def signal(self, data: StockData) -> OptionSignal:
        """Produce CALL/PUT signal based on dealer gamma positioning."""
        bullish_score = 0.0
        bearish_score = 0.0
        component_scores = {}

        # --- BULLISH COMPONENTS (favor CALL) ---

        # 1. Positive GEX: dealer long gamma cushions downside (20%)
        #    Base 0.50: positive gamma is a stabilizing floor, not a catalyst
        #    Negative GEX at 0.60 is justified — short gamma amplifies moves directionally
        gex_bull = 0.0
        if data.gex.net_gex > 0:
            if data.returns_1d < -0.003:
                gex_bull = 1.0  # Dip in positive gamma = strong mean-reversion buy
            else:
                gex_bull = 0.50  # Positive gamma = supportive floor
        component_scores["pos_gex_dip"] = gex_bull
        bullish_score += 0.20 * gex_bull

        # 2. Price near gamma flip from below (15%)
        flip_dist = self._price_vs_gamma_flip(data)
        flip_bull = 0.0
        if -0.02 < flip_dist < 0.01:
            flip_bull = 1.0
        component_scores["near_flip_bull"] = flip_bull
        bullish_score += 0.15 * flip_bull

        # 3. Vanna bullish (15%)
        vanna_sig = self._vanna_signal(data)
        vanna_bull = max(vanna_sig, 0.0)
        component_scores["vanna_bull"] = vanna_bull
        bullish_score += 0.15 * vanna_bull

        # 4. Charm bullish (10%)
        charm_sig = self._charm_signal(data)
        charm_bull = max(charm_sig, 0.0)
        component_scores["charm_bull"] = charm_bull
        bullish_score += 0.10 * charm_bull

        # 5. Call wall above + put wall support (15%)
        wall_sig = self._wall_magnet_signal(data)
        wall_bull = max(wall_sig, 0.0)
        component_scores["wall_magnet_bull"] = wall_bull
        bullish_score += 0.15 * wall_bull

        # 6. Price near put wall from above: support magnet (5%)
        #    Symmetric to bearish call wall rejection — dealers defend put wall
        pw = data.gex.put_wall
        put_support_bull = 0.0
        if pw > 0 and data.price > 0:
            dist_above_pw = (data.price - pw) / data.price
            if 0 < dist_above_pw < 0.01:
                put_support_bull = 0.80  # Sitting right on put wall support
            elif 0.01 <= dist_above_pw < 0.02:
                put_support_bull = 0.40  # Near put wall support
        component_scores["at_put_wall_support"] = put_support_bull
        bullish_score += 0.05 * put_support_bull

        # --- BEARISH COMPONENTS (favor PUT) ---

        # 6. Negative GEX: destabilizing, amplifies selloffs (20%)
        gex_bear = 0.0
        if data.gex.net_gex < 0:
            gex_bear = 0.60
            if data.returns_1d > 0.003:
                # Negative GEX + rally = unsustainable, reversal setup
                gex_bear = 1.0
        component_scores["neg_gex"] = gex_bear
        bearish_score += 0.20 * gex_bear

        # 7. Price above call wall: hit resistance (10%)
        cw = data.gex.call_wall
        wall_resist_bear = 0.0
        if cw > 0 and data.price > 0:
            dist_above = (data.price - cw) / data.price
            if dist_above > 0:
                wall_resist_bear = min(dist_above / 0.02, 1.0)
            elif 0 > dist_above > -0.01:
                wall_resist_bear = 0.70  # Very close to resistance
        component_scores["at_call_wall"] = wall_resist_bear
        bearish_score += 0.10 * wall_resist_bear

        # 8. Vanna bearish (15%)
        vanna_bear = max(-vanna_sig, 0.0)
        component_scores["vanna_bear"] = vanna_bear
        bearish_score += 0.15 * vanna_bear

        # 9. Charm bearish (10%)
        charm_bear = max(-charm_sig, 0.0)
        component_scores["charm_bear"] = charm_bear
        bearish_score += 0.10 * charm_bear

        # 10. Below gamma flip and falling: accelerating negative gamma (15%)
        flip_bear = 0.0
        if flip_dist < -0.01 and data.returns_1d < -0.005:
            flip_bear = min(abs(flip_dist) / 0.03, 1.0)
        component_scores["below_flip_falling"] = flip_bear
        bearish_score += 0.15 * flip_bear

        # 11. Wall magnet pulling down (10%)
        wall_bear = max(-wall_sig, 0.0) if wall_sig < 0 else 0.0
        component_scores["wall_magnet_bear"] = wall_bear
        bearish_score += 0.10 * wall_bear

        # --- Direction ---
        bullish_score = min(bullish_score, 1.0)
        bearish_score = min(bearish_score, 1.0)
        component_scores["bullish_total"] = round(bullish_score, 4)
        component_scores["bearish_total"] = round(bearish_score, 4)

        net = bullish_score - bearish_score
        if net > 0.08:
            direction = "CALL"
            conviction = bullish_score
            # GEX signals are short-term: prefer closer expiry, higher delta
            preferred_delta = 0.45
            preferred_dte = 21
        elif net < -0.08:
            direction = "PUT"
            conviction = bearish_score
            # Puts in negative gamma: slightly OTM for leverage on accelerating move
            preferred_delta = 0.35
            preferred_dte = 21
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
        """Exit score based on gamma regime reversal."""
        sig = self.signal(data)
        if position_direction == "CALL" and sig.direction == "PUT":
            return min(0.50 + sig.conviction * 0.50, 1.0)
        if position_direction == "PUT" and sig.direction == "CALL":
            return min(0.50 + sig.conviction * 0.50, 1.0)
        if sig.direction == "NEUTRAL":
            return 0.25
        return 0.0

    def swing_score(self, data: StockData) -> float:
        """Overnight hold viability based on gamma profile."""
        score = 0.0
        if data.gex.net_gex > 0:
            score += 0.35
        pin = self._pin_risk_score(data)
        score += 0.25 * pin
        vanna_sig = self._vanna_signal(data)
        if vanna_sig > 0:
            score += 0.20 * vanna_sig
        if self._price_vs_gamma_flip(data) > 0.01:
            score += 0.20
        return round(min(score, 1.0), 4)
