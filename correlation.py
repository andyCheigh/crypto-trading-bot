"""
Portfolio Correlation Management — sector concentration and beta exposure limits.

Prevents the portfolio from loading up on correlated positions. At a real desk,
if all 10 positions are tech puts and NASDAQ bounces 2%, you're done. This
module enforces sector limits, portfolio beta caps, and diversification scoring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from data_fetcher import StockData
    from portfolio import OptionPosition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector mapping — derived from config.py universe groupings
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {}

_SECTOR_DEFS = {
    "mega_tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
        "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO", "NFLX", "QCOM",
        "TXN", "AMAT", "MU", "NOW",
    ],
    "mid_tech": [
        "SNOW", "PLTR", "PANW", "CRWD", "ZS", "DDOG", "NET", "SHOP",
        "COIN", "MRVL", "KLAC", "LRCX", "SNPS", "CDNS", "FTNT", "TEAM",
        "WDAY", "HUBS", "OKTA", "VEEV",
    ],
    "finance": [
        "JPM", "V", "MA", "GS", "MS", "BAC", "WFC", "C", "BLK", "SCHW",
        "AXP", "USB", "PNC", "TFC", "COF", "ICE", "CME", "SPGI", "MCO",
        "MSCI",
    ],
    "healthcare": [
        "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR",
        "BMY", "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT", "SYK",
        "BSX", "ZTS", "CI",
    ],
    "consumer_disc": [
        "HD", "LOW", "NKE", "SBUX", "MCD", "TJX", "ROST", "CMG", "YUM",
        "DPZ", "LULU", "BKNG", "ABNB", "UBER", "LYFT", "DASH", "ETSY",
        "W", "DKNG", "PENN",
    ],
    "consumer_staples": [
        "WMT", "PG", "KO", "PEP", "COST", "CL", "MDLZ", "KHC", "GIS",
        "SJM",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
        "HAL", "DVN", "FANG", "PXD", "WMB", "KMI",
    ],
    "industrials": [
        "CAT", "DE", "UPS", "RTX", "BA", "HON", "GE", "LMT", "UNP", "MMM",
        "FDX", "WM", "EMR", "ITW", "ROK", "ETN", "PH", "GD", "NOC", "TDG",
    ],
    "materials": [
        "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "STLD", "CF",
        "MOS",
    ],
    "communication": [
        "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO", "RBLX",
        "MTCH", "ZM", "ROKU", "SPOT", "PINS", "SNAP",
    ],
    "reits_utilities": [
        "AMT", "PLD", "CCI", "EQIX", "SPG", "NEE", "DUK", "SO", "D",
        "AEP",
    ],
    "semiconductors": [
        "TSM", "ASML", "ARM", "ON", "SWKS", "QRVO", "ADI", "NXPI", "MCHP",
        "GFS",
    ],
    "fintech": [
        "PYPL", "FIS", "FISV", "GPN", "AFRM", "SOFI", "HOOD", "BILL",
        "TOST", "MQ",
    ],
    "intl_adr": [
        "BABA", "SE", "MELI", "NU", "GRAB", "PDD", "JD", "BIDU", "NIO",
        "LI",
    ],
    "etf": ["SPY"],
    "misc": [
        "ACN", "IBM", "GM", "F", "RIVN", "LCID", "SMCI", "AI", "IONQ",
        "RGTI",
    ],
}

for sector, tickers in _SECTOR_DEFS.items():
    for t in tickers:
        SECTOR_MAP[t] = sector


# ---------------------------------------------------------------------------
# Diversification penalty by sector count
# ---------------------------------------------------------------------------

_DIVERSITY_PENALTY = {
    0: 1.0,     # empty sector — no penalty
    1: 0.90,    # one position — mild penalty
    2: 0.70,    # two positions — significant penalty
}
# 3+ positions → blocked (handled by max_per_sector check)


class CorrelationManager:
    """Enforces sector concentration limits and portfolio beta caps."""

    def __init__(
        self,
        max_per_sector: int = config.MAX_PER_SECTOR,
        max_portfolio_beta: float = config.MAX_PORTFOLIO_BETA,
    ):
        self.max_per_sector = max_per_sector
        self.max_portfolio_beta = max_portfolio_beta

    @staticmethod
    def sector_of(symbol: str) -> str:
        return SECTOR_MAP.get(symbol, "misc")

    def sector_counts(self, positions: dict[str, "OptionPosition"]) -> dict[str, int]:
        """Count positions per sector."""
        counts: dict[str, int] = {}
        for pos in positions.values():
            sec = self.sector_of(pos.symbol)
            counts[sec] = counts.get(sec, 0) + 1
        return counts

    def portfolio_beta(
        self,
        positions: dict[str, "OptionPosition"],
        data_cache: dict[str, "StockData"],
        total_equity: float,
    ) -> float:
        """Compute portfolio-level beta exposure.

        For options: effective_beta = delta * contracts * 100 * underlying_price * beta / equity
        Sums across all positions for aggregate exposure.
        """
        if total_equity <= 0:
            return 0.0

        total_beta = 0.0
        for pos in positions.values():
            data = data_cache.get(pos.symbol)
            if not data:
                continue
            # Use entry delta as proxy (updated delta would require live fetch)
            underlying_beta = data.beta if data.beta > 0 else 1.0
            notional = abs(pos.entry_delta) * pos.contracts * config.CONTRACT_MULTIPLIER * data.price
            total_beta += notional * underlying_beta / total_equity

        return round(total_beta, 2)

    def can_add_position(
        self,
        symbol: str,
        positions: dict[str, "OptionPosition"],
        data_cache: dict[str, "StockData"],
        candidate_beta: float,
        total_equity: float,
    ) -> tuple[bool, str]:
        """Check if adding this symbol would violate concentration or beta constraints.

        Returns:
            (allowed, reason_if_blocked)
        """
        sector = self.sector_of(symbol)
        counts = self.sector_counts(positions)
        current_count = counts.get(sector, 0)

        # Sector limit
        if current_count >= self.max_per_sector:
            return False, f"Sector '{sector}' at max ({current_count}/{self.max_per_sector})"

        # Portfolio beta check
        current_beta = self.portfolio_beta(positions, data_cache, total_equity)
        # Rough estimate of new position's beta contribution
        est_beta_add = abs(candidate_beta) * 0.10  # assume ~10% notional
        if current_beta + est_beta_add > self.max_portfolio_beta:
            return False, (
                f"Portfolio beta would exceed cap "
                f"({current_beta:.2f} + ~{est_beta_add:.2f} > {self.max_portfolio_beta})"
            )

        return True, ""

    def diversification_multiplier(
        self,
        symbol: str,
        positions: dict[str, "OptionPosition"],
    ) -> float:
        """Return a conviction multiplier that penalizes concentrated sectors.

        0 positions in sector → 1.0x (no penalty)
        1 position → 0.90x
        2 positions → 0.70x
        3+ positions → 0.0x (blocked)
        """
        sector = self.sector_of(symbol)
        counts = self.sector_counts(positions)
        current = counts.get(sector, 0)

        if current >= self.max_per_sector:
            return 0.0
        return _DIVERSITY_PENALTY.get(current, 0.0)
