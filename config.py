INITIAL_CAPITAL = 10_000.0
MAX_HOLDINGS = 10
SELL_CHECK_INTERVAL = 30      # seconds (avoids yfinance rate limits with concurrent buy scans)
BUY_SCAN_INTERVAL = 120       # seconds (avoids yfinance rate limits with 200+ symbols)

# ---------------------------------------------------------------------------
# Options Contract Selection
# ---------------------------------------------------------------------------
TARGET_DELTA_RANGE = (0.30, 0.50)    # Sweet spot: leverage + decent probability
PREFERRED_DTE_MIN = 14               # Avoid extreme theta decay (weeklies)
PREFERRED_DTE_MAX = 45               # Avoid low gamma/vega (far-dated)
MIN_OPEN_INTEREST = 50               # Liquidity filter (50 sufficient for small account)
MAX_BID_ASK_SPREAD_PCT = 0.10        # Max 10% bid-ask spread
CONTRACT_MULTIPLIER = 100             # Standard options multiplier

# ---------------------------------------------------------------------------
# Greeks-Based Exit Criteria
# ---------------------------------------------------------------------------
PREMIUM_STOP_LOSS_PCT = -0.50        # Exit if premium drops 50%
PREMIUM_TAKE_PROFIT_PCT = 1.00       # Exit if premium doubles (100% gain)
PREMIUM_TRAILING_STOP_PCT = 0.30     # Exit if premium drops 30% from peak
DELTA_STOP_LOSS = 0.10               # Exit if |delta| drops below 0.10 (deep OTM)
IV_CRUSH_EXIT_PCT = 0.20             # Exit if IV drops 20%+ from entry
MAX_THETA_DECAY_PCT = 0.05           # Exit if daily theta > 5% of premium (~10-12 DTE territory, early warning before 3 DTE hard cutoff)
MIN_HOLD_SECONDS = 180               # Minimum hold time before evaluating exits (3 min cooldown)
NEAR_EXPIRY_DTE = 3                  # Force close options within 3 DTE (gamma risk)

# Universe: 200 liquid stocks with active options markets
STOCK_UNIVERSE = [
    # Mega-cap tech (20)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO", "NFLX", "QCOM",
    "TXN", "AMAT", "MU", "NOW",
    # Mid-cap tech & software (20)
    "SNOW", "PLTR", "PANW", "CRWD", "ZS", "DDOG", "NET", "SHOP",
    "COIN", "MRVL", "KLAC", "LRCX", "SNPS", "CDNS", "FTNT", "TEAM",
    "WDAY", "HUBS", "OKTA", "VEEV",
    # Finance (20)
    "JPM", "V", "MA", "GS", "MS", "BAC", "WFC", "C", "BLK", "SCHW",
    "AXP", "USB", "PNC", "TFC", "COF", "ICE", "CME", "SPGI", "MCO",
    "MSCI",
    # Healthcare & biotech (20)
    "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDT", "SYK",
    "BSX", "ZTS", "CI",
    # Consumer discretionary (20)
    "HD", "LOW", "NKE", "SBUX", "MCD", "TJX", "ROST", "CMG", "YUM",
    "DPZ", "LULU", "BKNG", "ABNB", "UBER", "LYFT", "DASH", "ETSY",
    "W", "DKNG", "PENN",
    # Consumer staples (10)
    "WMT", "PG", "KO", "PEP", "COST", "CL", "MDLZ", "KHC", "GIS",
    "SJM",
    # Energy (15)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    "HAL", "DVN", "FANG", "PXD", "WMB", "KMI",
    # Industrials (20)
    "CAT", "DE", "UPS", "RTX", "BA", "HON", "GE", "LMT", "UNP", "MMM",
    "FDX", "WM", "EMR", "ITW", "ROK", "ETN", "PH", "GD", "NOC", "TDG",
    # Materials & mining (10)
    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "STLD", "CF",
    "MOS",
    # Communication & media (15)
    "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO", "RBLX",
    "MTCH", "ZM", "ROKU", "SPOT", "PINS", "SNAP",
    # REITs & utilities (10)
    "AMT", "PLD", "CCI", "EQIX", "SPG", "NEE", "DUK", "SO", "D",
    "AEP",
    # Semiconductors (10)
    "TSM", "ASML", "ARM", "ON", "SWKS", "QRVO", "ADI", "NXPI", "MCHP",
    "GFS",
    # Fintech & payments (10)
    "PYPL", "FIS", "FISV", "GPN", "AFRM", "SOFI", "HOOD", "BILL",
    "TOST", "MQ",
    # International ADRs (10)
    "BABA", "SE", "MELI", "NU", "GRAB", "PDD", "JD", "BIDU", "NIO",
    "LI",
    # ETFs — most liquid options markets in the world
    "SPY",
    # Misc high-options-volume (10)
    "ACN", "IBM", "GM", "F", "RIVN", "LCID", "SMCI", "AI", "IONQ",
    "RGTI",
]

# Algorithm weights for ensemble scoring (sum to 1.0)
ALGO_WEIGHTS = {
    "vol_arb":          0.35,   # Volatility Arbitrage (IV vs RV, vol regime)
    "gamma_exposure":   0.35,   # Dealer Gamma Exposure & Vanna/Charm flows
    "options_flow":     0.30,   # Options Order Flow & Smart Money Sentiment
}

# Buy threshold: ensemble conviction must exceed this to trigger a trade
# With weights summing to 1.0, 0.35 requires 2+ algos to agree with solid conviction
# (e.g., 2 algos at 0.50 each = 0.35*0.50 + 0.35*0.50 = 0.35)
BUY_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Kelly Position Sizing
# ---------------------------------------------------------------------------
KELLY_FRACTION = 0.50        # Half-Kelly — industry standard for estimation error safety
MAX_POSITION_PCT = 0.15      # Hard cap: 15% of available cash per trade
MIN_POSITION_PCT = 0.03      # Floor: 3% to avoid dust positions
POSITION_SIZE_PCT = 0.10     # Fallback if Kelly unavailable

# ---------------------------------------------------------------------------
# Portfolio Correlation Management
# ---------------------------------------------------------------------------
MAX_PER_SECTOR = 3           # Max positions per sector (prevents concentration)
MAX_PORTFOLIO_BETA = 2.5     # Portfolio-level beta cap (reasonable for levered options)

# ---------------------------------------------------------------------------
# Risk Circuit Breakers
# ---------------------------------------------------------------------------
MAX_DRAWDOWN_PCT = 0.20          # Kill switch: stop ALL new buys if equity drops 20% from peak
                                  # Every real desk has this. Prevents catastrophic loss spirals.

# End-of-day settings
EOD_CLOSE_TIME_MINUTES_BEFORE = 15  # Start closing positions 15 min before close (3:45 PM ET)
SWING_HOLD_THRESHOLD = 0.60         # Swing score must exceed this to hold overnight
