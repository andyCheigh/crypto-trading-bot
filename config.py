INITIAL_CAPITAL = 10_000.0
MAX_HOLDINGS = 10
SELL_CHECK_INTERVAL = 15      # seconds
BUY_SCAN_INTERVAL = 60        # seconds

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

# Buy threshold: ensemble score must exceed this to trigger a buy
BUY_THRESHOLD = 0.50

# Sell thresholds
STOP_LOSS_PCT = -0.025       # -2.5% hard stop (tighter for day trading)
TAKE_PROFIT_PCT = 0.04       # +4% take profit
TRAILING_STOP_PCT = 0.015    # 1.5% trailing stop from peak

# Per-position sizing: fraction of available cash
POSITION_SIZE_PCT = 0.10     # 10% of available cash per trade

# End-of-day settings
EOD_CLOSE_TIME_MINUTES_BEFORE = 15  # Start closing positions 15 min before close (3:45 PM ET)
SWING_HOLD_THRESHOLD = 0.60         # Swing score must exceed this to hold overnight
