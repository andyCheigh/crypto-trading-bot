INITIAL_CAPITAL = 10_000.0
MAX_HOLDINGS = 7
SELL_CHECK_INTERVAL = 15      # seconds
BUY_SCAN_INTERVAL = 60        # seconds

# Universe: 100 liquid large-caps with active options markets
STOCK_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO", "NFLX", "QCOM",
    "TXN", "AMAT", "MU", "NOW",
    # Finance
    "JPM", "V", "MA", "GS", "MS", "BAC", "WFC", "C", "BLK", "SCHW",
    # Healthcare
    "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR",
    "BMY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "HD", "LOW", "NKE", "SBUX",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    "HAL",
    # Industrials
    "CAT", "DE", "UPS", "RTX", "BA", "HON", "GE", "LMT", "UNP", "MMM",
    # Communication
    "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "ATVI", "EA", "WBD",
    "PARA",
    # Other large-caps
    "ACN", "IBM", "PYPL", "SQ", "SHOP", "UBER", "ABNB", "COIN", "SNOW",
    "PLTR", "PANW", "CRWD", "ZS", "DDOG", "NET", "MELI", "SE", "BABA",
    "TSM", "ASML",
]

# Algorithm weights for ensemble scoring (sum to 1.0)
ALGO_WEIGHTS = {
    "stat_arb":    0.35,   # Statistical Arbitrage / Mean Reversion
    "momentum":    0.35,   # Momentum + Volume-Weighted
    "vol_surface": 0.30,   # Implied Volatility Surface / Greeks
}

# Buy threshold: ensemble score must exceed this to trigger a buy
BUY_THRESHOLD = 0.60

# Sell thresholds
STOP_LOSS_PCT = -0.03        # -3% hard stop
TAKE_PROFIT_PCT = 0.05       # +5% take profit
TRAILING_STOP_PCT = 0.02     # 2% trailing stop from peak

# Per-position sizing: fraction of available cash
POSITION_SIZE_PCT = 0.14     # 14% of available cash per trade
