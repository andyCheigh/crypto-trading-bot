INITIAL_CAPITAL = 10_000.0
MAX_HOLDINGS = 5
SELL_CHECK_INTERVAL = 15      # seconds
BUY_SCAN_INTERVAL = 60        # seconds

# Universe: liquid large-caps with active options markets
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "KO", "PEP", "COST", "AVGO", "LLY", "TMO", "ACN", "MCD",
    "NFLX", "CRM", "AMD", "INTC", "CSCO", "ADBE", "ORCL", "BA", "GS",
    "CAT", "DE", "UPS", "RTX",
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
POSITION_SIZE_PCT = 0.20     # 20% of available cash per trade
