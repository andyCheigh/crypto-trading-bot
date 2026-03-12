# Crypto Trading Bot

Multi-signal consensus trading bot for cryptocurrency markets. Uses EMA crossover, RSI, Bollinger Bands, and MACD to generate high-confidence trade signals with built-in risk management.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Copy and configure your API keys
cp .env.example .env
# Edit .env with your exchange API credentials

# Run in dry-run mode (default)
python -m bot.main

# Or with a custom config
python -m bot.main config.yaml
```

## Configuration

Edit `config.yaml` to customize:

- **Exchange**: Binance (default), Coinbase, Kraken, Bybit, OKX, or any CCXT-supported exchange
- **Mode**: `dry_run` (paper trading) or `live` (real money)
- **Symbols**: Any trading pair available on your exchange
- **Strategy parameters**: EMA periods, RSI thresholds, BB width, MACD settings
- **Risk management**: Position sizing, stop-loss, take-profit, max drawdown circuit breaker

## How It Works

The bot requires **3 out of 4** signals to agree before entering or exiting a trade:

| Signal | Buy Condition | Sell Condition |
|--------|--------------|----------------|
| EMA Crossover | Fast EMA crosses above slow | Fast crosses below slow |
| RSI | Below 30 (oversold) | Above 70 (overbought) |
| Bollinger Bands | Price at lower band | Price at upper band |
| MACD | Histogram crosses positive | Histogram crosses negative |

## Dry Run Mode

Set `mode: dry_run` in config.yaml. Configure your starting balance:

```yaml
dry_run:
  starting_balance:
    USDT: 10000.0
    # Or simulate with existing holdings:
    # BTC: 0.15
    # ETH: 2.5
```

## Risk Management

- **Position sizing**: Max 25% of portfolio per trade
- **Stop-loss**: 3% (configurable)
- **Take-profit**: 6% (2:1 reward/risk ratio)
- **Circuit breaker**: Halts trading at 15% max drawdown
- **Max positions**: 3 concurrent positions

## Disclaimer

This bot is for educational purposes. Cryptocurrency trading involves substantial risk. Past performance does not guarantee future results. Use at your own risk.
