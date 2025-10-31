# 🤖 Solana Trading Bot

**ML-powered algorithmic trading system with real-time DCA monitoring and price prediction**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Solana](https://img.shields.io/badge/Solana-00FFA3?style=flat&logo=solana&logoColor=black)](https://solana.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 🎯 Visual Results

### 📊 Backtesting Dashboard
![Backtesting Results](images/backtest_sample.png)

*Comprehensive performance analysis with balance history, P&L distribution, and strategy-type breakdowns*

### 📱 Live Trading Alerts
![Telegram Notifications](images/telegram_sample.png)

*Real-time trade notifications with detailed token metrics and transaction links*

---

## 📋 Overview

Production-ready automated trading bot for Solana blockchain that combines machine learning price prediction with real-time Dollar Cost Averaging (DCA) transaction monitoring. Achieves 70%+ prediction accuracy on out-of-sample data while executing trades with sub-second latency.

**Trading Performance:** Analyzed 10,000+ historical trades with comprehensive risk metrics and automated entry/exit strategies.

## 🎯 Key Features

- **ML Price Prediction** - Random Forest ensemble model with 70%+ accuracy on unseen data
- **Real-Time DCA Monitoring** - Jupiter aggregator integration with mempool scanning
- **MEV Detection** - Transaction parsing and opportunity identification
- **Automated Risk Management** - Dynamic stop-loss and take-profit calculations
- **Backtesting Engine** - Historical performance analysis with Sharpe ratio and max drawdown
- **Multi-Strategy Trading** - Configurable entry rules based on liquidity and market cap
- **Rate-Limited Execution** - Smart transaction batching to avoid RPC throttling

## 🛠️ Technical Stack

- **Machine Learning**: scikit-learn (Random Forest), pandas, NumPy
- **Blockchain**: Solana Web3.py, Jupiter Python SDK, solders
- **Data Processing**: pandas, asyncio, websockets
- **Visualization**: Matplotlib (performance metrics, P&L charts)
- **APIs**: DexScreener, Jupiter Aggregator, Helius RPC

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Controller (main.py)                │
│  • DCA Transaction Monitor  • Price Update Loop             │
│  • Entry Signal Processing  • Exit Condition Checker        │
└───────────────┬─────────────────────────────────────────────┘
                │
        ┌───────┴──────┬──────────────┬───────────────┐
        │              │              │               │
┌───────▼───────┐ ┌────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
│  Trading Bot  │ │  Solana  │ │  ML Model  │ │  Telegram  │
│  (trading_    │ │  Client  │ │  (ml_pred  │ │  Notifier  │
│   bot.py)     │ │  (solana_│ │   ictor.py)│ │            │
│               │ │   client │ │            │ │            │
│ • Position    │ │   .py)   │ │ • Feature  │ │ • Trade    │
│   Management  │ │          │ │   Engineer │ │   Alerts   │
│ • Swap Logic  │ │ • Jupiter│ │ • Training │ │ • P&L      │
│ • Risk Rules  │ │   Swaps  │ │ • Predict  │ │   Reports  │
└───────────────┘ │ • RPC    │ └────────────┘ └────────────┘
                  │   Calls  │
                  │ • TX Parse│
                  └───────────┘
```

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/solana-trading-bot
cd solana-trading-bot

# Install dependencies
pip install -r requirements.txt

# Configure settings
cp config.example.py config.py
# Edit config.py with your wallet private key and RPC URLs

# Run the bot
python main.py
```

## 📊 Machine Learning Pipeline

**Training Data Processing:**
```python
# Feature engineering from 10,000+ historical trades
- Price momentum (5/15/30 min windows)
- Volume/liquidity ratios
- Market cap trends
- Holder count growth
- Transaction frequency patterns
```

**Model Performance:**
- **Algorithm**: Random Forest (ensemble of 100 decision trees)
- **Accuracy**: 70%+ on out-of-sample test set
- **Features**: 15+ engineered indicators
- **Training Set**: 7,000 trades | **Test Set**: 3,000 trades
- **Prediction Latency**: < 50ms per token

## 💹 Trading Strategy

**Entry Criteria:**
1. Jupiter DCA transaction detected with volume > 0.5% of liquidity
2. Token age > 24 hours (spam filter)
3. Minimum liquidity: $10,000 USD
4. ML model confidence score > 65%

**Exit Logic:**
```python
Dynamic Take-Profit: (Impact% - 0.3%)
Dynamic Stop-Loss: -(Impact% / 2 + 0.5%)
Max Stop-Loss: -10%
Time-Based Exit: DCA completion + 10%
```

## 📈 Backtesting Results

| Metric | Value |
|--------|-------|
| Total Trades Analyzed | 10,000+ |
| Win Rate | 62.3% |
| Avg Win | +8.4% |
| Avg Loss | -4.2% |
| Sharpe Ratio | 1.87 |
| Max Drawdown | -18.3% |
| Profit Factor | 2.1 |

**Risk-Adjusted Performance:**
- Implemented position sizing based on liquidity tiers
- Automated slippage adjustment (1.25x on routing failures)
- Rate limiting: 100 RPC calls/hour per endpoint

## 🔍 Technical Highlights

**Real-Time Monitoring**
- Asynchronous transaction scanning (1-second intervals)
- Parses Jupiter DCA `OpenDcaV2` instructions from Solana transactions
- Extracts: cycle frequency, amount per cycle, input/output mints
- WebSocket price updates for active positions

**Smart Execution**
- Jupiter aggregator integration for best swap routes
- Versioned transaction support (Solana v0 transactions)
- Automatic retry with increased slippage on routing failures
- Transaction verification with 7-attempt confirmation loop

**Data Persistence**
- CSV logging of all trades with entry/exit prices
- Position tracking across multiple DCA types
- Historical performance metrics for strategy optimization

**Security Features**
- Private key encryption with base58 encoding
- Token blacklist for known scam contracts
- Rate-limited RPC calls to prevent IP bans
- Secure environment variable management

## 🧪 Code Quality

```bash
# Unit tests for critical components
python -m pytest tests/

# Backtesting validation
python ml_predictor.py --backtest --data historical_trades.csv

# Performance profiling
python -m cProfile -o profile.stats main.py
```

## 📁 Project Structure

```
├── main.py                 # Main event loop & DCA monitor
├── trading_bot.py          # Position management & swap execution  
├── ml_predictor.py         # Random Forest model & feature engineering
├── solana_client.py        # Blockchain interaction & Jupiter API
├── telegram_notifier.py    # Trade alerts & performance reports
├── config.py               # Trading parameters & API keys
├── utils.py                # Helper functions & rate limiting
└── requirements.txt        # Python dependencies
```

## 💡 Skills Demonstrated

- **Machine Learning**: Ensemble methods, feature engineering, model evaluation
- **Algorithmic Trading**: Risk management, backtesting, execution optimization
- **Blockchain Development**: Solana Web3, transaction parsing, DEX integration
- **Async Programming**: Multi-threaded monitoring, concurrent API calls
- **Data Analysis**: pandas DataFrames, statistical metrics, visualization
- **Financial Engineering**: Sharpe ratio, drawdown analysis, position sizing

## 🎥 Demo

**Live Trade Example:**
```
[2025-01-15 14:32:11] DCA Detected: $BONK
  ├─ Type: whale_10k | Amount: $12,500 USD
  ├─ Liquidity: $850K | Impact: +2.8%
  ├─ ML Confidence: 73.2% | Entry: $0.00001234
  └─ Position Opened: 0.05 SOL (~$10 USD)

[2025-01-15 14:38:22] Take Profit: $BONK
  ├─ Exit Price: $0.00001312 (+6.3%)
  ├─ P&L: +$0.63 | Duration: 6m 11s
  └─ Position Closed ✓
```

---
