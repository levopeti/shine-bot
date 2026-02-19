# AI Trading Framework

## Overview

A comprehensive Python framework for automated trading using reinforcement learning agents across multiple asset classes (stocks, crypto, commodities, forex).

## Features

- **Multi-Asset Support**: Trade stocks, cryptocurrencies, commodities, and forex
- **Multiple AI Agents**: Machine Learning, Deep Learning (DQN), and LLM-based agents
- **Reinforcement Learning**: Train agents using DQN/DDQN with experience replay
- **Pattern Recognition**: Analyzes correlations across asset classes
- **Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages
- **Backtesting Engine**: Validate strategies on historical data
- **Real-time Dashboard**: Interactive visualization with Plotly/Dash
- **Risk Management**: Position sizing, transaction costs, portfolio constraints
- **Live Trading Ready**: Integration framework for broker APIs

## Project Structure

```
trading_framework/
├── config/                 # Configuration settings
├── data/                   # Data fetching and processing
├── agents/                 # AI trading agents
├── environment/            # Trading environment (Gym-compatible)
├── strategies/             # Trading strategies
├── backtesting/            # Backtesting engine
├── live_trading/           # Live trading integration
├── visualization/          # Charts and dashboard
├── models/                 # Trained model storage
├── main.py                 # Main execution script
└── requirements.txt        # Dependencies
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (optional, for real data)
export ALPHA_VANTAGE_API_KEY="your_key"
export BINANCE_API_KEY="your_key"
```

## Usage

### 1. Train Agent

Train a reinforcement learning agent on historical data:

```bash
# Deep Learning (DQN)
python main.py --mode train --agent dl --episodes 100

# Random Forest
python main.py --mode train --agent ml_rf --episodes 50

# Gradient Boosting
python main.py --mode train --agent ml_gb --episodes 50

# Ensemble
python main.py --mode train --agent ensemble --episodes 100
```

### 2. Backtest Strategy

Test the trained agent on historical data:

```bash
python main.py --mode backtest --model-path models/saved_models/agent_final.pth
```

### 3. Run Dashboard

Launch the interactive dashboard:

```bash
python main.py --mode dashboard --model-path models/saved_models/agent_final.pth
```

Then open your browser to `http://localhost:8050`

### 4. Live Trading (Advanced)

Connect to a real broker API for live trading:

```bash
python main.py --mode live --model-path models/saved_models/agent_final.pth
```

**Warning**: Live trading involves real money. Test thoroughly in simulation first!

## Configuration

Edit `config/settings.py` to customize:

- Assets to trade
- Training parameters
- Risk management rules
- API credentials

## Agent Types

### 1. Deep Learning Agent (DQN)
- Neural network-based decision making
- Experience replay for stable learning
- Epsilon-greedy exploration

### 2. Machine Learning Agent
- Random Forest or Gradient Boosting
- Supervised learning from historical data
- Fast inference

### 3. LLM Agent (Experimental)
- Natural language reasoning
- Placeholder for future integration

### 4. Ensemble Strategy
- Combine multiple agents
- Weighted voting for robust decisions

## Performance Metrics

The framework calculates:
- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio

## Technical Indicators

Automatically calculates:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- MACD
- RSI
- Bollinger Bands
- Volume indicators

## Backtesting

The backtesting engine simulates realistic trading:
- Transaction costs
- Position limits
- Market data replay
- Trade history tracking

## Visualization

Interactive charts include:
- Portfolio value over time
- Asset price comparisons
- Drawdown analysis
- Trade distribution
- Correlation matrices

## Risk Management

Built-in safeguards:
- Maximum position size limits
- Transaction cost modeling
- Cash reserve requirements
- Portfolio diversification

## Data Sources

Default data sources (via yfinance):
- Stocks: Yahoo Finance
- Crypto: Binance/Yahoo Finance
- Commodities: Futures contracts
- Forex: Currency pairs

Optional premium APIs:
- Alpha Vantage
- Polygon.io
- Twelve Data

## Extending the Framework

### Add New Agents

Create a new agent class in `agents/` implementing the base interface:

```python
class MyCustomAgent(BaseAgent):
    def select_action(self, state):
        # Your logic here
        pass
```

### Add New Indicators

Extend `DataProcessor.calculate_technical_indicators()` with custom indicators.

### Custom Reward Functions

Modify `environment/reward_functions.py` to implement custom reward logic.

## License

This project is for educational purposes. Use at your own risk.

## Disclaimer

This software is provided for educational and research purposes only. Trading financial instruments carries a high level of risk and may not be suitable for all investors. Past performance is not indicative of future results. Always conduct thorough testing before deploying any trading strategy with real capital.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For questions or issues, please open an issue on GitHub.

## Roadmap

- [ ] Add PPO and A2C algorithms
- [ ] Implement LLM agent integration
- [ ] Add more technical indicators
- [ ] Portfolio optimization module
- [ ] Sentiment analysis integration
- [ ] Multi-timeframe analysis
- [ ] Advanced order types
- [ ] Paper trading simulation

## Version

Current version: 1.0.0

## Author

Trading Framework Development Team
