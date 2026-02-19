# config/settings.py

import os
from datetime import datetime, timedelta

class Config:
    """
    Central configuration for the trading framework
    """

    # API Keys (load from environment variables for security)
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')

    # Data settings
    ASSETS = {
        'stocks': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
        'crypto': ['BTC/USD', 'ETH/USD', 'SOL/USD'],
        'commodities': ['GC=F', 'CL=F', 'SI=F'],  # Gold, Oil, Silver
        'forex': ['EUR/USD', 'GBP/USD', 'JPY/USD']
    }

    # ======== 30 ÉV BEÁLLÍTÁS ========
    TODAY = datetime.now()

    # Training: 24 év (1996-2020)
    TRAIN_START_DATE = TODAY - timedelta(days=30 * 365)
    TRAIN_END_DATE = TODAY - timedelta(days=6 * 365)

    # Test: 6 év (2020-2026)
    TEST_START_DATE = TRAIN_END_DATE + timedelta(days=1)
    TEST_END_DATE = TODAY

    # ======== KRITIKUS: ROLLING WINDOW ========
    DATA_INTERVAL = '1d'  # Csak napi! (30 év intraday irreális)
    EPISODE_LENGTH = 252  # 1 év = 252 trading nap
    NUM_EPISODES = 100  # 100× random 1 éves ablak

    # ======== EGYÉB ========
    INITIAL_BALANCE = 100_000
    MAX_POSITION_SIZE = 0.2
    TRANSACTION_COST = 0.001
    BATCH_SIZE = 128  # Nagyobb batch
    MEMORY_SIZE = 20_000  # Nagyobb memory

    # Agent training parameters
    RL_CONFIG = {
        'learning_rate': 0.0001,
        'gamma': 0.99,           # Discount factor
        'epsilon_start': 1.0,    # Exploration rate
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'memory_size': 10000,
        'update_frequency': 4
    }

    # Model paths
    MODEL_DIR = 'models/saved_models'
    CHECKPOINT_DIR = 'models/checkpoints'

    # Visualization settings
    DASHBOARD_PORT = 8050
    UPDATE_INTERVAL = 5000  # milliseconds
