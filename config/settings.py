# config/settings.py

import os
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class GoldOptionConfig:
    INITIAL_BALANCE: float = 100000
    POSITION_SIZE: float = 0.1  # Fix 10% pot
    MAX_OPEN_OPTIONS: int = 5  # MAX 5 OPció!
    TRANSACTION_COST: float = 0.0005
    EPISODE_LENGTH: int = 78  # M5, 1 nap
    DEADZONE: float = 0.1  # -0.1 .. 0.1 = HOLD
    STRIKE_DISTANCE: float = 0.02  # 2% strike
    TP_DISTANCE: float = 0.015  # 1.5% TP
    SL_DISTANCE: float = 0.01  # 1% SL
    PREMIUM_PCT: float = 0.005  # 0.5% prémium


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
        'stocks': [
            # Tech
            'AAPL',  # Apple
            # 'MSFT',  # Microsoft
            # 'IBM',  # IBM
            # 'ORCL',  # Oracle
            # 'GOOGL',  # Google
            # 'TSLA',  # Tesla 2010-09-24
            # 'NVDA',  # Nvidia

            # Blue chips
            # 'JPM',  # JP Morgan
            # 'KO',  # Coca-Cola
            # 'PG',  # Procter & Gamble
            # 'JNJ',  # Johnson & Johnson
            # 'WMT',  # Walmart
            # 'XOM',  # Exxon Mobil
            # 'CVX',  # Chevron
            #
            # # ETF-ek
            # 'SPY',  # S&P 500
            # 'QQQ',  # Nasdaq 100
            # 'IWM'  # Russell 2000
        ],
        'crypto': [
            # 'BTC-USD',  # 2014-11-05
            # 'ETH-USD',  # 2017-12-28
            # 'BNB-USD',  # 2017-12-28
            # 'SOL-USD',  # 2020-05-29
            # 'XRP-USD',  # 2017-12-28
        ],
        'commodities': [
            # 'GC=F',  # Arany
            # 'SI=F',  # Ezüst
            # 'PL=F',  # Platina
            # 'PA=F',  # Palládium
            #
            # # Energia
            # 'CL=F',  # WTI Olaj
            # 'BZ=F',  # Brent Olaj
            # 'NG=F',  # Földgáz
            #
            # # Mezőgazdaság
            # 'ZC=F',  # Kukorica
            # 'ZS=F',  # Szója
            # 'ZW=F',  # Búza
            # 'KC=F',  # Kávé
            # 'CC=F',  # Kakaó
        ],
        'forex': [
            # # Majors
            # 'EUR/USD',
            # 'GBP/USD',
            # 'USD/JPY',
            # 'USD/CHF',
            # 'AUD/USD',
            # 'USD/CAD',
            # 'NZD/USD',
            #
            # # Cross
            # 'EUR/JPY',
            # 'GBP/JPY',
            # 'EUR/GBP'
        ]
    }

    """
    ???
    INDEX_FUTURES = {
    # USA
    'ES=F': 'E-mini S&P 500',        # Legnépszerűbb!
    'NQ=F': 'E-mini Nasdaq 100',     # Tech index
    'YM=F': 'E-mini Dow Jones',      # 30 blue chip
    'RTY=F': 'E-mini Russell 2000',  # Small cap
    
    # Globális
    'NI225=F': 'Nikkei 225 (Japan)',
    'FTSE=F': 'FTSE 100 (UK)',
    'GDAXI=F': 'DAX (Germany)',
    'FCHI=F': 'CAC 40 (France)',
}
    """

    # ======== 30 ÉV BEÁLLÍTÁS ========
    TODAY = datetime.now()

    # Training
    TRAIN_START_DATE = datetime(year=2010, month=4, day=10)  # start of SOL,  TODAY - timedelta(days=5.99 * 365)
    TRAIN_END_DATE = TODAY - timedelta(days=3 * 365)

    # Test
    TEST_START_DATE = datetime(year=2010, month=4, day=10) #  TRAIN_END_DATE + timedelta(days=1)
    TEST_END_DATE = TODAY

    # ======== KRITIKUS: ROLLING WINDOW ========
    DATA_INTERVAL = '1d'  # Csak napi! (30 év intraday irreális)
    EPISODE_LENGTH = 200  # 1 év = 252 trading nap
    NUM_EPISODES = 100  # 100× random 1 éves ablak

    # ======== EGYÉB ========
    INITIAL_BALANCE = 1_000
    MAX_POSITION_SIZE = 0.2  # 0.2
    TRANSACTION_COST = 0.001
    BATCH_SIZE = 128
    MEMORY_SIZE = 20_000

    # Agent training parameters
    RL_CONFIG = {
        'learning_rate': 0.0001,
        'gamma': 0.99,  # Discount factor
        'epsilon_start': 1.0,  # Exploration rate
        'epsilon_min': 0.01,
        'epsilon_decay': 0.999,  # 0.995
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
