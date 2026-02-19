# data/data_fetcher.py

import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from binance.client import Client
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetch market data from multiple sources (stocks, crypto, commodities, forex)
    """

    def __init__(self, config):
        self.config = config
        # self.alpha_vantage = TimeSeries(key=config.ALPHA_VANTAGE_API_KEY, output_format='pandas')
        # self.binance_client = Client(config.BINANCE_API_KEY, config.BINANCE_SECRET_KEY) if config.BINANCE_API_KEY else None

    def fetch_stocks(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch stock data using yfinance
        """
        logger.info(f"Fetching stock data for {symbols}")
        data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker', progress=False)
        return data

    def fetch_crypto(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch cryptocurrency data using yfinance (alternative: Binance API)
        """
        logger.info(f"Fetching crypto data for {symbols}")
        # Convert symbols to yfinance format (e.g., BTC/USD -> BTC-USD)
        yf_symbols = [s.replace('/', '-') for s in symbols]
        data = yf.download(yf_symbols, start=start_date, end=end_date, group_by='ticker', progress=False)
        return data

    def fetch_commodities(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch commodity futures data
        """
        logger.info(f"Fetching commodity data for {symbols}")
        data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker', progress=False)
        return data

    def fetch_forex(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch forex data
        """
        logger.info(f"Fetching forex data for {symbols}")
        # Convert to yfinance format (EUR/USD -> EURUSD=X)
        yf_symbols = [s.replace('/', '') + '=X' for s in symbols]
        data = yf.download(yf_symbols, start=start_date, end=end_date, group_by='ticker', progress=False)
        return data

    def fetch_all_market_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all asset classes
        Returns a dictionary with keys: 'stocks', 'crypto', 'commodities', 'forex'
        """
        all_data = {}

        try:
            all_data['stocks'] = self.fetch_stocks(
                self.config.ASSETS['stocks'], start_date, end_date
            )
        except Exception as e:
            logger.error(f"Error fetching stocks: {e}")
            all_data['stocks'] = pd.DataFrame()

        try:
            all_data['crypto'] = self.fetch_crypto(
                self.config.ASSETS['crypto'], start_date, end_date
            )
        except Exception as e:
            logger.error(f"Error fetching crypto: {e}")
            all_data['crypto'] = pd.DataFrame()

        try:
            all_data['commodities'] = self.fetch_commodities(
                self.config.ASSETS['commodities'], start_date, end_date
            )
        except Exception as e:
            logger.error(f"Error fetching commodities: {e}")
            all_data['commodities'] = pd.DataFrame()

        try:
            all_data['forex'] = self.fetch_forex(
                self.config.ASSETS['forex'], start_date, end_date
            )
        except Exception as e:
            logger.error(f"Error fetching forex: {e}")
            all_data['forex'] = pd.DataFrame()

        return all_data
