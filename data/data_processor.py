# data/data_processor.py

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Preprocess and engineer features from market data
    """

    def __init__(self):
        self.scaler = StandardScaler()

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) """
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()

        # Price changes
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        return df

    def process_multi_asset_data(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine and process data from multiple asset classes
        Create a unified dataframe with all features
        """
        processed_dfs = list()

        for asset_class, data in all_data.items():
            if data.empty:
                continue

            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Process each symbol
                for symbol in data.columns.levels[0]:
                    try:
                        symbol_data = data[symbol].copy()
                        symbol_data = self.calculate_technical_indicators(symbol_data)

                        # Add prefix to columns
                        symbol_data = symbol_data.add_prefix(f"{asset_class}_{symbol}_")
                        processed_dfs.append(symbol_data)
                    except Exception as e:
                        logger.warning(f"Error processing {symbol}: {e}")
            else:
                # Single symbol
                data = self.calculate_technical_indicators(data)
                data = data.add_prefix(f"{asset_class}_")
                processed_dfs.append(data)

        # Combine all dataframes
        if processed_dfs:
            combined_df = pd.concat(processed_dfs, axis=1)

            # Forward fill missing values
            combined_df = combined_df.ffill()

            # Drop remaining NaN rows
            combined_df = combined_df.dropna()

            return combined_df
        else:
            return pd.DataFrame()

    def create_state_features(self, df: pd.DataFrame, lookback_window: int = 10) -> np.ndarray:
        """
        Create state features for reinforcement learning
        Uses a lookback window to capture temporal patterns
        """
        features = []

        for i in range(lookback_window, len(df)):
            window_data = df.iloc[i-lookback_window:i].values.flatten()
            features.append(window_data)

        features = np.array(features)

        # Normalize features
        features = self.scaler.fit_transform(features)

        return features

    @staticmethod
    def split_data(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Split data into training and testing sets """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        return train_df, test_df
