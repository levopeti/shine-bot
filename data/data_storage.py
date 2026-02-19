# data/data_storage.py

import pickle
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Handle data caching and storage to avoid repeated API calls
    """

    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def save_data(self, data, filename):
        """
        Save data to pickle file
        """
        filepath = os.path.join(self.cache_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to {filepath}")

    def load_data(self, filename):
        """
        Load data from pickle file
        """
        filepath = os.path.join(self.cache_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded from {filepath}")
            return data
        else:
            logger.warning(f"File {filepath} not found")
            return None

    def is_cached(self, filename, max_age_hours=24):
        """
        Check if cached data exists and is recent enough
        """
        filepath = os.path.join(self.cache_dir, filename)
        if not os.path.exists(filepath):
            return False

        # Check file age
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        age_hours = (datetime.now() - file_time).total_seconds() / 3600

        return age_hours < max_age_hours
