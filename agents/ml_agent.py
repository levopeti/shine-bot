# agents/ml_agent.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logger = logging.getLogger(__name__)


class MLAgent:
    """
    Machine Learning agent using classical ML algorithms
    (Random Forest, Gradient Boosting)
    """

    def __init__(self, state_dim: int, action_dim: int, config, model_type='random_forest'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.model_type = model_type

        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        self.training_labels = []

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action using trained ML model
        """
        if not self.is_trained:
            # Random action if not trained
            return np.random.uniform(-1, 1, self.action_dim)

        # Scale state
        state_scaled = self.scaler.transform(state.reshape(1, -1))

        # Predict action class
        action_class = self.model.predict(state_scaled)[0]

        # Convert class to action values
        action = self._class_to_action(action_class)

        return action

    def _class_to_action(self, action_class: int) -> np.ndarray:
        """
        Convert action class to continuous action values
        0: sell, 1: hold, 2: buy
        """
        actions = np.zeros(self.action_dim)

        if action_class == 0:  # Sell
            actions[:] = -0.5
        elif action_class == 2:  # Buy
            actions[:] = 0.5
        # else: hold (keep zeros)

        return actions

    def store_training_data(self, state, action, reward):
        """
        Store data for supervised training
        """
        # Convert action to class label
        action_class = self._action_to_class(action, reward)

        self.training_data.append(state)
        self.training_labels.append(action_class)

    def _action_to_class(self, action: np.ndarray, reward: float) -> int:
        """
        Convert action and reward to class label
        """
        if reward > 0:
            if np.mean(action) > 0.1:
                return 2  # Buy was good
            elif np.mean(action) < -0.1:
                return 0  # Sell was good
        else:
            if np.mean(action) > 0.1:
                return 0  # Buy was bad, should have sold
            elif np.mean(action) < -0.1:
                return 2  # Sell was bad, should have bought

        return 1  # Hold

    def train_model(self):
        """
        Train the ML model on collected data
        """
        if len(self.training_data) < 100:
            logger.warning("Not enough training data")
            return 0.0

        X = np.array(self.training_data)
        y = np.array(self.training_labels)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Calculate accuracy
        accuracy = self.model.score(X_scaled, y)
        logger.info(f"Model trained with accuracy: {accuracy:.2f}")

        return accuracy

    def save_model(self, path: str):
        """
        Save model to file
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'model_type': self.model_type
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"ML model saved to {path}")

    def load_model(self, path: str):
        """
        Load model from file
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.model_type = model_data['model_type']

        logger.info(f"ML model loaded from {path}")

    def set_eval_mode(self):
        """
        Set to evaluation mode
        """
        pass

    def set_train_mode(self):
        """
        Set to training mode
        """
        pass
