# live_trading/order_manager.py

import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manage order execution and tracking
    """

    def __init__(self, broker_api, config):
        self.broker = broker_api
        self.config = config
        self.pending_orders = []
        self.completed_orders = []

    def execute_actions(self, actions: Dict[str, float], current_prices: Dict[str, float]):
        """
        Convert agent actions to actual orders

        Args:
            actions: Dict mapping asset symbols to action values (-1 to 1)
            current_prices: Dict mapping asset symbols to current prices
        """
        for symbol, action in actions.items():
            if abs(action) < 0.1:
                continue  # Skip hold actions

            if action > 0:  # Buy
                self._execute_buy(symbol, action, current_prices[symbol])
            else:  # Sell
                self._execute_sell(symbol, abs(action), current_prices[symbol])

    def _execute_buy(self, symbol: str, action_strength: float, price: float):
        """
        Execute buy order
        """
        # Calculate quantity based on action strength and available cash
        account_info = self.broker.get_account_info()
        available_cash = account_info.get('cash', 0)

        max_buy_value = available_cash * self.config.MAX_POSITION_SIZE
        buy_value = max_buy_value * action_strength
        quantity = int(buy_value / price)

        if quantity > 0:
            logger.info(f"Executing BUY: {quantity} shares of {symbol} at ${price:.2f}")

            order = self.broker.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type='market',
                side='buy'
            )

            self.pending_orders.append({
                'order_id': order['order_id'],
                'symbol': symbol,
                'quantity': quantity,
                'side': 'buy',
                'timestamp': datetime.now()
            })

    def _execute_sell(self, symbol: str, action_strength: float, price: float):
        """
        Execute sell order
        """
        # Get current position
        positions = self.broker.get_positions()
        position = next((p for p in positions if p['symbol'] == symbol), None)

        if position:
            current_qty = position['quantity']
            sell_qty = int(current_qty * action_strength)

            if sell_qty > 0:
                logger.info(f"Executing SELL: {sell_qty} shares of {symbol} at ${price:.2f}")

                order = self.broker.place_order(
                    symbol=symbol,
                    quantity=sell_qty,
                    order_type='market',
                    side='sell'
                )

                self.pending_orders.append({
                    'order_id': order['order_id'],
                    'symbol': symbol,
                    'quantity': sell_qty,
                    'side': 'sell',
                    'timestamp': datetime.now()
                })

    def check_pending_orders(self):
        """
        Check status of pending orders
        """
        # Placeholder for checking order status
        logger.info(f"Checking {len(self.pending_orders)} pending orders")

    def get_order_history(self) -> List[Dict]:
        """
        Get completed order history
        """
        return self.completed_orders
