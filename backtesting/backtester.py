# backtesting/backtester.py

import numpy as np
import pandas as pd
from typing import Dict, List
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting engine for evaluating trading strategies
    """

    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.results = None

    def run_backtest(self, num_episodes: int = 1) -> Dict:
        """
        Run backtest for specified number of episodes
        """
        logger.info("Starting backtest...")

        # Set agent to evaluation mode
        self.agent.set_eval_mode()

        all_portfolio_values = []
        all_returns = []
        all_trade_histories = []

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False

            episode_portfolio_values = [self.config.INITIAL_BALANCE]

            while not done:
                # Get action from agent
                action = self.agent.select_action(state)

                # Execute action
                next_state, reward, done, info = self.env.step(action)

                # Record portfolio value
                episode_portfolio_values.append(info['portfolio_value'])

                state = next_state

            # Store results
            all_portfolio_values.append(episode_portfolio_values)
            all_returns.append(info['return'])
            all_trade_histories.append(self.env.trade_history)

            logger.info(f"Episode {episode + 1}/{num_episodes} - Final Return: {info['return']*100:.2f}%")

        # Calculate metrics
        self.results = self._calculate_metrics(
            all_portfolio_values,
            all_returns,
            all_trade_histories
        )

        logger.info("Backtest completed!")
        return self.results

    def _calculate_metrics(self, portfolio_values: List, returns: List, trade_histories: List) -> Dict:
        """
        Calculate performance metrics
        """
        # Average portfolio values across episodes
        avg_portfolio_values = np.mean(portfolio_values, axis=0)

        # Calculate returns
        final_value = avg_portfolio_values[-1]
        initial_value = self.config.INITIAL_BALANCE
        total_return = (final_value - initial_value) / initial_value

        # Daily returns
        daily_returns = np.diff(avg_portfolio_values) / avg_portfolio_values[:-1]

        # Sharpe ratio (annualized)
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        cumulative_returns = avg_portfolio_values / initial_value
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Win rate
        positive_returns = np.sum(np.array(daily_returns) > 0)
        total_days = len(daily_returns)
        win_rate = positive_returns / total_days if total_days > 0 else 0

        # Total trades
        total_trades = sum(len(th) for th in trade_histories)

        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_portfolio_value': final_value,
            'total_trades': total_trades,
            'portfolio_values': avg_portfolio_values,
            'daily_returns': daily_returns
        }

        return metrics

    def print_results(self):
        """
        Print backtest results
        """
        if self.results is None:
            logger.warning("No results to print. Run backtest first.")
            return

        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Return: {self.results['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.results['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {self.results['win_rate']*100:.2f}%")
        print(f"Final Portfolio Value: ${self.results['final_portfolio_value']:.2f}")
        print(f"Total Trades: {self.results['total_trades']}")
        print("="*60 + "\n")
