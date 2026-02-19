# main.py

import os
import argparse
import logging
from datetime import datetime

# Import framework components
from config.settings import Config
from data.data_fetcher import DataFetcher
from data.data_processor import DataProcessor
from environment.trading_env import TradingEnvironment
from agents.dl_agent import DeepLearningAgent
from backtesting.backtester import Backtester
from visualization.dashboard import TradingDashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_agent(config, num_episodes=100):
    """
    Train the trading agent using reinforcement learning
    """
    logger.info("="*60)
    logger.info("TRAINING MODE")
    logger.info("="*60)

    # Fetch data
    logger.info("Fetching market data...")
    data_fetcher = DataFetcher(config)
    all_data = data_fetcher.fetch_all_market_data(
        config.TRAIN_START_DATE,
        config.TRAIN_END_DATE
    )

    # Process data
    logger.info("Processing data...")
    processor = DataProcessor()
    processed_data = processor.process_multi_asset_data(all_data)

    if processed_data.empty:
        logger.error("No data available for training!")
        return

    logger.info(f"Processed data shape: {processed_data.shape}")

    # Create environment
    logger.info("Creating trading environment...")
    env = TradingEnvironment(processed_data, config, mode='train')

    # Create agent
    logger.info("Initializing agent...")
    agent = DeepLearningAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=config
    )

    # Training loop
    logger.info(f"Starting training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_loss = 0
        steps = 0

        while not done:
            # Select action
            action = agent.select_action(state)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train agent
            loss = agent.train_step()

            episode_reward += reward
            episode_loss += loss
            steps += 1
            state = next_state

        # Log progress
        avg_loss = episode_loss / steps if steps > 0 else 0
        logger.info(
            f"Episode {episode+1}/{num_episodes} - "
            f"Return: {info['return']*100:.2f}% - "
            f"Portfolio: ${info['portfolio_value']:.2f} - "
            f"Trades: {info['total_trades']} - "
            f"Epsilon: {agent.epsilon:.3f} - "
            f"Avg Loss: {avg_loss:.4f}"
        )

        # Save checkpoint every 10 episodes
        if (episode + 1) % 10 == 0:
            model_path = os.path.join(
                config.MODEL_DIR,
                f"agent_episode_{episode+1}.pth"
            )
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            agent.save_model(model_path)

    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, "agent_final.pth")
    agent.save_model(final_model_path)

    logger.info("Training completed!")
    return agent, env


def backtest_agent(config, model_path=None):
    """
    Backtest the trained agent on historical data
    """
    logger.info("="*60)
    logger.info("BACKTESTING MODE")
    logger.info("="*60)

    # Fetch data
    logger.info("Fetching market data...")
    data_fetcher = DataFetcher(config)
    all_data = data_fetcher.fetch_all_market_data(
        config.TEST_START_DATE,
        config.TEST_END_DATE
    )

    # Process data
    logger.info("Processing data...")
    processor = DataProcessor()
    processed_data = processor.process_multi_asset_data(all_data)

    if processed_data.empty:
        logger.error("No data available for backtesting!")
        return

    logger.info(f"Processed data shape: {processed_data.shape}")

    # Create environment
    logger.info("Creating trading environment...")
    env = TradingEnvironment(processed_data, config, mode='test')

    # Create and load agent
    logger.info("Loading trained agent...")
    agent = DeepLearningAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=config
    )

    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
    else:
        logger.warning("No model found! Using random agent.")

    # Run backtest
    backtester = Backtester(env, agent, config)
    results = backtester.run_backtest(num_episodes=1)
    backtester.print_results()

    return results, processed_data


def run_live_trading(config, model_path):
    """
    Run live trading with real broker API (placeholder)
    """
    logger.info("="*60)
    logger.info("LIVE TRADING MODE")
    logger.info("="*60)
    logger.warning("Live trading not fully implemented. Use with caution!")

    # This would integrate with real broker APIs
    # Example: Interactive Brokers, Alpaca, etc.
    pass


def run_dashboard(config, backtest_results, data):
    """
    Run visualization dashboard
    """
    logger.info("Starting dashboard...")

    dashboard = TradingDashboard(config)

    # You can add callbacks here to update dashboard with real-time data

    dashboard.run(debug=True)


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description="AI Trading Framework")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'backtest', 'live', 'dashboard'],
        required=True,
        help='Execution mode'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to saved model for backtesting/live trading'
    )

    args = parser.parse_args()

    # Load configuration
    config = Config()

    if args.mode == 'train':
        train_agent(config, num_episodes=args.episodes)

    elif args.mode == 'backtest':
        model_path = args.model_path or os.path.join(config.MODEL_DIR, "agent_final.pth")
        results, data = backtest_agent(config, model_path)

    elif args.mode == 'live':
        if not args.model_path:
            logger.error("Model path required for live trading!")
            return
        run_live_trading(config, args.model_path)

    elif args.mode == 'dashboard':
        # Run dashboard with sample data
        model_path = args.model_path or os.path.join(config.MODEL_DIR, "agent_final.pth")
        results, data = backtest_agent(config, model_path)
        run_dashboard(config, results, data)


if __name__ == "__main__":
    main()
