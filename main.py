# main.py - WITH AGENT SELECTION

import os
import argparse
import logging
from datetime import datetime

from config.settings import Config
from data.data_fetcher import DataFetcher
from data.data_processor import DataProcessor
from environment.trading_env import TradingEnvironment

# Import all agent types
from agents.dl_agent import DeepLearningAgent
from agents.ml_agent import MLAgent
from agents.llm_agent import LLMAgent
from strategies.ensemble_strategy import EnsembleStrategy

from backtesting.backtester import Backtester
from visualization.dashboard import TradingDashboard

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_agent(agent_type: str, state_dim: int, action_dim: int, config):
    """Create agent based on type: 'dl', 'ml_rf', 'ml_gb', 'llm', 'ensemble'"""
    logger.info(f"Creating agent: {agent_type}")

    if agent_type in ['dl', 'dqn']:
        return DeepLearningAgent(state_dim, action_dim, config)
    elif agent_type == 'ml_rf':
        return MLAgent(state_dim, action_dim, config, model_type='random_forest')
    elif agent_type == 'ml_gb':
        return MLAgent(state_dim, action_dim, config, model_type='gradient_boosting')
    elif agent_type == 'llm':
        return LLMAgent(state_dim, action_dim, config)
    elif agent_type == 'ensemble':
        agents = [
            DeepLearningAgent(state_dim, action_dim, config),
            MLAgent(state_dim, action_dim, config, model_type='random_forest'),
            MLAgent(state_dim, action_dim, config, model_type='gradient_boosting')
        ]
        return EnsembleStrategy(agents, weights=[0.5, 0.25, 0.25])
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train_agent(config, agent_type='dl', num_episodes=100):
    logger.info("="*60)
    logger.info(f"TRAINING - Agent: {agent_type}")
    logger.info("="*60)

    data_fetcher = DataFetcher(config)
    all_data = data_fetcher.fetch_all_market_data(config.TRAIN_START_DATE, config.TRAIN_END_DATE)

    processor = DataProcessor()
    processed_data = processor.process_multi_asset_data(all_data)

    if processed_data.empty:
        logger.error("No data!")
        return

    env = TradingEnvironment(processed_data, config, mode='train')
    agent = create_agent(agent_type, env.observation_space.shape[0], env.action_space.shape[0], config)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_loss = 0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            if hasattr(agent, 'store_transition'):
                agent.store_transition(state, action, reward, next_state, done)
            if hasattr(agent, 'store_training_data'):
                agent.store_training_data(state, action, reward)
            if hasattr(agent, 'train_step'):
                episode_loss += agent.train_step()

            steps += 1
            state = next_state

        if hasattr(agent, 'train_model') and (episode + 1) % 10 == 0:
            agent.train_model()

        avg_loss = episode_loss / steps if steps > 0 else 0
        logger.info(
            f"Ep {episode+1}/{num_episodes} - Return: {info['return']*100:.2f}% - "
            f"Portfolio: ${info['portfolio_value']:.2f} - Trades: {info['total_trades']} - Loss: {avg_loss:.4f}"
        )

        if (episode + 1) % 10 == 0:
            model_path = os.path.join(config.MODEL_DIR, f"{agent_type}_agent_ep{episode+1}.pth")
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            if hasattr(agent, 'save_model'):
                agent.save_model(model_path)

    final_path = os.path.join(config.MODEL_DIR, f"{agent_type}_agent_final.pth")
    if hasattr(agent, 'save_model'):
        agent.save_model(final_path)

    logger.info("Training completed!")
    return agent, env


def backtest_agent(config, agent_type='dl', model_path=None):
    logger.info("="*60)
    logger.info(f"BACKTESTING - Agent: {agent_type}")
    logger.info("="*60)

    data_fetcher = DataFetcher(config)
    all_data = data_fetcher.fetch_all_market_data(config.TEST_START_DATE, config.TEST_END_DATE)

    processor = DataProcessor()
    processed_data = processor.process_multi_asset_data(all_data)

    if processed_data.empty:
        logger.error("No data!")
        return

    env = TradingEnvironment(processed_data, config, mode='test')
    agent = create_agent(agent_type, env.observation_space.shape[0], env.action_space.shape[0], config)

    if model_path and os.path.exists(model_path):
        if hasattr(agent, 'load_model'):
            agent.load_model(model_path)
    else:
        default_path = os.path.join(config.MODEL_DIR, f"{agent_type}_agent_final.pth")
        if os.path.exists(default_path) and hasattr(agent, 'load_model'):
            agent.load_model(default_path)
        else:
            logger.warning("No model found! Using untrained agent.")

    backtester = Backtester(env, agent, config)
    results = backtester.run_backtest(num_episodes=1)
    backtester.print_results()

    return results, processed_data


def run_dashboard(config, agent_type='dl', model_path=None):
    logger.info("Starting dashboard...")
    results, data = backtest_agent(config, agent_type, model_path)
    dashboard = TradingDashboard(config)
    dashboard.run(debug=True)


def main():
    parser = argparse.ArgumentParser(
        description="AI Trading Framework",
        epilog="""
Examples:
  python main.py --mode train --agent dl --episodes 100
  python main.py --mode train --agent ml_rf --episodes 50
  python main.py --mode backtest --agent ensemble
        """
    )

    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'dashboard'], required=True)
    parser.add_argument('--agent', type=str, choices=['dl', 'dqn', 'ml_rf', 'ml_gb', 'llm', 'ensemble'], default='dl')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--model-path', type=str, default=None)

    args = parser.parse_args()
    config = Config()

    if args.mode == 'train':
        train_agent(config, agent_type=args.agent, num_episodes=args.episodes)
    elif args.mode == 'backtest':
        backtest_agent(config, agent_type=args.agent, model_path=args.model_path)
    elif args.mode == 'dashboard':
        run_dashboard(config, agent_type=args.agent, model_path=args.model_path)


if __name__ == "__main__":
    main()
