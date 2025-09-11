from models.ddpg_agent import DDPG
from environment.stock_env import PortfolioEnv
from utils.helper import *
from tqdm import tqdm

from utils.data_handler import DataLoader

import argparse

"""
Limit all lines to maximum of 79 characters
===============================================================================
"""

def parse_args():
    """
    Parse command line arguments.

    The other arguments not defined in this function are directly passed to
    main.py
    For instance, an option like "--beta 1" is given directly to the main
    script.

    :return: the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_length', type=int, default=50)
    parser.add_argument('--window_period', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--feature_type', type=str, default='ohlcv')
    parser.add_argument('--activation', type=str, default='softmax')
    parser.add_argument('--actor_predictor', type=str, default='mlp')
    parser.add_argument('--critic_predictor', type=str, default='alstm')

    parser.add_argument('--rebalancing', type=int, default=20)
    parser.add_argument('--buy_fee', type=float, default=0.0025)
    parser.add_argument('--sell_fee', type=float, default=0.0025)
    parser.add_argument('--time_cost', type=float, default=0)

    parser.add_argument('--mdd_alpha', type=float, default=0)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--action_bound', type=float, default=1)

    parser.add_argument('--random_start', type=str2bool, default=True)
    parser.add_argument('--episode', type=int, default=1500)
    parser.add_argument('--max_step', type=int, default=3000)
    parser.add_argument('--target_date', type=str, default='20240102')

    parser.add_argument('--train_period', type=int, default=10)
    parser.add_argument('--inverse', type=str2bool, default=False)
    parser.add_argument('--name', type=str, default='us_etf2')
    parser.add_argument('--file_path', type=str)
    return parser.parse_known_args()

def main():
    """
    config
    """
    args, unknown = parse_args()
    window_length = args.window_length
    window_period = args.window_period
    seed = args.seed
    feature_type = args.feature_type
    activation = args.activation
    actor_predictor = args.actor_predictor
    critic_predictor = args.critic_predictor
    rebalancing = args.rebalancing
    buy_fee = args.buy_fee
    sell_fee = args.sell_fee
    time_cost = args.time_cost
    mdd_alpha = args.mdd_alpha
    beta = args.beta
    action_bound = args.action_bound
    random_start = args.random_start
    episode = args.episode
    max_step = args.max_step
    target_date = args.target_date
    train_period = args.train_period
    name = args.name
    file_path = args.file_path

    data_loader = DataLoader(target_date, train_period, window_period, window_length,
                             file_path)
    train_data = data_loader.get_train_data()
    stocks = data_loader.abbreviation
    nb_classes = len(stocks) + 1

    action_dim = [nb_classes] # ex.) when num_stocks:20, length 21 vector
    state_dim = [nb_classes, int(window_length/window_period)]
    num_training_time = data_loader.num_training_time

    agent = DDPG(np.prod(state_dim), np.prod(action_dim), action_bound, seed,
                 feature_type,activation, actor_predictor, critic_predictor,
                 len(stocks),actor_num_layer=actor_num_layer,
                 actor_hidden=actor_hidden,critic_num_layer=critic_num_layer,
                 critic_hidden=critic_hidden)
    env = PortfolioEnv(train_data, len(stocks), num_training_time,
                       num_training_time-window_length-window_period,
                       window_length, window_period, rebalancing,
                       buy_fee, sell_fee, time_cost, mdd_alpha, beta,
                       action_bound, random_start, inverse=inverse)

    # Main training loop
    for _ in range(episode):
        state, info = env.reset()
        if inverse:
            state = apply_inverse_price_tensor(state)
        state = agent.obs_normalizer(state)

        episode_reward = 0
        for step in tqdm(range(max_step)):
            action = agent.select_action(state, agent.noise())
            next_state, reward, done, info = env.step(action)
            if inverse:
                next_state = apply_inverse_price_tensor(next_state)

            next_state = agent.obs_normalizer(next_state)
            agent.add_transition((state, next_state, action, reward, float(done)))
            agent.train()

            state = next_state
            episode_reward += reward
            if done:
                break

        print(f"Episode: {_}, Reward: {episode_reward}, PV: {info['portfolio_value']}, max-Q: {agent.maxQ}")

    # Save the trained model
    actor_path = get_actor_path(name, episode, window_length, window_period,
                                train_period, actor_predictor, critic_predictor,
                                mdd_alpha, beta, action_bound, seed, buy_fee,
                                sell_fee, time_cost, target_date, rebalancing)
    critic_path = get_critic_path(name, episode, window_length, window_period,
                                  train_period, actor_predictor, critic_predictor,
                                  mdd_alpha, beta, action_bound, seed, buy_fee,
                                  sell_fee, time_cost, target_date, rebalancing)
    agent.save_model(actor_path, critic_path)

if __name__ == '__main__':
    main()
