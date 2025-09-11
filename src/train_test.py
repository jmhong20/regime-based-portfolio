from environment.stock_env import PortfolioEnv
from utils.helper import *
from tqdm import tqdm
from models.ddpg_agent import DDPG

import argparse
from datetime import datetime

def parse_args():
    """
    Parse command line arguments.

    The other arguments not defined in this function are directly passed to main.py. For instance,
    an option like "--beta 1" is given directly to the main script.

    :return: the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_length', type=int, default=50)
    parser.add_argument('--window_period', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--feature_type', type=str, default='ohlcv')
    parser.add_argument('--rebalancing', type=int, default=20)
    parser.add_argument('--buy_fee', type=float, default=0.0025)
    parser.add_argument('--sell_fee', type=float, default=0.0025)
    parser.add_argument('--time_cost', type=float, default=0)

    parser.add_argument('--random_start', type=str2bool, default=True)
    parser.add_argument('--episode', type=int, default=1500)
    parser.add_argument('--max_step', type=int, default=3000)

    parser.add_argument('--target_date', type=str, default='20240102')
    parser.add_argument('--train_period', type=int, default=10)
    parser.add_argument('--country', type=str, default='us')
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
    rebalancing = args.rebalancing
    buy_fee = args.buy_fee
    sell_fee = args.sell_fee
    time_cost = args.time_cost
    random_start = args.random_start
    episode = args.episode
    max_step = args.max_step
    target_date = args.target_date
    train_period = args.train_period
    country = args.country

    if country == "us":
        data_path = '/home/jmhong20/rlpm_ete/data/us/snp500_data.csv'
        numpy_path = '/home/jmhong20/rlpm_ete/data/us/snp500_data.npy'

    history_df = pd.read_csv(data_path)
    print(len(history_df['ticker'].unique()))
    total_len = len(history_df['date'].unique())
    history = np.load(numpy_path)
    history_index = np.load(numpy_path.replace('.npy', '_index.npy'))

    end_date = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"
    start_date_d = datetime.strptime(end_date, "%Y-%m-%d").replace(year=datetime.strptime(end_date, "%Y-%m-%d").year - train_period)
    start_date = start_date_d.strftime("%Y-%m-%d")

    history_df_start = history_df[history_df['date'] >= start_date]
    start_index = total_len - len(history_df_start['date'].unique())
    history_df_end = history_df[history_df['date'] < end_date]
    end_index = len(history_df_end['date'].unique())

    history = history[:,start_index:end_index,:]
    history_index = history_index[:,start_index:end_index,:]

    print(history.shape)
    print(history_index.shape)
    # exit()
    train_data = np.concatenate([history, history_index], axis=0)

    num_training_time = end_index - start_index
    stocks = history_df['ticker'].unique()
    nb_classes = len(stocks)

    if feature_type == 'ohlcv':
        num_features = 5
    agent = DDPG(seed, nb_classes, nb_classes-1, window_length//window_period, num_features)
    env = PortfolioEnv(train_data, len(stocks)-1, num_training_time, num_training_time-window_length-window_period, window_length, window_period, rebalancing, buy_fee, sell_fee, time_cost, random_start)

    for _ in range(episode):
        state, info = env.reset()
        # print("RESET:", state[-1,0,-1], state[-1,-1,-1])
        state = agent.obs_normalizer(state)

        episode_reward = 0
        verbose = False
        # for step in tqdm(range(max_step)):
        for step in range(max_step):
        # for step in tqdm(range(1)):
            # print(state.shape)
            if step % 10 == 0:
                verbose = True
            action, portfolio_weights = agent.select_action(state, verbose=verbose, noise=agent.noise())
            verbose = False
            # print(action)
            # action = np.array([1.0] + [0.0 for __ in range(len(action)-1)])
            next_state, reward, done, info = env.step(action, portfolio_weights)
            next_state = agent.obs_normalizer(next_state)
            agent.add_transition((state, next_state, action, reward, float(done)))
            agent.train(training_phase='phase1')

            state = next_state
            episode_reward += reward
            if done:
                break

        print(f"Episode: {_}, Reward: {episode_reward}, PV: {info['portfolio_value']}, max-Q: {agent.maxQ}", flush=True)
        print(agent.replay_buffer.size() // agent.replay_buffer.batch_size)
        for batch_count in range(agent.replay_buffer.size() // agent.replay_buffer.batch_size):
            agent.train(training_phase='phase2')

    actor_path = get_actor_path(country, episode, window_length, window_period, train_period, seed, buy_fee, sell_fee, time_cost, target_date, rebalancing)
    critic_path = get_critic_path(country, episode, window_length, window_period, train_period, seed, buy_fee, sell_fee, time_cost, target_date, rebalancing)
    agent.save_model(actor_path, critic_path)

if __name__ == '__main__':
    main()