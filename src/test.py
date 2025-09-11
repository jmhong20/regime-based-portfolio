from models.ddpg_agent import DDPG
from environment.stock_env import PortfolioEnv
from utils.helper import *
from tqdm import tqdm
import os
import pandas as pd

from utils.data_handler import DataLoader

import argparse

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
    parser.add_argument('--action_show', type=str2bool, default=True)
    parser.add_argument('--inverse', type=str2bool, default=False)
    parser.add_argument('--name', type=str, default='us_etf2')
    parser.add_argument('--file_path', type=str, default="data/{}_{}.pkl".format("etf", "20250428"))

    parser.add_argument('--actor_num_layer', type=int, default=1)
    parser.add_argument('--actor_hidden', type=int, default=32)
    parser.add_argument('--critic_num_layer', type=int, default=1)
    parser.add_argument('--critic_hidden', type=int, default=32)

    parser.add_argument('--performance_check', type=str2bool, default=False)
    return parser.parse_known_args()

def main():
    """
    config
    """
    args, unknown = parse_args()
    print(args)
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
    action_show = args.action_show
    name = args.name
    file_path = args.file_path
    inverse = args.inverse

    actor_num_layer = args.actor_num_layer
    actor_hidden = args.actor_hidden
    critic_num_layer = args.critic_num_layer
    critic_hidden = args.critic_hidden

    performance_check = args.performance_check

    if not (actor_hidden == critic_hidden == 32 and actor_num_layer == critic_num_layer == 1):
        name = f"{name}_{actor_hidden}_{actor_num_layer}_{critic_hidden}_{critic_num_layer}"

    data_loader = DataLoader(target_date, train_period, window_period, window_length, file_path)
    stocks = data_loader.abbreviation
    nb_classes = len(stocks) + 1
    action_dim = [nb_classes] # ex.) when num_stocks:20, length 21 vector
    state_dim = [nb_classes, int(window_length/window_period)]

    agent = DDPG(np.prod(state_dim), np.prod(action_dim), action_bound, seed, feature_type, activation, actor_predictor, critic_predictor, len(stocks), actor_num_layer=actor_num_layer, actor_hidden=actor_hidden, critic_num_layer=critic_num_layer, critic_hidden=critic_hidden)
    actor_path = get_actor_path(name, episode, window_length, window_period, train_period, actor_predictor, critic_predictor, mdd_alpha, beta, action_bound, seed, buy_fee, sell_fee, time_cost, target_date, rebalancing)
    critic_path = get_critic_path(name, episode, window_length, window_period, train_period, actor_predictor, critic_predictor, mdd_alpha, beta, action_bound, seed, buy_fee, sell_fee, time_cost, target_date, rebalancing)
    agent.load_model(actor_path, critic_path)
    agent.actor.eval()

    # Sanity Check (Check overfitting)
    train_data = data_loader.get_train_data()
    num_training_time = data_loader.num_training_time
    env = PortfolioEnv(train_data, len(stocks), num_training_time, num_training_time-window_length-window_period, window_length, window_period, rebalancing, buy_fee, sell_fee, time_cost, mdd_alpha, beta, action_bound, random_start, inverse=inverse)
    for _ in range(1):
        state, info = env.reset()
        if inverse:
            state = apply_inverse_price_tensor(state)
        state = agent.obs_normalizer(state)

        episode_reward = 0
        for step in tqdm(range(max_step)):
            action = agent.select_action(state) #+ agent.noise()
            next_state, reward, done, info = env.step(action)
            if inverse:
                next_state = apply_inverse_price_tensor(next_state)
            next_state = agent.obs_normalizer(next_state)
            agent.add_transition((state, next_state, action, reward, float(done)))
            
            # agent.train()

            state = next_state
            episode_reward += reward
            if done:
                break

        print(f"Episode: {_}, Reward: {episode_reward}, PV: {info['portfolio_value']}")

    # Real backtest
    if performance_check:
        start_pos = [sp for sp in range(rebalancing + 1)]
    else:
        start_pos = [0]
    result_pv = []

    for sp in start_pos:
        print(f"START POS: {sp}")
        test_data = data_loader.get_test_data()
        # print(test_data)
        # breakpoint()
        num_training_time = len(test_data[0][:, -1])
        env = PortfolioEnv(test_data, len(stocks), num_training_time, num_training_time-window_length-window_period, window_length, window_period, rebalancing, buy_fee, sell_fee, time_cost, mdd_alpha, beta, action_bound, random_start, inverse=inverse, start_pos=sp, performance_check=performance_check)
        for _ in range(1):
            state, info = env.reset()
            if inverse:
                state = apply_inverse_price_tensor(state)
            # print action
            if action_show:
                print(state[-1][0][-1], state[-1][-1][-1])
            # print action
            state = agent.obs_normalizer(state)

            episode_reward = 0
            for step in tqdm(range(max_step)):
                action = agent.select_action(state) #+ agent.noise()
                # action = np.array([1/len(action) for __ in range(len(action))])

                # print action
                if action_show:
                    stock_names = ['현금']
                    for code in data_loader.abbreviation:
                        if name.split('_')[0] == 'kr':
                            stock_names.append(kr_code_to_name(code))
                        else:
                            stock_names.append(code)
                    result = {stock_names[i]: round(action[i], 4) for i in range(len(action))}
                    result_df = pd.DataFrame([result])
                    pd.set_option('display.unicode.east_asian_width', True)
                    print(result_df)
                # print action

                next_state, reward, done, info = env.step(action)
                if inverse:
                    next_state = apply_inverse_price_tensor(next_state)

                # print action
                if action_show:
                    print(f"PV: {info['portfolio_value']}, MDD: {info['maximum_drawdown']}")
                    print()
                # print action

                # print action
                if action_show:
                    print(next_state[-1][0][-1], next_state[-1][-1][-1])
                # print action

                next_state = agent.obs_normalizer(next_state)
                agent.add_transition((state, next_state, action, reward, float(done)))
                
                # agent.train()

                state = next_state
                episode_reward += reward
                if done:
                    break

            print(f"Episode: {_}, Reward: {episode_reward}, PV: {info['portfolio_value']}, MDD: {info['maximum_drawdown']}, BENCH: {info['market_value']}, BENCH_MDD: {info['maximum_drawdown_bench']}")

        result_pv.append(info['portfolio_value'])
        # backtest_save_path = '{}_result.csv'.format(name)
    backtest_save_path = '{}_result.csv'.format("us_stocks2_lstm_lstm_20250605")

    new_row = {
        'anl': actor_num_layer,
        'ah': actor_hidden,
        'cnl': critic_num_layer,
        'ch': critic_hidden,
        'mean_pv': np.mean(np.array(result_pv)),
        'best_pv': np.max(np.array(result_pv)),
        'worst_pv': np.min(np.array(result_pv)),
        'std': np.std(np.array(result_pv))
    }
    new_row_df = pd.DataFrame([new_row])
    if os.path.exists(backtest_save_path):
        new_row_df.to_csv(backtest_save_path, mode='a', index=False, header=False)
    else:
        new_row_df.to_csv(backtest_save_path, header=['anl','ah','cnl','ch','mean_pv','best_pv','worst_pv','std'], mode='a', index=False)
    print(f"mean: {np.mean(np.array(result_pv))}")
    print(f"best: {np.max(np.array(result_pv))}")
    print(f"worst: {np.min(np.array(result_pv))}")
    print(f"std: {np.std(np.array(result_pv))}")

if __name__ == '__main__':
    main()