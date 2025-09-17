import random
import numpy as np
from utils.helper import *

eps = 1e-8

class DataGenerator:
    def __init__(self, history, num_training_time, window_length=50, window_period=5, rebalancing=20, random_start=True, start_pos=0, performance_check=False):
        self._data = history.copy()  # all data
        self.data = self._data
        self.idx = 0
        self.start_pos = start_pos
        self.step = 0
        self.window_length = window_length
        self.window_period = window_period
        self.rebalancing = rebalancing
        self.performance_check = performance_check

        self.max_steps = num_training_time - window_length - window_period - rebalancing + 1
        self.random_start = random_start

    def step_(self):
        self.step += self.rebalancing
        obs = self.data[:, self.step-self.window_period:self.step + self.window_length, :][:, ::-1, :][:, ::self.window_period, :][:, ::-1, :].copy()
        # print(self.step-self.window_period, self.step + self.window_length, ": ", self.max_steps, self._data.shape[1])
        # print(self.step)
        next_day = self.data[:, self.step-(self.rebalancing-self.window_length):self.step + self.window_length + 1, :].copy()
        done = self.step+self.rebalancing >= self.max_steps
        return obs, done, next_day

    def reset(self):
        self.step = self.window_period

        if self.random_start:
            self.idx = random.randint(self.window_length + 1, self.window_length + self.rebalancing + 1)
        else:
            self.idx = self.window_length + 1
            if self.window_period == 1: 
                self.idx -= 1

        if self.performance_check:
            self.idx += self.start_pos
        data = self._data[:, self.idx - self.window_length:self.idx + self.max_steps + 1, :]
        self.data = data
        return self.data[:, self.step-self.window_period:self.step + self.window_length, :][:, ::-1, :][:, ::self.window_period, :][:, ::-1, :].copy()

class PortfolioSim:
    def __init__(self, num_assets, buy_fee, sell_fee, time_cost, steps, mdd_alpha=0, beta=0):
        self.num_assets = num_assets
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.time_cost = time_cost
        self.steps = steps
        self.mdd_alpha = mdd_alpha
        self.beta = beta
        self.p0 = 0
        self.infos = []

    def step(self, w1, y1):
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        # assert y1[0] == 1.0, 'y1[0] must be 1'

        w0 = self.w0
        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into

        # x_t_prime, mu1 = calculate_transaction_fee(w_t=dw1, w_t_1=w1, f_sell=0.0025, f_buy=0)
        x_t_prime, mu1 = calculate_transaction_fee(w_t=dw1, w_t_1=w1, f_sell=self.sell_fee, f_buy=self.buy_fee)
        assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = self.p0 * (1 - mu1) * np.dot(y1, w1)  # (eq11) final portfolio value
        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = p1 / self.p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (self.p0 + eps))  # log rate of return
        # add mdd to reward
        r1_history = np.append(np.array([x['portfolio_value'] for x in self.infos]), p1)
        if len(r1_history[:-1]) != 0:
            mdd_old = max_drawdown(r1_history[:-1])
        else:
            mdd_old = 0
        
        benchmark_history = np.append(np.array([x['return'] for x in self.infos]), y1.mean())
        if len(benchmark_history[:-1]) != 0:
            mdd_old_bench = max_drawdown(benchmark_history[:-1])
        else:
            mdd_old_bench = 0
        try:
            mdd = max_drawdown(np.array([1+x for x in calculate_returns(r1_history)]))
            mdd_bench = max_drawdown(benchmark_history)
        except:
            mdd = 0
            mdd_bench = 0

        reward = r1  # (22) logarithmic accumulated return
        reward = reward - self.mdd_alpha * ( abs(mdd) - abs(mdd_old) ) - self.beta * 0
        # remember for next step
        self.p0 = p1
        self.w0 = w1

        # if we run out of money, we're done (losing all the money)
        done = p1 == 0

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "maximum_drawdown": mdd,
            "maximum_drawdown_bench": mdd_bench,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "individual": y1,
            "cost": mu1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.p0 = 1.0
        self.w0 = np.array([1.0] + [0.0] * self.num_assets)

class PortfolioEnv:
    def __init__(self, history, num_assets, num_training_time, steps, window_length, window_period, rebalancing, buy_fee, sell_fee, time_cost, random_start, start_pos=0, performance_check=False):
        self.num_assets = num_assets
        self.window_length = window_length
        self.window_period = window_period

        self.random_start = random_start

        self.src = DataGenerator(history, num_training_time, window_length, window_period, rebalancing, random_start, start_pos=start_pos, performance_check=performance_check)
        self.sim = PortfolioSim(num_assets, buy_fee, sell_fee, time_cost, steps)

        self.infos = []

    def step(self, action):
        np.testing.assert_almost_equal(action.shape,(self.num_assets + 1,))

        # normalise just in case
        # action = np.clip(action, 0, 1)
        weights = action
        weights /= (weights.sum() + eps)
        # weights[0] += np.clip(1 - weights.sum(), 0, 1)
        weights[0] += (1 - weights.sum())

        assert ((weights >= 0) * (weights <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        self.w0 = weights
        observation, done1, next_day = self.src.step_()
        # print(observation[0][:, -1])
        
        # cash_observation = np.ones((1, int(self.window_length / self.window_period)+1, observation.shape[2]))
        # observation=np.concatenate((cash_observation,observation),axis=0)

        cash_observation = np.ones((1, next_day.shape[1], next_day.shape[2]))
        next_day = np.concatenate((cash_observation, next_day), axis=0)

        next_day_open_price_vector = next_day[:, -1, 0]
        open_price_vector = next_day[:, 0, 0]
        y1 = next_day_open_price_vector / open_price_vector
        # print("REBAL:", next_day[-1,0,-1], next_day[-1,-1,-1])
        # y1[0] = 1.5

        reward, info, done2 = self.sim.step(weights, y1[:(self.num_assets + 1)])

        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        info['date'] = str(observation[-1, -1, -1]).replace('.0', '')
        info['steps'] = self.src.step_

        self.infos.append(info)

        return observation, reward, done1 or done2, info

    def reset(self):
        self.w0 = np.array([1.0] + [0.0] * (self.num_assets))
        self.infos = []
        self.sim.reset()
        observation = self.src.reset()
        # cash_observation = np.ones((1, int(self.window_length / self.window_period)+1, observation.shape[2]))
        # observation = np.concatenate((cash_observation, observation), axis=0)
        info = {}
        return observation, info
