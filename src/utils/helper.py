import numpy as np
import pandas as pd

eps = 1e-08

import pandas as pd

def pad_tickers_to_union_dates(
    df, date_col='date', ticker_col='ticker',
    fill='bfill',          # 'bfill' | 'ffill' | 'constant' | None
    fill_value=1,          # used when fill == 'constant'
    value_cols=None        # defaults to all columns except date/ticker
):
    """
    Pads each ticker to have rows on every date in the global union of dates.
    Adds a boolean column '_observed' indicating whether the row existed originally.

    fill:
      - 'bfill': for each ticker, back-fill missing values from the first available date
      - 'ffill': forward-fill (rarely what you want for back history)
      - 'constant': fill missing with 'fill_value'
      - None: leave NaNs (useful if you’ll handle later)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.drop_duplicates([ticker_col, date_col])

    if value_cols is None:
        value_cols = [c for c in df.columns if c not in (ticker_col, date_col)]

    # Union of all dates and all tickers
    all_dates   = pd.Index(sorted(df[date_col].unique()))
    all_tickers = pd.Index(sorted(df[ticker_col].unique()))
    full_index  = pd.MultiIndex.from_product([all_tickers, all_dates],
                                             names=[ticker_col, date_col])

    # Mark observed rows, then reindex to full grid
    df['_observed'] = True
    out = (df.set_index([ticker_col, date_col])
             .reindex(full_index)
             .reset_index())
    out['_observed'] = out['_observed'].fillna(False)

    # Fill strategy
    if value_cols:
        if fill in ('bfill', 'ffill'):
            out = out.sort_values([ticker_col, date_col])
            filler = {'bfill': 'bfill', 'ffill': 'ffill'}[fill]
            out[value_cols] = out.groupby(ticker_col, group_keys=False)[value_cols].apply(
                lambda g: getattr(g, filler)()
            )
        elif fill == 'constant':
            out[value_cols] = out[value_cols].fillna(fill_value)
        # else: leave NaNs

    return out

def calculate_returns(values):
    """
    Calculates the periodic returns from a numpy array of portfolio values.

    Args:
        values (np.ndarray): A 1D numpy array of portfolio values,
                             ordered chronologically.

    Returns:
        np.ndarray: A numpy array of the calculated returns for each period.
                    Returns an empty array if the input has fewer than
                    two values.
    """
    # It's not possible to calculate returns with fewer than 2 data points.
    if values.size < 2:
        return np.array([])

    # Use slicing to get previous and current values for all periods at once.
    previous_values = values[:-1]
    current_values = values[1:]

    # Use np.where to handle division by zero safely.
    # Where previous_value is 0, return 0.0, otherwise calculate the return.
    returns = np.where(
        previous_values == 0,
        0.0,
        (current_values - previous_values) / previous_values
    )

    return returns

def apply_inverse_price_tensor(state_np, starting_price=100.0):
    num_stocks, timesteps, features = state_np.shape
    inverse_state = np.zeros_like(state_np)
    for i in range(num_stocks):
        for j in range(features-2):
            original_series = state_np[i, :, j]
            inverse_series = generate_inverse_price(original_series, starting_price)
            inverse_state[i, :, j] = inverse_series
        for j in range(features-2, features):
            inverse_state[i, :, j] = state_np[i, :, j]

    return inverse_state

def generate_inverse_price(price_series, starting_price=100.0):
    price_series = np.array(price_series)
    normal_return = price_series[1:] / price_series[:-1]
    inverse_return = 1.0 / normal_return

    inverse_prices = [starting_price]
    for r in inverse_return:
        inverse_prices.append(inverse_prices[-1] * r)

    return np.array(inverse_prices)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def get_actor_path(country, episode, window_length, window_period, train_period, seed, buy_fee, sell_fee, time_cost, target_date, rebalancing):
    return "weights/{}/episode_{}_window_length_{}_period_{}_train_{}_rebal_{}/{}/buy_{}_sell_{}_time_{}/{}/actor.pth".format(country, episode, window_length, window_period, train_period, rebalancing, seed, buy_fee, sell_fee, time_cost, target_date)

def get_critic_path(country, episode, window_length, window_period, train_period, seed, buy_fee, sell_fee, time_cost, target_date, rebalancing):
    return "weights/{}/episode_{}_window_length_{}_period_{}_train_{}_rebal_{}/{}/buy_{}_sell_{}_time_{}/{}/critic.pth".format(country, episode, window_length, window_period, train_period, rebalancing, seed, buy_fee, sell_fee, time_cost, target_date)

def get_pca_path(dataset, window_length, window_period, rebalancing, epoch, batch, learning_rate, hidden_dim, dropout, num_layer, pca_dim, stock_name):
    return "market_regime_weights/{}/pca_model/window_length_{}_period_{}_rebal_{}/{}_{}_{}/{}_{}_{}/dim_{}/{}.pkl".format(dataset, window_length, window_period, rebalancing, epoch, batch, learning_rate, hidden_dim, dropout, num_layer, pca_dim, stock_name)

def get_gmm_path(dataset, window_length, window_period, rebalancing, epoch, batch, learning_rate, hidden_dim, dropout, num_layer, n_component, stock_name):
    return "market_regime_weights/{}/gmm_model/window_length_{}_period_{}_rebal_{}/{}_{}_{}/{}_{}_{}/n_{}/{}.pkl".format(dataset, window_length, window_period, rebalancing, epoch, batch, learning_rate, hidden_dim, dropout, num_layer, n_component, stock_name)

def get_encoder_path(window_length, window_period, rebalancing, epoch, batch, learning_rate, hidden_dim, dropout, num_layer, seed):
    return "weights/window_length_{}_period_{}_rebal_{}/{}_{}_{}/{}_{}_{}/{}".format(window_length, window_period, rebalancing, epoch, batch, learning_rate, hidden_dim, dropout, num_layer, seed)

def convert_to_float(x):
    return x.replace('-', '')

def calculate_transaction_fee(w_t: np.array, w_t_1: np.array, f_buy: float = 0.1, f_sell: float = 0.1) -> tuple:
    n = w_t.shape[0]
    x_t = np.zeros(n)
    while True:
        x_t_prime = np.copy(x_t)
        for i in range(n):
            # Iteratively updates each variable while fixing all others
            theta = np.sum(np.vectorize(lambda x: x * f_sell if x >= 0 else -x * f_buy)(x_t[1:]))
            x_t[i] = w_t[i] - w_t_1[i] * (1-theta)
        if np.linalg.norm(x_t_prime - x_t) < eps:
            break
    return x_t_prime, theta

def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)

def sharpe(returns, freq=30, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)

def calculate_variance(data):
    n = len(data)
    
    # Calculate the mean
    mean = sum(data) / n
    # Calculate the squared differences from the mean
    squared_diff = [(x - mean) ** 2 for x in data]
    # Calculate the variance
    variance = sum(squared_diff) / n
    return variance