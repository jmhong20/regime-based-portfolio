from datetime import datetime
import numpy as np
import pandas as pd
import os
import yfinance as yf

import requests
import argparse

from utils.helper import *

tickers_list = {
    'us_stocks2': [
        "AAPL", # 1
        "AMZN", # 2
        "AVGO", # 3
        "COST", # 4
        "GOOGL", # 5
        "HD", # 6
        "JNJ", # 7
        "JPM", # 8
        "LLY", # 9
        "MA", # 10
        "META", # 11
        "MSFT", # 12
        "NFLX", # 13
        "NVDA", # 14
        "PG", # 15
        "TSLA", # 16
        "UNH", # 17
        "V", # 18
        "WMT", # 19
        "XOM", # 20
    ],
    'us_stocks3': [
        "NVDA", # 1
        "MSFT", # 2
        "AAPL", # 3
        "AMZN", # 4
        "GOOGL", # 5
        "META", # 6
        "AVGO", # 7
        "TSM", # 8
        "TSLA", # 9
        "BRK", # 10
        "JPM", # 11
        "WMT", # 12
        "LLY", # 13
        "ORCL", # 14
        "V", # 15
        "NFLX", # 16
        "MA", # 17
        "XOM", # 18
        "COST", # 19
        "JNJ", # 20
        "BAC", # 21
        "PG", # 22
        "PLTR", # 23
        "HD", # 24
        "SAP", # 25
        "ABBV", # 26
        "CVX", # 27
        "KO", # 28
        "ASML", # 29
        "BABA", # 30
        "GE", # 31
        "PM", # 32
        "CSCO", # 33
        "WFC", # 34
        "IBM", # 35
        "TMUS", # 36
        "UNH", # 37
        "AMD", # 38
        "CRM", # 39
        "MS", # 40
    ]
}

def parse_args():
    """
    Parse command line arguments.

    The other arguments not defined in this function are directly passed to main.py. For instance,
    an option like "--beta 1" is given directly to the main script.

    :return: the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_type', type=str, default='ohlcv')
    parser.add_argument('--target_date', type=str, default=None)
    parser.add_argument('--train_period', type=int, default=30)
    parser.add_argument('--country', type=str, default="us")
    parser.add_argument('--task', type=str, default="load_df")
    parser.add_argument('--save_numpy', type=str2bool, default=True)
    return parser.parse_known_args()

def main():
    """
    config
    """
    args, unknown = parse_args()
    feature_type = args.feature_type
    target_date = args.target_date
    target_date = None if target_date == 'None' else target_date
    train_period = args.train_period
    country = args.country
    task = args.task
    save_numpy = args.save_numpy

    if country == "us2":
        # tickers = get_snp500_tickers()
        tickers = tickers_list["us_stocks2"]
        index_ticker = '^GSPC'
        download_data = download_snp500_data
        directory = '../data/us2'
        os.makedirs(directory, exist_ok=True)
        data_path = '../data/us2/snp500_data.csv'
        numpy_path = '../data/us2/snp500_data.npy'

    if target_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    else:
        end_date = target_date
    start_date_d = datetime.strptime(end_date, "%Y-%m-%d").replace(year=datetime.strptime(end_date, "%Y-%m-%d").year - train_period)
    start_date = start_date_d.strftime("%Y-%m-%d")
    print(start_date, end_date)

    if task == "download":
        download_data(tickers, start_date, end_date)
    elif task == "load_df":
        df = pd.read_csv(data_path)
        stocks_df = df[df['ticker'].isin(tickers)]
        stocks_df = stocks_df.sort_values(by=['ticker', 'date'])
        print(stocks_df.head())

        index_df = df[df['ticker'] == index_ticker]
        index_df = index_df.sort_values(by=['date'])
        print(index_df.head())

        if save_numpy:
            history = df_to_tensor(stocks_df, datetime.strptime(end_date, "%Y-%m-%d"))
            print(history.shape)
            index_history = df_to_tensor(index_df, datetime.strptime(end_date, "%Y-%m-%d"))
            print(index_history.shape)

            np.save(numpy_path, history)
            np.save(numpy_path.replace('.npy', '_index.npy'), index_history)



def process_tickers(tickers):
    return list(map(lambda x: x.replace('.', '-'), tickers))

SNP500_COMPANIES_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_snp500_tickers():
    # requests로 먼저 데이터를 가져온 후 pandas로 파싱
    response = requests.get(SNP500_COMPANIES_URL, headers=headers)
    response.raise_for_status()  # HTTP 에러가 있으면 예외 발생
    # HTML 내용을 pandas로 파싱
    snp500_companies = pd.read_html(response.text)[0]
    tickers = process_tickers(snp500_companies['Symbol'].tolist())
    return tickers

def pad_tickers_to_union_dates(df, date_col='date', ticker_col='ticker',fill='bfill', value_cols=None):
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
        out = out.sort_values([ticker_col, date_col])
        filler = {'bfill': 'bfill', 'ffill': 'ffill'}[fill]
        out[value_cols] = out.groupby(ticker_col, group_keys=False)[value_cols].apply(
            lambda g: getattr(g, filler)()
        )
    return out

def download_snp500_data(tickers, start_date, end_date):
    data = []
    index_ticker = '^GSPC'
    for ticker in tickers+[index_ticker]:
        d = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        d_flat = d.stack(level='Ticker').reset_index()
        d_flat.columns.name = None
        d_flat.rename(columns={
            "Date": "date",
            "Ticker": "ticker",
            "Volume": "volume",
            "Adj Close": "adjusted_close"
        }, inplace=True)
        data.append(d_flat.copy())
        print(d_flat)
    df = pd.concat(data, ignore_index=True)
    df["adjusted_open"] = df["Open"] * (df["adjusted_close"] / df["Close"])
    df["adjusted_high"] = df["High"] * (df["adjusted_close"] / df["Close"])
    df["adjusted_low"] = df["Low"] * (df["adjusted_close"] / df["Close"])
    df = df[["date", "adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close", "volume", "ticker"]]
    df["date"] = df["date"].astype(str)
    df = df.sort_values(by=["ticker", "date"])

    if df[df['volume']<=1e-1].empty != True:
        df = df.replace(0, np.nan)
        df = df.ffill()
    df = pad_tickers_to_union_dates(df)
    df = df.drop_duplicates(subset=['ticker', 'date'])
    tickers = df['ticker'].unique()
    min_length = float('inf')
    for i in range(len(tickers)):
        s = tickers[i]
        s_length = len(df[df['ticker']==s]['date'].unique())
        if s_length > 0 and s_length < min_length:
            min_length = s_length
    american_avail_dates = sorted(df['date'].unique())[-min_length:]
    df = df[df['date'].isin(american_avail_dates)].copy()
    df = df.sort_values(by=['ticker', 'date'])
    full_index = pd.MultiIndex.from_product(
        [tickers, american_avail_dates],
        names=['ticker', 'date']
    )
    df = df.set_index(['ticker', 'date'])
    df = df.reindex(full_index)
    df = df.groupby('ticker').fillna(method='ffill')
    df = df.reset_index()
    df = df.sort_values(by=['ticker', 'date'])
    eps = 1e-8  # or any small value you want
    df = df.replace(0, eps)

    df.to_csv('../data/snp500_data.csv')

def df_to_tensor(df, given_date,train_period=30):
    # Calculate the date train_period years before
    start_date = given_date.replace(year=given_date.year - train_period)
    result_date_str = start_date.strftime("%Y-%m-%d")
    df = df[df['date'] >= result_date_str].copy()

    abbreviation = df['ticker'].unique().tolist()

    # 모든 종목 날짜 동일하게 맞추기 
    min_length = float('inf')
    for i in range(len(abbreviation)):
        s = abbreviation[i]
        s_length = len(df[df['ticker']==s]['date'].unique())
        if s_length > 0 and s_length < min_length:
            min_length = s_length
    american_avail_dates = sorted(df['date'].unique())[-min_length:]
    df = df[df['date'].isin(american_avail_dates)].copy()
    df = df.sort_values(by=['ticker', 'date'])

    history = []
    for stock in abbreviation:
        new_df = df[df['ticker'] == stock].drop(columns='ticker')[['adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'volume', 'date']]
        new_df['date'] = new_df['date'].astype(str)
        new_df['date'] = new_df['date'].apply(convert_to_float)
        new_df['date'] = new_df['date'].astype(float)
        new_list = new_df.values.tolist()
        history.append(new_list)
    history = np.array(history)
    history = history[:, :, :25] # 별 의미 없음; 사용하는 전체 피쳐의 개수를 25개까지 제한하겠다는 것임

    return history

if __name__ == "__main__":
    main()
