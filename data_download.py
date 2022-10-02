import pandas as pd
import numpy as np
from config import config
import matplotlib.pylab as plt
import yfinance as yf
from pandas_datareader import data as pdr
import os


if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

ticker_list = config.NIFTY50


def download(start_date='2014-01-01', end_date="2022-01-01"):
    df = pdr.get_data_yahoo([ticker_list][0],
                            start=start_date, end=end_date)
    data = df.copy()
    data = data.stack().reset_index()
    data.columns.names = [None]
    data = data.drop(['Close'], axis=1)
    data.columns = ['date', 'tic', 'close', 'high', 'low', 'open', 'volume']
    no_datasets = []
    for i in ticker_list:
        no_data_points = data[data['tic'] == i].shape[0]
        no_datasets.append((i, no_data_points))
        data_points_df = pd.DataFrame(no_datasets)
    data_filtered = data
    data_filtered.to_csv('datasets/data.csv', index=False)


def close_prices():
    df_prices = pd.read_csv('./datasets/data.csv')
    df_prices = df_prices.reset_index().set_index(['tic', 'date']).sort_index()
    tic_list = list(set([i for i, j in df_prices.index]))
    df_close = pd.DataFrame()
    df_prices = df_prices.reset_index().set_index(['tic', 'date']).sort_index()
    df_close = pd.DataFrame()

    for ticker in tic_list:
        series = df_prices.xs(ticker).close
        df_close[ticker] = series

    df_close = df_close.reset_index()
    df_close.to_csv('datasets/close_prices.csv', index=False)
    df_close_full_stocks = df_close

    # %store df_close_full_stocks
    df_close_full_stocks.to_csv('datasets/df_close_full_stocks.csv')


if __name__ == "__main__":
    download()
    close_prices()
