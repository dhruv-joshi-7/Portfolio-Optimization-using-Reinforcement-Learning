import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
from ta.utils import dropna

stock_selection_number = 20

feature_list = ['volatility_atr', 'volatility_bbw', 'volume_obv', 'volume_cmf',
                'trend_macd', 'trend_adx', 'trend_sma_fast',
                'trend_ema_fast', 'trend_cci', 'momentum_rsi']

short_names = ['atr', 'bbw', 'obv', 'cmf',
               'macd', 'adx', 'sma', 'ema', 'cci', 'rsi']


# FILTERED STOCKS LIST
filters = pd.read_csv('./datasets/filters.csv', index_col='stock_name')
list_of_stocks = filters.head(stock_selection_number).index


def add_features(data, feature_list, short_names):

    data_col_names = list(data.columns)
    filter_names = data_col_names + feature_list
    col_rename = data_col_names + short_names

    data = add_all_ta_features(data, open="open", high="high",
                               low="low", close="close", volume="volume")

    data = data[filter_names]
    data.columns = col_rename  # rename the columns to use shortened indicator names
    data = data.dropna()

    return data


def add_cov_matrix(df):
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []  # create empty list for storing coveriance matrices at each time step

    # look back for constructing the coveriance matrix is one year
    lookback = 252
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i, :]
        price_lookback = data_lookback.pivot_table(
            index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        covs = return_lookback.cov().values
        covs = covs  # /covs.max()
        cov_list.append(covs)

    df_cov = pd.DataFrame(
        {'date': df.date.unique()[lookback:], 'cov_list': cov_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    return df


def augment_data():
    global feature_list
    global short_names
    data = pd.read_csv('./datasets/data.csv')
    prices_data = pd.read_csv('./datasets/close_prices.csv')
    data = data[data['tic'].isin(list_of_stocks)]
    data_with_features = data.copy()
    data_with_features = add_features(
        data_with_features, feature_list, short_names)
    feature_list = list(data_with_features.columns)[7:]
    data_with_features_covs = data_with_features.copy()
    data_with_features_covs = add_cov_matrix(data_with_features_covs)
    data_with_features_covs.to_csv("datasets/data_with_features_covs.csv")


if __name__ == "__main__":
    augment_data()
