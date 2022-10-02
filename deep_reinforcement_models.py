from pyexpat import features
import numpy as np
import pandas as pd

from numpy import array
from keras.models import Model, Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from finrl.finrl_meta.preprocessor.preprocessors import data_split
from backtest import backtest_strat, baseline_strat
from backtest import BackTestStats, BaselineStats, BackTestPlot, backtest_strat, baseline_strat
from config import config

from env_portfolio import StockPortfolioEnv
import models
from models import DRLAgent

from sklearn import preprocessing


def dataframe_of_features():

    df = pd.read_csv("datasets/data_with_features_covs.csv")
    # print(df.head())
    feature_start = 8
    features_list = list(df.columns)[feature_start:-1]
    features_df = df[features_list]
    features_df.index = df['date']

    features_array = np.array(features_df)
    features_scaler = preprocessing.MinMaxScaler()
    features_normalised = features_scaler.fit_transform(features_array)

    features_normalised = features_normalised.reshape(-1, 20, 10)

    model = Sequential()
    model.add(LSTM(4, activation='relu', input_shape=(20, 10)))
    model.add(RepeatVector(20))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(10)))
    model.compile(optimizer='adam', loss='mse')
    model.fit(features_normalised, features_normalised, epochs=100, verbose=1)

    # plot_model(model, show_shapes=True,
    #            to_file='./results/reconstruct_lstm_autoencoder.png')

    model.summary()
    model_feature = Model(inputs=model.inputs, outputs=model.layers[1].output)
    # plot_model(model_feature, show_shapes=True, show_layer_names=True,
    #            to_file='./results/lstm_encoder.png')
    yhat = model_feature.predict(features_normalised)
    features_reduced = yhat.reshape(-1, 4)
    df_reduced = df.copy()
    df_reduced = df_reduced .drop(features_list, axis=1)
    features_reduced_df = pd.DataFrame(features_reduced, columns=[
                                       'f01', 'f02', 'f03', 'f04'])
    df_reduced[['f01', 'f02', 'f03', 'f04']
               ] = features_reduced_df[['f01', 'f02', 'f03', 'f04']]
    data_df = df_reduced.copy()
    data_df.to_csv("datasets/data_df.csv")


def split_data():
    stock_selection_number = 20
    filters = pd.read_csv('./datasets/filters.csv', index_col='stock_name')

    filtered_stocks = filters.head(stock_selection_number).index
    data_df = pd.read_csv("datasets/data_df.csv")
    df_close_full_stocks = pd.read_csv("datasets/df_close_full_stocks.csv")

    df_prices = data_df.reset_index().set_index(['tic', 'date']).sort_index()
    df_close = pd.DataFrame()
    for ticker in filtered_stocks:
        series = df_prices.xs(ticker).close
        df_close[ticker] = series

    df_close = df_close.reset_index()
    train_pct = 0.8  # percentage of train data
    date_list = list(data_df.date.unique())  # List of dates in the data

    date_list_len = len(date_list)  # len of the date list
    train_data_len = int(train_pct * date_list_len)  # length of the train data

    train_start_date = date_list[0]
    train_end_date = date_list[train_data_len]

    test_start_date = date_list[train_data_len+1]
    test_end_date = date_list[-1]

    train_data = data_split(data_df, train_start_date, train_end_date)
    test_data = data_split(data_df, test_start_date, test_end_date)

    # Split the Close Prices dataset
    prices_train_data = df_close[df_close['date'] <= train_end_date]
    prices_test_data = df_close[df_close['date'] >= test_start_date]

    # split the Close Prices of all stocks
    prices_full_train = df_close_full_stocks[df_close_full_stocks['date']
                                             <= train_end_date]
    prices_full_test = df_close_full_stocks[df_close_full_stocks['date']
                                            >= test_start_date]

    # prices_train = prices_train_data.copy()
    prices_train_data.to_csv("datasets/prices_train.csv")

    # prices_test = prices_test_data.copy()
    prices_test_data.to_csv("datasets/prices_test.csv")

    train_df = train_data.copy()
    train_data.to_csv("datasets/train_df.csv")
    test_df = test_data.copy()
    test_data.to_csv("datasets/test_df.csv")

    # prices_full_train_df = prices_full_train.copy()
    prices_full_train.to_csv("datasets/prices_full_train_df.csv")

    # prices_full_test_df = prices_full_test.copy()
    prices_full_test.to_csv("datasets/prices_full_test_df.csv")
    return [train_df, test_df]


def A2C(env_train):
    agent = DRLAgent(env=env_train)
    A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
    model_a2c = agent.get_model(model_name="a2c", model_kwargs=A2C_PARAMS)
    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=50000)
    return trained_a2c


def PPO(env_train):
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.005,
        "learning_rate": 0.0001,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo,
                                    tb_log_name='ppo',
                                    total_timesteps=50000)
    return trained_ppo


def DDPG(env_train):
    agent = DRLAgent(env=env_train)
    DDPG_PARAMS = {"batch_size": 128,
                   "buffer_size": 50000, "learning_rate": 0.001}
    model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)
    trained_ddpg = agent.train_model(model=model_ddpg,
                                     tb_log_name='ddpg',
                                     total_timesteps=50000)
    return trained_ddpg


def test_and_train(train_df, test_df, env_kwargs, env_train):
    e_trade_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    trained_a2c = A2C(env_train)
    a2c_train_daily_return, a2c_train_weights = DRLAgent.DRL_prediction(model=trained_a2c,
                                                                        test_data=train_df,
                                                                        test_env=env_trade,
                                                                        test_obs=obs_trade)
    e_trade_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    trained_ppo = PPO(env_train)
    ppo_train_daily_return, ppo_train_weights = DRLAgent.DRL_prediction(model=trained_ppo,
                                                                        test_data=train_df,
                                                                        test_env=env_trade,
                                                                        test_obs=obs_trade)
    e_trade_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    trained_ddpg = DDPG(env_train)
    ddpg_train_daily_return, ddpg_train_weights = DRLAgent.DRL_prediction(model=trained_ddpg,
                                                                          test_data=train_df,
                                                                          test_env=env_trade,
                                                                          test_obs=obs_trade)

    a2c_train_daily_return.to_csv("datasets/a2c_train_daily_return.csv")
    ppo_train_daily_return.to_csv("datasets/ppo_train_daily_return.csv")
    ddpg_train_daily_return.to_csv("datasets/ddpg_train_daily_return.csv")

    e_trade_gym = StockPortfolioEnv(df=test_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    a2c_test_daily_return, a2c_test_weights = DRLAgent.DRL_prediction(model=trained_a2c,
                                                                      test_data=test_df,
                                                                      test_env=env_trade,
                                                                      test_obs=obs_trade)
    a2c_test_weights.to_csv('a2c_test_weights.csv')

    e_trade_gym = StockPortfolioEnv(df=test_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    ppo_test_daily_return, ppo_test_weights = DRLAgent.DRL_prediction(model=trained_ppo,
                                                                      test_data=test_df,
                                                                      test_env=env_trade,
                                                                      test_obs=obs_trade)
    ppo_test_weights.to_csv('ppo_test_weights')

    e_trade_gym = StockPortfolioEnv(df=test_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    ddpg_test_daily_return, ddpg_test_weights = DRLAgent.DRL_prediction(model=trained_ddpg,
                                                                        test_data=test_df,
                                                                        test_env=env_trade,
                                                                        test_obs=obs_trade)
    ddpg_test_weights.to_csv('ddpg_test_weights')

    a2c_test_portfolio = a2c_test_daily_return.copy()
    a2c_test_returns = a2c_test_daily_return.copy()

    ppo_test_portfolio = ppo_test_daily_return.copy()
    ppo_test_returns = ppo_test_daily_return.copy()

    ddpg_test_portfolio = ddpg_test_daily_return.copy()
    ddpg_test_returns = ddpg_test_daily_return.copy()

    a2c_test_portfolio.to_csv('datasets/a2c_test_portfolio.csv')
    a2c_test_returns.to_csv('datasets/a2c_test_returns.csv')

    ppo_test_portfolio.to_csv('datasets/ppo_test_portfolio.csv')
    ppo_test_returns.to_csv('datasets/ppo_test_returns.csv')

    ddpg_test_portfolio.to_csv('datasets/ddpg_test_portfolio.csv')
    ddpg_test_returns.to_csv('datasets/ddpg_test_returns.csv')


def main():
    [train_df, test_df] = split_data()
    print((train_df['cov_list'][0]))

    stock_dimension = len(train_df.tic.unique())
    state_space = stock_dimension

    weights_initial = [1/stock_dimension]*stock_dimension
    tech_indicator_list = ['f01', 'f02', 'f03', 'f04']
    env_kwargs = {
        "hmax": 500,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 0,
        'initial_weights': [1/stock_dimension]*stock_dimension
    }
    e_train_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    test_and_train(train_df, test_df, env_kwargs, env_train)


if __name__ == "__main__":
    dataframe_of_features()
    split_data()
    # main()
