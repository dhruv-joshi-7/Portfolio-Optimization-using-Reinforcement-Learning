from backtest import backtest_strat, baseline_strat
from backtest import BackTestStats, BaselineStats, BackTestPlot, backtest_strat, baseline_strat
from config import config

from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import env_portfolio
from env_portfolio import StockPortfolioEnv
import models
from models import DRLAgent


train_df = pd.read_csv("datasets/train_df.csv")
test_df = pd.read_csv("datasets/test_df.csv")


def parse_cov(s):
    import re
    arr = s.split(']\n [')
    arr[0] = arr[0][2:]
    arr[-1] = arr[-1][:-2]
    # for t in arr:
    #     print(t)
    #     print(list(map(float, re.split('  | -|\n -|\n  ', t))))
    res = pd.Series(
        [list(map(float, re.split('  | -|\n -|\n  ', t))) for t in arr])
    return res


ltrain = list(parse_cov(train_df['cov_list'][0]))
lt = [np.array(ltrain)]
for i in range(1, len(train_df['cov_list'])):
    lt.append(np.array(ltrain))
# print(lt)
train_df['cov_list'] = pd.DataFrame(lt)
# print(train_df)

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
print(type(env_train))


def A2C():
    agent = DRLAgent(env=env_train)
    A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
    model_a2c = agent.get_model(model_name="a2c", model_kwargs=A2C_PARAMS)
    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=50000)
    return trained_a2c


def PPO():
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


def DDPG():
    agent = DRLAgent(env=env_train)
    DDPG_PARAMS = {"batch_size": 128,
                   "buffer_size": 50000, "learning_rate": 0.001}
    model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)
    trained_ddpg = agent.train_model(model=model_ddpg,
                                     tb_log_name='ddpg',
                                     total_timesteps=50000)
    return trained_ddpg


def test_and_train():
    e_trade_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    trained_a2c = A2C()
    a2c_train_daily_return, a2c_train_weights = DRLAgent.DRL_prediction(model=trained_a2c,
                                                                        test_data=train_df,
                                                                        test_env=env_trade,
                                                                        test_obs=obs_trade)
    e_trade_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    trained_ppo = PPO()
    ppo_train_daily_return, ppo_train_weights = DRLAgent.DRL_prediction(model=trained_ppo,
                                                                        test_data=train_df,
                                                                        test_env=env_trade,
                                                                        test_obs=obs_trade)
    e_trade_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    trained_ddpg = DDPG()
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


if __name__ == "__main__":
    test_and_train()
