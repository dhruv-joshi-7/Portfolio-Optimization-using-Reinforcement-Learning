from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from models import DRLAgent
import pandas as pd
from env_portfolio import StockPortfolioEnv
import json
import matplotlib.pyplot as plt

investment = 1000000


def test(trained_a2c):
    test_df = pd.read_csv("datasets/test_df.csv")
    print(type(test_df['cov_list'][0]))

    test_df['cov_list'][0] = json.load(test_df['cov_list'][0])
    print(test_df['cov_list'][0])

    tech_indicator_list = ['f01', 'f02', 'f03', 'f04']
    stock_dimension = len(test_df.tic.unique())
    state_space = stock_dimension
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
    e_trade_gym = StockPortfolioEnv(df=test_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    a2c_test_daily_return, a2c_test_weights = DRLAgent.DRL_prediction(model=trained_a2c,
                                                                      test_data=test_df,
                                                                      test_env=env_trade,
                                                                      test_obs=obs_trade)
    print(a2c_test_daily_return)


def load_models(show=0):
    trained_ppo = PPO.load("trained_models/ppo")
    trained_a2c = A2C.load("trained_models/a2c")
    trained_ddpg = DDPG.load("trained_models/ddpg")
    if(show):
        print("A2C ---------------------")
        print(trained_a2c.get_parameters())

        print("ppo ---------------------")
        print(trained_ppo.get_parameters())

        print("ddpg --------------------")
        print(trained_ddpg.get_parameters())

    return [trained_a2c, trained_ppo, trained_ddpg]


def print_weights(dates):
    a2c_weights = pd.read_csv("returns/a2c_test_weights.csv")
    a2c_weights = a2c_weights.drop('date', axis=1)
    cols = a2c_weights.columns

    a2c_weights = a2c_weights.values.tolist()
    for i in range(len(a2c_weights)):
        ax = plt.gca()
        ax.set_ylim([0.0, max(a2c_weights[i]) + 0.05])
        plt.xticks(rotation=90)
        plt.stem(cols, a2c_weights[i], linefmt='--')
        plt.pause(0.1)
        plt.clf()
        plt.title("Portoflio Weights on {x}".format(x=dates[i]))
    plt.show()


def returns(dates):
    [trained_a2c, trained_ppo, trained_ddpg] = load_models()
    a2c_test = pd.read_csv("returns/a2c_test.csv")
    ppo_test = pd.read_csv("returns/ppo_test.csv")
    ddpg_test = pd.read_csv("returns/ddpg_test.csv")

    # test(trained_a2c)

    a2c_test_cum_returns = (1 + a2c_test['daily_return']).cumprod()
    a2c_test_cum_returns.name = 'a2c'

    ppo_test_cum_returns = (1 + ppo_test['daily_return']).cumprod()
    ppo_test_cum_returns.name = 'ppo'

    ddpg_test_cum_returns = (1 + ddpg_test['daily_return']).cumprod()
    ddpg_test_cum_returns.name = 'ddpg'

    ddpg_test_cum_returns = (ddpg_test_cum_returns *
                             investment)
    a2c_test_cum_returns = a2c_test_cum_returns*investment
    ppo_test_cum_returns = ppo_test_cum_returns*investment

    final_result = pd.concat([dates, a2c_test_cum_returns,
                              ppo_test_cum_returns, ddpg_test_cum_returns], axis=1)
    print(final_result)


if __name__ == "__main__":
    prices_test = pd.read_csv("datasets/prices_test.csv")
    prices_test = prices_test.drop("Unnamed: 0", axis=1)
    dates = prices_test['date'][: -1]
    # returns(dates)
    print_weights(dates)
