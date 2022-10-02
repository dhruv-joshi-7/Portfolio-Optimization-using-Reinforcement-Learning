import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from IPython.display import display, HTML
from datetime import datetime
import pypfopt

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt import risk_models
from pypfopt import expected_returns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def show_clean_p(port_df):
    p1_show_1 = (port_df.transpose()[0]).map(
        lambda x: "{:.3%}".format(x)).to_frame().transpose()
    return display(HTML(p1_show_1.to_html()))


def get_daily_max_drawdown(prices, window):
    max_rolling = prices.rolling(min_periods=1, window=window).max()
    daily_drawdown = (prices / max_rolling) - 1
    max_daily_drawdown = daily_drawdown.rolling(
        min_periods=1, window=window).min()
    return daily_drawdown, max_daily_drawdown


def sharpe_ratio():

    prices_full_train_df = pd.read_csv("datasets/prices_full_train_df.csv")
    prices_full_test_df = pd.read_csv("datasets/prices_full_test_df.csv")

    prices_train_df = prices_full_train_df.copy()
    prices_test_df = prices_full_test_df.copy()

    prices_test_df = prices_test_df.drop(
        ['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
    prices_train_df = prices_train_df.drop(
        ['Unnamed: 0.1', 'Unnamed: 0'], axis=1)

    # print(prices_train_df.head())

    prices_train_df = prices_train_df.reset_index(
        drop=True).set_index(['date'])
    prices_test_df = prices_test_df.reset_index(drop=True).set_index(['date'])

    # Define a Function for Displaying the Cleaned Weights

    # Get List of all ticker symbols
    ticker_list = list(prices_train_df.columns)
    n_assets = len(ticker_list)  # Number of assets

    uniform_weights = np.ones((n_assets))/n_assets

    uniform_weights_port = pd.DataFrame([uniform_weights], columns=ticker_list)

    print("\nuniform weights portfolio:\n")
    show_clean_p(uniform_weights_port)

    # Plotting the Daily Draw Down

    ticker_symb = ['SBIN.NS']
    prices = prices_train_df[ticker_symb]
    window = 250

    max_rolling = prices.rolling(min_periods=1, window=window).max()

    daily_drawdown, max_daily_drawdown = get_daily_max_drawdown(prices, window)
    daily_drawdown.name = "{} daily drawdown".format(ticker_symb)
    #daily_drawdown = pd.DataFrame(daily_drawdown)

    fig, ax = plt.subplots(figsize=(6, 4))
    daily_drawdown.plot(ax=ax)
    ax.set_title("Daily Drawdown", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    fig.savefig('results/daily_drawdown.png')

    # Using the average daily return to calculate portfolio return

    returns = prices_train_df.pct_change()  # get the assets daily returns
    mean_daily_returns = returns.mean().values

    uw_returns = np.dot(mean_daily_returns, uniform_weights)

    print(
        "uniform weights portfolio average daily return = {:.4%}".format(uw_returns))

    # Annualized Return, Variance and Standard Deviation

    def get_annualized_return(prices, weigths):
        months = (pd.to_datetime(prices_train_df.index)
                  [-1] - pd.to_datetime(prices_train_df.index)[0]) / np.timedelta64(1, 'M')
        months = np.floor(months)
        total_return = (prices.iloc[-1].dot(weigths) -
                        prices.iloc[0].dot(weigths)) / prices.iloc[0].dot(weigths)
        annualized_return = ((1 + total_return) ** (12 / months)) - 1
        return annualized_return

    uw_annual_return = get_annualized_return(prices_train_df, uniform_weights)

    def get_portfolio_variance(returns, weigths):
        covariance_returns = returns.cov() * 250
        return np.dot(weigths.T, np.dot(covariance_returns, weigths))

    uw_var = get_portfolio_variance(returns, uniform_weights)

    print("uniform weights portfolio annualized return = {:.4%}".format(
        uw_annual_return))
    print(
        "uniform weights portfolio annualized variance = {:.1%}".format(uw_var))
    print("uniform weights portfolio annualized std = {:.1%}".format(
        np.sqrt(uw_var)))

    # Sharpe ratio

    uniform_returns = returns.dot(uniform_weights)

    rfr = 0.04  # Risk free rate

    uw_vol = uniform_returns.std() * np.sqrt(250)

    uw_sharpe_ratio = ((uw_annual_return - rfr) / uw_vol)

    print("uniform weights portfolio sharpe ratio = {:.2f}".format(
        uw_sharpe_ratio))

    # Plotting the cummulative return
    uniform_cum_returns = (1 + uniform_returns).cumprod()
    uniform_cum_returns.name = "portifolio 1: uniform weights"

    fig, ax = plt.subplots(figsize=(6, 4))
    uniform_cum_returns.plot(ax=ax, alpha=0.4)

    plt.legend(loc="best")
    plt.grid(True)
    ax.set_ylabel("cummulative return")
    ax.set_title('Uniform Weights Culmulative Returns', fontsize=14)

    fig.savefig('results/uniform_weights_portfolio.png')

    mu = expected_returns.mean_historical_return(prices_train_df)
    Sigma = risk_models.sample_cov(prices_train_df)
    ef = EfficientFrontier(mu, Sigma)
    ef.add_objective(objective_functions.L2_reg, gamma=1)

    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    max_sharpe_portfolio = pd.DataFrame(cleaned_weights, index=[0])

    print("max sharpe portfolio:")
    show_clean_p(max_sharpe_portfolio)

    _ = ef.portfolio_performance(verbose=True, risk_free_rate=rfr)

    print()

    # plt.figure(dpi = 1200)
    plt.subplots(figsize=(6, 3))
    plt.title('Maximum Sharpe and Equal Weights Portfolios')
    plt.bar(max_sharpe_portfolio.T.index,
            max_sharpe_portfolio.T[0], alpha=0.4, label='max_sharpe_port')
    plt.bar(uniform_weights_port.T.index,
            uniform_weights_port.T[0], alpha=0.4, label='equal_weights_port')
    plt.xlabel('Stock Name')
    plt.ylabel('Stock Weight')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
    plt.savefig('weights.png')

    max_sharpe_portfolio.to_csv("datasets/max_sharpe_portfolio.csv")
    uniform_weights_port.to_csv("datasets/uniform_weights_port.csv")
    prices_train_df.to_csv("datasets/prices_train_df.csv")
    prices_test_df.to_csv("datasets/prices_test_df.csv")


if __name__ == "__main__":
    sharpe_ratio()
