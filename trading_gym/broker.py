import yfinance as yf

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

START_INDEX = 0
WINDOW = 200
AVAILABLE_TICKERS = ["MSFT"]

states = []
for ticker in AVAILABLE_TICKERS:
    states.append(yf.Ticker(ticker).history(
        period="7d", interval="1m", auto_adjust=True).reset_index())


class Bank():
    def __init__(self):
        self.time = 0
        self.tickers = []
        for state in states:
            self.tickers.append(
                state.iloc[START_INDEX:START_INDEX+WINDOW][["High", "Low", "Close", "Volume"]].to_numpy())

        self.tickers = np.array(self.tickers)
        print(self.tickers.shape)

    def get_price(self, ticker):
        index = AVAILABLE_TICKERS.index(ticker)
        return self.tickers[index][-1][1]

    def get_ticker_states(self):
        return self.tickers

    def tick(self):
        self.time += 1

        self.tickers = []
        for state in states:
            self.tickers.append(
                state.iloc[START_INDEX+self.time:START_INDEX+WINDOW+self.time][["High", "Low", "Close", "Volume"]].to_numpy())

        self.tickers = np.array(self.tickers)


class Portfolio():
    def __init__(self, cash=1000, order_price=0, sell_price=0):
        self.cash = cash
        self.order_price = order_price
        self.sell_price = sell_price
        self.portfolio = {}
        self.bank = Bank()

        self.num_sells = 0
        self.num_buys = 0

    def buy(self, ticker, amount):
        price = self.bank.get_price(ticker)
        if price > self.cash:
            return -1

        self.cash -= price + self.order_price

        if ticker not in self.portfolio:
            self.portfolio[ticker] = 0

        self.portfolio[ticker] += 1
        self.num_buys += 1
        return 0

    def sell(self, ticker, amount):

        if not ticker in self.portfolio:
            return -1

        if self.portfolio[ticker] < amount:
            return -1

        price = self.bank.get_price(ticker)

        self.cash += (price * amount) - self.sell_price
        self.portfolio[ticker] -= amount
        self.num_sells += 1
        return 0

    def get_worth(self):
        worth = self.cash
        for ticker in self.portfolio:
            worth += self.portfolio[ticker] * self.bank.get_price(ticker)

        return worth

    def get_state(self):
        return self.portfolio, self.bank.get_ticker_states()

    def tick(self):
        self.bank.tick()


class BrokerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.reset()
        self.portfolio = Portfolio()
        self.worth = self.portfolio.get_worth()

    def reset(self):
        self.portfolio = Portfolio()
        self.worth = self.portfolio.get_worth()
        return self.portfolio.get_state()

    def render(self):
        print("-----------\nState: " + str(self.portfolio.get_state()
                                           [0]) + "\nValue: " + str(self.worth))

        print("Sells: " + str(self.portfolio.num_sells))
        print("Buys: " + str(self.portfolio.num_buys))

    def step(self, action):
        """
        action: List of tuples with ("TICKER", action) where action: buy=0, sell=1
        """
        self.portfolio.tick()

        outcome = 0
        for (ticker, order) in action:
            if order == 0:
                outcome += self.portfolio.buy(ticker, 1)
            elif order == 1:
                outcome += self.portfolio.sell(ticker, 1)

        difference = self.portfolio.get_worth() - self.worth
        self.worth = self.portfolio.get_worth()

        info = AVAILABLE_TICKERS

        return self.portfolio.get_state(), difference, False, info
