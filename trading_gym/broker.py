import yfinance as yf

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from ta import add_all_ta_features
from ta.utils import dropna

from trading_gym.feature_gen import FeatureGenerator


from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


START_INDEX = 0
MAX_INDEX = 1000
WINDOW = 200
AVAILABLE_TICKERS = ["MSFT"]


def fetch_data(period, intervall):
    states = []
    for ticker in AVAILABLE_TICKERS:
        states.append(yf.Ticker(ticker).history(
            period=period, interval=intervall, auto_adjust=True).reset_index())
    return states


class Bank():
    def __init__(self, period, interval):
        self.time = np.random.randint(MAX_INDEX)
        self.tickers = []

        states = fetch_data(period, interval)
        for state in states:
            state_edit = state.drop(columns=['Dividends', 'Stock Splits'])
            state_edit = dropna(state_edit)
            state_edit = state_edit.reset_index()
            # state_edit = add_all_ta_features(
            #    state_edit, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

            self.tickers.append(state_edit)
            #    state.iloc[START_INDEX:START_INDEX+WINDOW][["High", "Low", "Close", "Volume"]].to_numpy())
        self.tick()

    def get_price(self, ticker):
        index = AVAILABLE_TICKERS.index(ticker)
        return self.state[index].iloc[-1]["Close"]

    def get_ticker_states(self):
        return self.state

    def reached_limit(self):
        return self.time >= MAX_INDEX

    def tick(self):
        self.time += 1

        self.state = []
        for curr in self.tickers:
            self.state.append(
                curr.iloc[START_INDEX+self.time:START_INDEX+WINDOW+self.time]
            )

        #self.tickers = np.array(self.tickers)


class Portfolio():
    def __init__(self, period, interval, cash=1000, order_price=0, sell_price=0):
        self.cash = cash
        self.order_price = order_price
        self.sell_price = sell_price
        self.portfolio = {}
        self.bank = Bank(period, interval)

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

    def __init__(self, period="10y", interval="1d"):
        self.period = period
        self.interval = interval
        self.feature_gen = FeatureGenerator()
        self.reset()
        self.worth = self.portfolio.get_worth()

    def _get_state(self):
        return self.portfolio.get_state()

    def get_portfolio(self):
        return self.portfolio

    def reset(self):
        self.portfolio = Portfolio(self.period, self.interval)
        self.worth = self.portfolio.get_worth()
        return self.feature_gen.generate_single(self._get_state()[1])

    def render(self, mode):
        print("State: " + str(self.portfolio.get_state()
                              [0]) + "\nValue: " + str(self.worth))

        print("Sells: " + str(self.portfolio.num_sells))
        print("Buys: " + str(self.portfolio.num_buys))
        print("-----------\n\n")

    def step(self, action):
        """
        action: List of tuples with ("TICKER", action) where action: buy=0, sell=1, hold=2
        """
        action = [("MSFT", action)]
        self.portfolio.tick()

        outcome = 0
        for (ticker, order) in action:
            if order == 0:
                outcome += self.portfolio.buy(ticker, 1)
            elif order == 1:
                outcome += self.portfolio.sell(ticker, 1)

        difference = self.portfolio.get_worth() - self.worth
        self.worth = self.portfolio.get_worth()

        info = {"tickers": AVAILABLE_TICKERS}

        return self.feature_gen.generate_single(self._get_state()[1]), difference, False, info


class BrokerEnvTF(py_environment.PyEnvironment):

    def __init__(self, period="10y", interval="1d"):
        self.period = period
        self.interval = interval
        self.feature_gen = FeatureGenerator()
        self.reset()
        self.worth = self.portfolio.get_worth()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, 5, 199,), dtype=np.float64, name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_portfolio(self):
        return self.portfolio

    def _reset(self):
        self.portfolio = Portfolio(self.period, self.interval)
        self.worth = self.portfolio.get_worth()
        return ts.restart(
            #np.array([[0, 2], [2, 3]], dtype=np.int32),
            self.feature_gen.generate_single(self._get_state()[1]),
        )

    def _render(self, mode):
        print("State: " + str(self.portfolio.get_state()
                              [0]) + "\nValue: " + str(self.worth))

        print("Sells: " + str(self.portfolio.num_sells))
        print("Buys: " + str(self.portfolio.num_buys))
        print("-----------\n\n")

    def _get_state(self):
        return self.portfolio.get_state()

    def _step(self, action):
        """
        action: List of tuples with ("TICKER", action) where action: buy=0, sell=1, hold=2
        """
        action = [("MSFT", action)]
        self.portfolio.tick()

        outcome = 0
        for (ticker, order) in action:
            if order == 0:
                outcome += self.portfolio.buy(ticker, 1)
            elif order == 1:
                outcome += self.portfolio.sell(ticker, 1)

        difference = self.portfolio.get_worth() - self.worth
        self.worth = self.portfolio.get_worth()

        info = {"tickers": AVAILABLE_TICKERS}

        """
        When time limit is reached return termination to avoid out of bounds
        """
        if self.portfolio.bank.reached_limit():
            return ts.termination(
                self.feature_gen.generate_single(self._get_state()[1]),
                reward=difference,
            )

        return ts.transition(
            self.feature_gen.generate_single(self._get_state()[1]),
            reward=difference,
            discount=0.1
        )
