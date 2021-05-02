from ta import add_all_ta_features
from ta.utils import dropna
import numpy as np


class FeatureGenerator():

    def __init__(self):
        pass

    def percent_change(self, data):
        return np.diff(data)  # / data[..., :-1]

    def generate(self, data):
        if "MSFT" in data[0]["stocks"]:
            can_sell = 1 if (data[0]["stocks"]["MSFT"] > 0) else 0
        else:
            can_sell = 0

        can_buy = 1 if data[0]["cash"] > data[1][0].iloc[-1]["Close"] else 0

        features = []
        for ticker in data[1]:
            features.append(
                self.percent_change(
                    ticker[["Volume", "Close", "Open", "High", "Low",
                            # "volume_adi", "volume_nvi", "volatility_bbw",
                            # "volatility_kchi", "trend_macd", "trend_sma_slow",
                            # "momentum_rsi", "momentum_kama", "others_dlr"
                            ]].to_numpy().T
                )
            )

        features = np.array(features).flatten().tolist()

        features += [can_buy, can_sell]

        return np.array(features).reshape(1, -1)

    def generate_single(self, data):
        ticker = 0
        array = self.percent_change(data[ticker][["Volume", "Close", "Open", "High", "Low",
                                                  # "volume_adi", "volume_nvi", "volatility_bbw",
                                                  # "volatility_kchi", "trend_macd", "trend_sma_slow",
                                                  # "momentum_rsi", "momentum_kama", "others_dlr"
                                                  ]].to_numpy().T)

        return np.array([array], dtype=np.float64).reshape(1, -1)
