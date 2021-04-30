from ta import add_all_ta_features
from ta.utils import dropna
import numpy as np


class FeatureGenerator():

    def __init__(self):
        pass

    def percent_change(self, data):
        return np.diff(data) # / data[..., :-1]

    def generate(self, data):
        features = []

        for ticker in data[1]:
            features.append(
                ticker[["Volume", "Close", "Open", "High", "Low",
                        "volume_adi", "volume_nvi", "volatility_bbw",
                        "volatility_kchi", "trend_macd", "trend_sma_slow",
                        "momentum_rsi", "momentum_kama", "others_dlr"
                        ]].to_numpy()
            )

        return self.percent_change(np.array(features).squeeze().T)

    def generate_single(self, data):
        ticker = 0
        return self.percent_change(data[ticker][["Volume", "Close", "Open", "High", "Low",
                                           "volume_adi", "volume_nvi", "volatility_bbw",
                                           "volatility_kchi", "trend_macd", "trend_sma_slow",
                                           "momentum_rsi", "momentum_kama", "others_dlr"
                                           ]].to_numpy().T)
