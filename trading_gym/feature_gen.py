from ta import add_all_ta_features
from ta.utils import dropna
import numpy as np

class FeatureGenerator():

    def __init__(self):
        pass

    def generate(self, data):
        features = []

        for ticker in data:        
            features.append(data[0])

        return np.array(features)

