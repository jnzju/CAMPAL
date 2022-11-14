import numpy as np
from .strategy import Strategy
from .builder import STRATEGIES


@STRATEGIES.register_module()
class RandomSampling(Strategy):
    def __init__(self, dataset, net, args, logger, timestamp):
        super(RandomSampling, self).__init__(dataset, net, args, logger, timestamp)

    def query(self, n):
        return np.random.choice(np.where(self.dataset.INDEX_LB == 0)[0], n, replace=False)
