import torch
from .score_strategy import ScoreStrategy
from .builder import STRATEGIES


@STRATEGIES.register_module()
class BALDDropout(ScoreStrategy):
    def __init__(self, dataset, net, args, logger, timestamp, n_drop=5):
        super(BALDDropout, self).__init__(dataset, net, args, logger, timestamp)
        self.n_drop = n_drop

    def calculating_scores(self, dataset_u):
        prob = self.predict(self.clf, dataset_u, metric='prob', n_drop=self.n_drop, dropout_split=True)
        pb = prob.mean(0)
        entropy1 = (-pb * torch.log(pb)).sum(1)
        entropy2 = (-prob * torch.log(prob)).sum(2).mean(0)
        return (entropy1 - entropy2).cpu()
