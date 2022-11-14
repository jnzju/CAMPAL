from .score_strategy import ScoreStrategy
from .builder import STRATEGIES


@STRATEGIES.register_module()
class MarginSampling(ScoreStrategy):
	def __init__(self, dataset, net, args, logger, timestamp):
		super(MarginSampling, self).__init__(dataset, net, args, logger, timestamp)

	def calculating_scores(self, dataset_u):
		return self.predict(self.clf, dataset_u, 'margin', log_show=False)
