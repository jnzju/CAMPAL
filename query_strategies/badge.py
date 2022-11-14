from .representation_strategy import RepresentationStrategy
from scipy import stats
import numpy as np
from .builder import STRATEGIES
from datasets.dataloader import GetHandler


def init_centers(X, K, dist_mat):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(indsAll) < K:
        if len(indsAll) == 1:
            D2 = dist_mat[:, ind].ravel().astype(float)
        else:
            newD = dist_mat[:, indsAll[-1]].ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(indsAll)) + '\t' + str(sum(D2)), flush=True)
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        indsAll.append(ind)
        cent += 1
    return indsAll


@STRATEGIES.register_module()
class BadgeSampling(RepresentationStrategy):
    def __init__(self, dataset, net, args, logger, timestamp):
        super(BadgeSampling, self).__init__(dataset, net, args, logger, timestamp)

    def calculating_embeddings(self, dataset_u):
        return self.get_embedding(self.clf, dataset_u, 'grad')

    def query(self, n, aug_args_list=None):
        dataset_u = GetHandler(self.dataset, 'train_u', self.dataset.default_val_transform, 1,
                               self.args.aug_ratio_ulb, self.args.mix_ratio_ulb, self.args.aug_ulb_evaluation_mode,
                               strength_mode=self.args.aug_ulb_strength_mode, args_list=aug_args_list)
        dist_mat = self.calculating_dist_matrix(dataset_u, dataset_u)

        dataset_u_ori = GetHandler(self.dataset, 'train_u', self.dataset.default_val_transform)
        grad_embedding = self.calculating_embeddings(dataset_u_ori)
        chosen = init_centers(grad_embedding, n, dist_mat.cpu().numpy())
        return chosen
