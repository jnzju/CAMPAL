import numpy as np
import torch
from .representation_strategy import RepresentationStrategy
from .builder import STRATEGIES
from datasets.dataloader import GetHandler


@STRATEGIES.register_module()
class CoreSet(RepresentationStrategy):
    def __init__(self, dataset, net, args, logger, timestamp):
        super(CoreSet, self).__init__(dataset, net, args, logger, timestamp)

    def calculating_embeddings(self, dataset_u):
        return self.get_embedding(self.clf, dataset_u)

    def query(self, n, aug_args_list=None):
        dataset_u = GetHandler(self.dataset, 'train_u', self.dataset.default_val_transform, 1,
                               self.args.aug_ratio_ulb, self.args.mix_ratio_ulb, self.args.aug_ulb_evaluation_mode,
                               strength_mode=self.args.aug_ulb_strength_mode, args_list=aug_args_list)
        dataset_l = GetHandler(self.dataset, 'train', self.dataset.default_val_transform, 1,
                               self.args.aug_ratio_ulb, self.args.mix_ratio_ulb, self.args.aug_ulb_evaluation_mode,
                               strength_mode=self.args.aug_ulb_strength_mode, args_list=aug_args_list)

        dist_full = torch.cat([self.calculating_dist_matrix(dataset_l, dataset_u),
                               self.calculating_dist_matrix(dataset_u, dataset_u)])

        temp_idxs_lb = torch.tensor([True] * len(self.dataset.DATA_INFOS['train']) +
                                    [False] * len(self.dataset.DATA_INFOS['train_u']))
        selected_samples = []
        unselected_samples = list(range(len(self.dataset.DATA_INFOS['train_u'])))

        for _ in range(n):
            sub_dist_mat = dist_full[temp_idxs_lb, :][:, ~temp_idxs_lb[len(self.dataset.DATA_INFOS['train']):]]
            sub_dist_set = torch.min(sub_dist_mat, dim=0)[0]
            ulb_idx_local = torch.argmin(sub_dist_set)
            ulb_idx_global = unselected_samples[ulb_idx_local]
            selected_samples.append(ulb_idx_global)
            del unselected_samples[ulb_idx_local]
            temp_idxs_lb[ulb_idx_global + len(self.dataset.DATA_INFOS['train'])] = True

        return np.array(selected_samples, dtype=int)
