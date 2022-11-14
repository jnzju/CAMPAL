from .base_dataset import BaseDataset
from torch.utils.data import Dataset, ConcatDataset
from .dataset_wrappers import RepeatDataset
from query_strategies.utils import get_one_hot_label
from query_strategies.augmentors import *
import random


class Handler(Dataset):
    def __init__(self, dataset: BaseDataset, split, transform=None, to_soft=False):
        self.dataset = dataset
        self.split = split
        self.transform = transform
        self.to_soft = to_soft

    def __getitem__(self, idx):
        if self.to_soft:
            x, y, no, idx = self.dataset.prepare_data(idx, self.split, self.transform)
            y = get_one_hot_label(y, len(self.dataset.CLASSES))
            return x, y, no, idx
        else:
            return self.dataset.prepare_data(idx, self.split, self.transform)

    def __len__(self):
        return len(self.dataset.DATA_INFOS[self.split])


class HandlerWithAug(Handler):
    def __init__(self, dataset: BaseDataset, split, transform=None, to_soft=False,
                 aug_mode='StrengthGuidedAugmentSingle', strength_mode='all', args_list=None, ablation_aug_type=None):
        super(HandlerWithAug, self).__init__(dataset, split, transform, to_soft)
        self.aug = {}
        self.aug_mode = aug_mode
        self.args_dict = []
        self.ablation_aug_type = ablation_aug_type
        self._allocate_aug_args(strength_mode, args_list)
        self._allocate_aug()

    def _allocate_aug_args(self, strength_mode, args_list):
        if args_list is None:
            self.args_dict = [{}] * len(self.dataset.DATA_INFOS[self.split])
            return

        if self.aug_mode in ['AutoAugment', 'TrivialAugmentWide']:
            self.args_dict = [{}] * len(self.dataset.DATA_INFOS[self.split])
            return
        elif self.aug_mode in ['RandAugment', 'StrengthGuidedAugmentSingle']:
            if strength_mode == 'all':
                if self.aug_mode == 'RandAugment':
                    args_dict = {'num_ops': args_list}
                else:
                    args_dict = {'num_ops': args_list, 'ablation_aug': self.ablation_aug_type}
            else:
                if self.aug_mode == 'RandAugment':
                    args_dict = [{'num_ops': int(strength)} for strength in args_list]
                else:
                    args_dict = [{'num_ops': int(strength),
                                  'ablation_aug': self.ablation_aug_type} for strength in args_list]
        else:
            raise NotImplementedError

        if strength_mode in ['sample', None]:
            self.args_dict = args_dict
        elif strength_mode == 'class':
            self.args_dict = [args_dict[int(self.dataset.DATA_INFOS[self.split][i]['gt_label'])]
                              for i in range(len(self.dataset.DATA_INFOS[self.split]))]
        elif strength_mode == 'all':
            self.args_dict = [args_dict] * len(self.dataset.DATA_INFOS[self.split])
        else:
            raise NotImplementedError

    def _allocate_aug(self):
        if self.aug_mode == 'StrengthGuidedAugmentSingle':
            self.aug = {i: StrengthGuidedAugmentSingle(**self.args_dict[i])
                        for i in range(len(self.dataset.DATA_INFOS[self.split]))}
        elif self.aug_mode == 'AutoAugment':
            if self.dataset.class__.__name__ in ['cifar10', 'cifar100']:
                policy = AutoAugmentPolicy.CIFAR10
            elif self.dataset.__class__.__name__ in ['svhn', 'fashionmnist']:
                policy = AutoAugmentPolicy.SVHN
            else:
                policy = AutoAugmentPolicy.IMAGENET
            self.aug = {i: AutoAugment(policy) for i in range(len(self.dataset.DATA_INFOS[self.split]))}
        elif self.aug_mode == 'RandAugment':
            self.aug = {i: RandAugment(**self.args_dict[i]) for i in range(len(self.dataset.DATA_INFOS[self.split]))}
        elif self.aug_mode == 'TrivialAugmentWide':
            self.aug = {i: TrivialAugmentWide() for i in range(len(self.dataset.DATA_INFOS[self.split]))}
        else:
            raise ValueError("The provided augmenter {} is not recognized.".format(self.aug_mode))

    def __getitem__(self, idx):
        if self.to_soft:
            x, y, no, idx = self.dataset.prepare_data(idx, self.split, self.transform, self.aug[idx])
            y = get_one_hot_label(y, len(self.dataset.CLASSES))
            return x, y, no, idx
        else:
            return self.dataset.prepare_data(idx, self.split, self.transform, self.aug[idx])

    def __len__(self):
        return len(self.dataset.DATA_INFOS[self.split])


class HandlerWithMix(Handler):
    def __init__(self, dataset: BaseDataset, split, transform=None, to_soft=False,
                 mix_mode='StrengthGuidedAugmentMixing', split_b=None,
                 strength_mode='all', args_list=None, num_strength_bins=4, ablation_mix_type=None):
        super(HandlerWithMix, self).__init__(dataset, split, transform, to_soft)
        self.aug = {i: None for i in range(len(self.dataset.DATA_INFOS[self.split]))}
        self.mix_mode = mix_mode
        self.args_dict = []
        self.split_b = split_b if split_b is not None else self.split
        self.ablation_aug = ablation_mix_type
        self._allocate_aug_args(strength_mode, args_list, num_strength_bins)
        self._allocate_aug()

    def _allocate_aug_args(self, strength_mode, args_list, num_strength_bins):
        if args_list is None:
            self.args_dict = [{}] * len(self.dataset.DATA_INFOS[self.split])
            return

        if self.mix_mode in ['StrengthGuidedAugmentMixing']:
            if strength_mode == 'all':
                args_dict = {'magnitude': args_list, 'num_magnitude_bins': num_strength_bins + 1,
                             'ablation_aug': self.ablation_aug}
            else:
                args_dict = [{'magnitude': int(strength), 'num_magnitude_bins': num_strength_bins + 1,
                              'ablation_aug': self.ablation_aug} for strength in args_list]
        else:
            raise NotImplementedError
        if strength_mode == 'sample':
            self.args_dict = args_dict
        elif strength_mode == 'class':
            self.args_dict = [args_dict[int(self.dataset.DATA_INFOS[self.split][i]['gt_label'])]
                              for i in range(len(self.dataset.DATA_INFOS[self.split]))]
        elif strength_mode == 'all':
            self.args_dict = [args_dict] * len(self.dataset.DATA_INFOS[self.split])
        else:
            raise NotImplementedError

    def _allocate_aug(self):
        if self.split_b is None:
            i_rand_perm = list(range(len(self.dataset.DATA_INFOS[self.split])))
        else:
            i_rand_perm = random.choices(list(range(len(self.dataset.DATA_INFOS[self.split_b]))),
                                         k=len(self.dataset.DATA_INFOS[self.split]))
        random.shuffle(i_rand_perm)
        if self.mix_mode == 'StrengthGuidedAugmentMixing':
            self.aug = {i: (i_rand_perm[i], StrengthGuidedAugmentMixing(**self.args_dict[i]))
                        for i in range(len(self.dataset.DATA_INFOS[self.split]))}
        else:
            raise ValueError("The provided augmenter {} is not recognized.".format(self.aug_type))

    def __getitem__(self, idx):
        x1, y1, no1, _ = self.dataset.prepare_data(idx, self.split, self.transform)
        x2, y2, no2, _ = self.dataset.prepare_data(self.aug[idx][0], self.split_b, self.transform)
        x, y = self.aug[idx][1](x1, x2, y1, y2, len(self.dataset.CLASSES))
        return x, y, no1, idx

    def __len__(self):
        return len(self.dataset.DATA_INFOS[self.split])


def GetHandler(dataset: BaseDataset, split: str, transform=None, repeat_times=1,
               single_aug_times=0, mix_aug_times=0, aug_mode='StrengthGuidedAugment', split_b=None,
               strength_mode='all', args_list=None, num_strength_bins=4,
               ablation_aug_type=None, ablation_mix_type=None):
    to_soft = False
    if mix_aug_times > 0:
        to_soft = True
    h_list = [RepeatDataset(Handler(dataset, split, transform, to_soft), repeat_times)]
    for _ in range(single_aug_times):
        if aug_mode == 'StrengthGuidedAugment':
            aug_mode_detail = 'StrengthGuidedAugmentSingle'
        else:
            aug_mode_detail = aug_mode
        h_list.append(HandlerWithAug(dataset, split, transform, to_soft,
                                     aug_mode_detail, strength_mode, args_list, ablation_aug_type))
    for _ in range(mix_aug_times):
        if aug_mode == 'StrengthGuidedAugment':
            mix_mode_detail = 'StrengthGuidedAugmentMixing'
        else:
            break
        h_list.append(HandlerWithMix(dataset, split, transform, to_soft, mix_mode_detail,
                                     split_b, strength_mode, args_list, num_strength_bins, ablation_mix_type))
    h = ConcatDataset(h_list)
    return h
