from torchvision import datasets
import numpy as np
from .builder import DATASETS
from .base_dataset import BaseDataset
from torchvision.transforms import transforms
import os.path as osp
from query_strategies.utils import UnNormalize
from itertools import chain


@DATASETS.register_module()
class caltech101(BaseDataset):
    def __init__(self, data_path=None, initial_size=None):
        self.raw_all = None
        self.raw_tr = None
        self.raw_vl = None
        self.raw_te = None
        self.num_map = dict()
        super(caltech101, self).__init__(data_path, initial_size)

    def split_data(self):
        assert self.raw_all is not None
        cls_to_idx_end = {i: 0 for i in range(102)}
        for rank, (_, cls_idx) in enumerate(self.raw_all):
            cls_to_idx_end[cls_idx] = rank + 1
        val_idxs = list(chain(*[list(range(cls_to_idx_end_elem - 20, cls_to_idx_end_elem - 10))
                                for cls_to_idx_end_elem in cls_to_idx_end.values()]))
        test_idxs = list(chain(*[list(range(cls_to_idx_end_elem - 10, cls_to_idx_end_elem))
                                 for cls_to_idx_end_elem in cls_to_idx_end.values()]))
        return val_idxs, test_idxs

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data/Caltech-101'
        self.raw_all = datasets.ImageFolder(osp.join(self.DATA_PATH, 'data'))
        val_idxs, test_idxs = self.split_data()
        train_idxs = list(set(range(len(self.raw_all))) - set(val_idxs) - set(test_idxs))
        num_tr = len(self.raw_all) - len(val_idxs) - len(test_idxs)
        num_vl = len(val_idxs)
        num_te = len(test_idxs)
        self.DATA_INFOS['train_full'] = [{'no': i, 'img': self.raw_all.imgs[sub][0],
                                          'gt_label': self.raw_all.targets[sub]} for i, sub in enumerate(train_idxs)]
        self.DATA_INFOS['val'] = [{'no': - (i + 1), 'img': self.raw_all.imgs[sub][0],
                                   'gt_label': self.raw_all.targets[sub]} for i, sub in enumerate(val_idxs)]
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': self.raw_all.imgs[sub][0],
                                    'gt_label': self.raw_all.targets[sub]} for i, sub in enumerate(test_idxs)]
        self.num_samples = num_tr + num_vl + num_te
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = self.raw_all.classes

    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        x_path, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']
        x = self.raw_all.loader(x_path)
        if aug_transform is not None:
            x = aug_transform(x)
        if transform is None:
            x = self.TRANSFORM[split](x)
        else:
            x = transform(x)
        return x, y, self.DATA_INFOS[split][idx]['no'], idx

    @property
    def default_train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def default_val_transform(self):
        return transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def inverse_transform(self):
        return transforms.Compose([
            UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToPILImage(),
            transforms.Resize(size=256)
        ])

    def get_raw_data(self, idx, split='train'):
        transform = self.default_val_transform
        x_path = self.DATA_INFOS[split][idx]['img']
        x = self.raw_tr.loader(x_path)
        x = transform(x)
        return x


@DATASETS.register_module()
class caltech256(BaseDataset):
    def __init__(self, data_path=None, initial_size=None):
        self.raw_all = None
        self.raw_tr = None
        self.raw_vl = None
        self.raw_te = None
        self.num_map = dict()
        super(caltech256, self).__init__(data_path, initial_size)

    def split_data(self):
        assert self.raw_all is not None
        cls_to_idx_end = {i: 0 for i in range(257)}
        for rank, (_, cls_idx) in enumerate(self.raw_all):
            cls_to_idx_end[cls_idx] = rank + 1
        val_idxs = list(chain(*[list(range(cls_to_idx_end_elem - 20, cls_to_idx_end_elem - 10))
                                for cls_to_idx_end_elem in cls_to_idx_end.values()]))
        test_idxs = list(chain(*[list(range(cls_to_idx_end_elem - 10, cls_to_idx_end_elem))
                                 for cls_to_idx_end_elem in cls_to_idx_end.values()]))
        return val_idxs, test_idxs

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data/Caltech-256'
        self.raw_all = datasets.ImageFolder(osp.join(self.DATA_PATH, 'data'))
        val_idxs, test_idxs = self.split_data()
        train_idxs = list(set(range(len(self.raw_all))) - set(val_idxs) - set(test_idxs))
        num_tr = len(self.raw_all) - len(val_idxs) - len(test_idxs)
        num_vl = len(val_idxs)
        num_te = len(test_idxs)
        self.DATA_INFOS['train_full'] = [{'no': i, 'img': self.raw_all.imgs[sub][0],
                                          'gt_label': self.raw_all.targets[sub]} for i, sub in enumerate(train_idxs)]
        self.DATA_INFOS['val'] = [{'no': - (i + 1), 'img': self.raw_all.imgs[sub][0],
                                   'gt_label': self.raw_all.targets[sub]} for i, sub in enumerate(val_idxs)]
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': self.raw_all.imgs[sub][0],
                                    'gt_label': self.raw_all.targets[sub]} for i, sub in enumerate(test_idxs)]
        self.num_samples = num_tr + num_vl + num_te
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = self.raw_all.classes

    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        x_path, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']
        x = self.raw_all.loader(x_path)
        if aug_transform is not None:
            x = aug_transform(x)
        if transform is None:
            x = self.TRANSFORM[split](x)
        else:
            x = transform(x)
        return x, y, self.DATA_INFOS[split][idx]['no'], idx

    @property
    def default_train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def default_val_transform(self):
        return transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def inverse_transform(self):
        return transforms.Compose([
            UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToPILImage(),
            transforms.Resize(size=256)
        ])

    def get_raw_data(self, idx, split='train'):
        transform = self.default_val_transform
        x_path = self.DATA_INFOS[split][idx]['img']
        x = self.raw_tr.loader(x_path)
        x = transform(x)
        return x
