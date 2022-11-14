from abc import ABCMeta
import numpy as np
from torchvision.transforms import Compose


class BaseDataset(object, metaclass=ABCMeta):

    CLASSES = None
    DATA_PATH = None
    INDEX_LB = np.zeros([], dtype=bool)
    INDEX_ULB = np.zeros([], dtype=bool)
    QUERIED_HISTORY = []
    DATA_INFOS = {'train': [], 'train_full': [], 'train_u': [],
                  'val': [], 'test': []}
    TRANSFORM = {'train': Compose([]), 'train_full': Compose([]), 'train_u': Compose([]),
                 'val': Compose([]), 'test': Compose([])}
    SUBSET = 10000
    ORI_SIZE = 0

    def __init__(self,
                 data_path=None,
                 initial_size=None,
                 subset=None):
        super(BaseDataset, self).__init__()
        self.TRANSFORM = {'train': self.default_train_transform,
                          'train_full': self.default_val_transform,
                          'train_u': self.default_val_transform,
                          'val': self.default_val_transform,
                          'test': self.default_val_transform}
        self.DATA_PATH = data_path
        self.load_data()
        self.num_samples = len(self.DATA_INFOS['train_full'])
        if initial_size is None:
            initial_size = self.num_samples // 100
        if subset is None:
            self.SUBSET = max(10000, self.num_samples // 10)
        self.initialize_lb(initial_size)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_cat_ids(self, idx):
        return self.DATA_INFOS[idx]['gt_label'].astype(np.int)

    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    @property
    def default_train_transform(self):
        raise NotImplementedError

    @property
    def default_val_transform(self):
        raise NotImplementedError

    @property
    def inverse_transform(self):
        raise NotImplementedError

    def update_lb(self, new_lb):
        self.INDEX_LB[new_lb] = True
        self.DATA_INFOS['train'] = list(np.array(self.DATA_INFOS['train_full'])[self.INDEX_LB])
        self.select_ulb()
        self.QUERIED_HISTORY.append(new_lb.tolist())

    def initialize_lb(self, size):
        idxs_tmp = np.arange(len(self.DATA_INFOS['train_full']))
        self.ORI_SIZE = len(self.DATA_INFOS['train_full'])
        initial_lb = np.random.choice(idxs_tmp, size, replace=False)
        self.INDEX_LB[initial_lb] = True
        self.DATA_INFOS['train'] = list(np.array(self.DATA_INFOS['train_full'])[self.INDEX_LB])
        self.select_ulb()
        self.QUERIED_HISTORY.append(initial_lb.tolist())

    def select_ulb(self):
        U_TEMP = np.arange(len(self.DATA_INFOS['train_full']))[~self.INDEX_LB]
        if self.SUBSET >= len(self.DATA_INFOS['train_full']) - len(self.DATA_INFOS['train']):
            U_SELECTED = U_TEMP
        else:
            U_SELECTED = np.random.choice(U_TEMP, self.SUBSET, replace=False)
        self.INDEX_ULB = np.array([True if x in U_SELECTED else False
                                   for x in range(len(self.DATA_INFOS['train_full']))])
        self.DATA_INFOS['train_u'] = np.array(self.DATA_INFOS['train_full'])[self.INDEX_ULB].tolist()
