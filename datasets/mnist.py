import numpy as np
from torchvision import datasets
from PIL import Image
from torchvision.transforms import transforms
from .builder import DATASETS
from .base_dataset import BaseDataset
from query_strategies.utils import UnNormalize


@DATASETS.register_module()
class mnist(BaseDataset):
    def __init__(self,
                 data_path=None,
                 initial_size=None):
        super(mnist, self).__init__(data_path, initial_size)

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data'
        raw_tr = datasets.MNIST(self.DATA_PATH, train=True, download=True)
        raw_te = datasets.MNIST(self.DATA_PATH, train=False, download=True)
        num_tr = len(raw_tr.targets)
        num_vl = 0
        num_te = len(raw_te.targets)
        class_to_idx_list = {i: [] for i in range(len(raw_tr.classes))}
        val_idx_list = []
        for idx, target in enumerate(raw_tr.targets):
            class_to_idx_list[int(target)].append(idx)
        for _, class_elem in class_to_idx_list.items():
            val_idx_list.extend(class_elem[-1000:])
            num_vl += 1000
            num_tr -= 1000

        self.DATA_INFOS['train_full'] = [{'no': i, 'img': raw_tr.data[i],
                                          'gt_label': raw_tr.targets[i].item()} for i in range(num_tr + num_vl)]
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), val_idx_list).tolist()
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': raw_te.data[i],
                                    'gt_label': raw_te.targets[i].item()} for i in range(num_te)]
        self.num_samples = num_tr + num_te
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = raw_tr.classes

    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        x, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']
        x = Image.fromarray(x.numpy(), mode='L').convert('RGB')
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
            transforms.Pad(2, padding_mode='reflect'),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.1307,), (0.3081,))])

    @property
    def default_val_transform(self):
        return self.default_train_transform

    @property
    def inverse_transform(self):
        return transforms.Compose([
            UnNormalize((0.1307,), (0.3081,)),
            transforms.ToPILImage(),
            transforms.CenterCrop(size=28),
            transforms.Grayscale()
        ])

    def get_raw_data(self, idx, split='train'):
        transform = self.default_val_transform
        x = self.DATA_INFOS[split][idx]['img']
        x = Image.fromarray(x.numpy(), mode='L')
        x = transform(x)
        return x


@DATASETS.register_module()
class fashionmnist(mnist):
    def __init__(self,
                 data_path=None,
                 initial_size=None):
        super(fashionmnist, self).__init__(data_path, initial_size)

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data'
        raw_tr = datasets.FashionMNIST(self.DATA_PATH, train=True, download=True)
        raw_te = datasets.FashionMNIST(self.DATA_PATH, train=False, download=True)
        num_tr = len(raw_tr.targets)
        num_vl = 0
        num_te = len(raw_te.targets)
        class_to_idx_list = {i: [] for i in range(len(raw_tr.classes))}
        val_idx_list = []
        for idx, target in enumerate(raw_tr.targets):
            class_to_idx_list[int(target)].append(idx)
        for _, class_elem in class_to_idx_list.items():
            val_idx_list.extend(class_elem[-1000:])
            num_vl += 1000
            num_tr -= 1000

        self.DATA_INFOS['train_full'] = [{'no': i, 'img': raw_tr.data[i], 'gt_label': raw_tr.targets[i].item()}
                                         for i in range(num_tr + num_vl)]
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), val_idx_list).tolist()
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': raw_te.data[i], 'gt_label': raw_te.targets[i].item()}
                                   for i in range(num_te)]
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = raw_tr.classes

    @property
    def default_train_transform(self):
        return transforms.Compose([
            transforms.Pad(2, padding_mode='reflect'),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.2861,), (0.3530,))])

    @property
    def inverse_transform(self):
        return transforms.Compose([
            UnNormalize((0.2861,), (0.3530,)),
            transforms.ToPILImage(),
            transforms.CenterCrop(size=28),
            transforms.Grayscale()
        ])


@DATASETS.register_module()
class emnist(mnist):
    def __init__(self,
                 data_path=None,
                 initial_size=None):
        super(emnist, self).__init__(data_path, initial_size)

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data'
        raw_tr = datasets.EMNIST(self.DATA_PATH, split='byclass', train=True, download=True)
        raw_te = datasets.EMNIST(self.DATA_PATH, split='byclass', train=False, download=True)
        num_tr = len(raw_tr.targets)
        num_vl = 0
        num_te = len(raw_te.targets)
        class_to_idx_list = {i: [] for i in range(len(raw_tr.classes))}
        val_idx_list = []
        for idx, target in enumerate(raw_tr.targets):
            class_to_idx_list[int(target)].append(idx)
        for _, class_elem in class_to_idx_list.items():
            val_idx_list.extend(class_elem[-1000:])
            num_vl += 1000
            num_tr -= 1000

        self.DATA_INFOS['train_full'] = [{'no': i, 'img': raw_tr.data[i], 'gt_label': raw_tr.targets[i].item()}
                                         for i in range(num_tr + num_vl)]
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), val_idx_list).tolist()
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': raw_te.data[i], 'gt_label': raw_te.targets[i].item()}
                                   for i in range(num_te)]
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = raw_tr.classes


@DATASETS.register_module()
class kmnist(mnist):
    def __init__(self,
                 data_path=None,
                 initial_size=None):
        super(kmnist, self).__init__(data_path, initial_size)

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data'
        raw_tr = datasets.KMNIST(self.DATA_PATH, train=True, download=True)
        raw_te = datasets.KMNIST(self.DATA_PATH, train=False, download=True)
        num_tr = len(raw_tr.targets)
        num_vl = 0
        num_te = len(raw_te.targets)
        class_to_idx_list = {i: [] for i in range(len(raw_tr.classes))}
        val_idx_list = []
        for idx, target in enumerate(raw_tr.targets):
            class_to_idx_list[int(target)].append(idx)
        for _, class_elem in class_to_idx_list.items():
            val_idx_list.extend(class_elem[-1000:])
            num_vl += 1000
            num_tr -= 1000

        self.DATA_INFOS['train_full'] = [{'no': i, 'img': raw_tr.data[i], 'gt_label': raw_tr.targets[i].item()}
                                         for i in range(num_tr + num_vl)]
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), val_idx_list).tolist()
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': raw_te.data[i], 'gt_label': raw_te.targets[i].item()}
                                   for i in range(num_te)]
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = raw_tr.classes


@DATASETS.register_module()
class qmnist(mnist):
    def __init__(self,
                 data_path=None,
                 initial_size=None):
        super(qmnist, self).__init__(data_path, initial_size)

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data'
        raw_tr = datasets.QMNIST(self.DATA_PATH, train=True, download=True)
        raw_te = datasets.QMNIST(self.DATA_PATH, train=False, download=True)
        num_tr = len(raw_tr.targets)
        num_vl = 0
        num_te = len(raw_te.targets)
        class_to_idx_list = {i: [] for i in range(len(raw_tr.classes))}
        val_idx_list = []
        for idx, target in enumerate(raw_tr.targets):
            class_to_idx_list[int(target)].append(idx)
        for _, class_elem in class_to_idx_list.items():
            val_idx_list.extend(class_elem[-1000:])
            num_vl += 1000
            num_tr -= 1000

        self.DATA_INFOS['train_full'] = [{'no': i, 'img': raw_tr.data[i], 'gt_label': raw_tr.targets[i][0].item()}
                                         for i in range(num_tr + num_vl)]
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), val_idx_list).tolist()
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': raw_te.data[i], 'gt_label': raw_te.targets[i][0].item()}
                                   for i in range(num_te)]
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = raw_tr.classes
