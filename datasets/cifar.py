from torchvision import datasets
from PIL import Image
import numpy as np
from .builder import DATASETS
from .base_dataset import BaseDataset
from torchvision.transforms import transforms
from query_strategies.utils import UnNormalize


@DATASETS.register_module()
class cifar10(BaseDataset):
    def __init__(self,
                 data_path=None,
                 initial_size=None):
        super(cifar10, self).__init__(data_path, initial_size)

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data/cifar10'
        raw_tr = datasets.CIFAR10(self.DATA_PATH, train=True, download=True)
        raw_te = datasets.CIFAR10(self.DATA_PATH, train=False, download=True)
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
                                          'gt_label': raw_tr.targets[i]} for i in range(num_tr + num_vl)]
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), val_idx_list).tolist()
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': raw_te.data[i],
                                    'gt_label': raw_te.targets[i]} for i in range(num_te)]
        self.num_samples = num_tr + num_vl + num_te
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = raw_tr.classes

    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        x, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']
        x = Image.fromarray(x)
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
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])

    @property
    def default_val_transform(self):
        return transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])

    @property
    def inverse_transform(self):
        return transforms.Compose([
            UnNormalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010]),
            transforms.ToPILImage()
        ])

    def get_raw_data(self, idx, split='train'):
        transform = self.default_val_transform
        x = self.DATA_INFOS[split][idx]['img']
        x = Image.fromarray(x)
        x = transform(x)
        return x


@DATASETS.register_module()
class cifar100(cifar10):
    def __init__(self,
                 data_path=None,
                 initial_size=None):
        super(cifar100, self).__init__(data_path, initial_size)

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data/cifar100'
        raw_tr = datasets.CIFAR100(self.DATA_PATH, train=True, download=True)
        raw_te = datasets.CIFAR100(self.DATA_PATH, train=False, download=True)
        num_tr = len(raw_tr.targets)
        num_vl = 0
        num_te = len(raw_te.targets)
        class_to_idx_list = {i: [] for i in range(len(raw_tr.classes))}
        val_idx_list = []
        for idx, target in enumerate(raw_tr.targets):
            class_to_idx_list[int(target)].append(idx)
        for _, class_elem in class_to_idx_list.items():
            val_idx_list.extend(class_elem[-100:])
            num_vl += 100
            num_tr -= 100

        self.DATA_INFOS['train_full'] = [{'no': i, 'img': raw_tr.data[i],
                                          'gt_label': raw_tr.targets[i]} for i in range(num_tr + num_vl)]
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), val_idx_list).tolist()
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': raw_te.data[i],
                                    'gt_label': raw_te.targets[i]} for i in range(num_te)]
        self.num_samples = num_tr + num_vl + num_te
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = raw_tr.classes

    @property
    def default_train_transform(self):
        return transforms.Compose([
            transforms.Pad(2, padding_mode='reflect'),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])

    @property
    def default_val_transform(self):
        return transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])

    @property
    def inverse_transform(self):
        return transforms.Compose([
            UnNormalize(mean=[0.5071, 0.4867, 0.4408],
                        std=[0.2675, 0.2565, 0.2761]),
            transforms.ToPILImage()
        ])
