from torchvision import datasets
import numpy as np
from .builder import DATASETS
from .base_dataset import BaseDataset
from torchvision.transforms import transforms
import os.path as osp
from query_strategies.utils import UnNormalize


@DATASETS.register_module()
class tinyimagenet(BaseDataset):
    def __init__(self,
                 data_path=None,
                 initial_size=None):
        self.raw_tr = None
        self.raw_vl = None
        self.raw_te = None
        self.num_map = dict()
        super(tinyimagenet, self).__init__(data_path, initial_size)

    def load_class_map(self):
        with open(osp.join(self.DATA_PATH, 'words.txt'), 'r') as f:
            for line in f:
                key = line[:10].replace('\t','')
                val = line[10:].split(', ')
                for i in range(len(val)):
                    val[i] = val[i].replace('\n', '')
                self.num_map[key] = val[0]

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data/tiny-imagenet-200'
        self.raw_tr = datasets.ImageFolder(osp.join(self.DATA_PATH, 'train'))
        self.raw_te = datasets.ImageFolder(osp.join(self.DATA_PATH, 'val'))
        num_tr = len(self.raw_tr.targets)
        num_vl = 0
        num_te = len(self.raw_te.targets)
        class_to_idx_list = {i: [] for i in range(len(self.raw_tr.classes))}
        val_idx_list = []
        for idx, target in enumerate(self.raw_tr.targets):
            class_to_idx_list[int(target)].append(idx)
        for _, class_elem in class_to_idx_list.items():
            val_idx_list.extend(class_elem[-1000:])
            num_vl += 1000
            num_tr -= 1000

        self.DATA_INFOS['train_full'] = [{'no': i, 'img': self.raw_tr.imgs[i][0],
                                          'gt_label': self.raw_tr.targets[i]} for i in range(num_tr + num_vl)]
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), val_idx_list).tolist()
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': self.raw_te.imgs[i][0],
                                    'gt_label': self.raw_te.targets[i]} for i in range(num_te)]
        self.num_samples = num_tr + num_vl + num_te
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.load_class_map()
        self.CLASSES = [self.num_map[num] for num in self.raw_tr.classes]

    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        x_path, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']
        x = self.raw_tr.loader(x_path)
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
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                 std=[0.2302, 0.2265, 0.2262])
        ])

    @property
    def default_val_transform(self):
        return transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                 std=[0.2302, 0.2265, 0.2262])
        ])

    @property
    def inverse_transform(self):
        return transforms.Compose([
            UnNormalize(mean=[0.4802, 0.4481, 0.3975],
                        std=[0.2302, 0.2265, 0.2262]),
            transforms.ToPILImage(),
            transforms.Resize(size=64)
        ])

    def get_raw_data(self, idx, split='train'):
        transform = self.default_val_transform
        x_path = self.DATA_INFOS[split][idx]['img']
        x = self.raw_tr.loader(x_path)
        x = transform(x)
        return x
