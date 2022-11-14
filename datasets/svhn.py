from torchvision import datasets
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
from .builder import DATASETS
from .base_dataset import BaseDataset
from query_strategies.utils import UnNormalize


@DATASETS.register_module()
class svhn(BaseDataset):
    def __init__(self,
                 data_path=None,
                 initial_size=None):
        super(svhn, self).__init__(data_path, initial_size)

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data/SVHN'
        raw_tr = datasets.SVHN(self.DATA_PATH, split='train', download=True)
        raw_te = datasets.SVHN(self.DATA_PATH, split='test', download=True)
        num_tr = len(raw_tr.labels)
        num_vl = 0
        num_te = len(raw_te.labels)
        class_to_idx_list = {i: [] for i in range(10)}
        val_idx_list = []
        for idx, target in enumerate(raw_tr.labels):
            class_to_idx_list[int(target)].append(idx)
        for _, class_elem in class_to_idx_list.items():
            val_idx_list.extend(class_elem[-1000:])
            num_vl += 1000
            num_tr -= 1000

        self.DATA_INFOS['train_full'] = [{'no': i, 'img': raw_tr.data[i],
                                          'gt_label': raw_tr.labels[i].item()} for i in range(num_tr + num_vl)]
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), val_idx_list).tolist()
        self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': raw_te.data[i],
                                    'gt_label': raw_te.labels[i].item()} for i in range(num_te)]
        self.num_samples = num_tr + num_vl + num_te
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = [str(i) for i in range(10)]

    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        x, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']
        x = Image.fromarray(np.transpose(x, (1, 2, 0)))
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
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                 std=[0.1980, 0.2010, 0.1970])
        ])

    @property
    def default_val_transform(self):
        return transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                 std=[0.1980, 0.2010, 0.1970])
        ])

    @property
    def inverse_transform(self):
        return transforms.Compose([
            UnNormalize(mean=[0.4377, 0.4438, 0.4728],
                        std=[0.1980, 0.2010, 0.1970]),
            transforms.ToPILImage()
        ])

    def get_raw_data(self, idx, split='train'):
        transform = self.default_val_transform
        x = self.DATA_INFOS[split][idx]['img']
        x = Image.fromarray(np.transpose(x, (1, 2, 0)))
        x = transform(x)
        return x
