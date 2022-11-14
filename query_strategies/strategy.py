import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import os
from getpass import getuser
from socket import gethostname
from utils.progressbar import track_iter_progress
from utils.text import TextLogger
from utils.timer import Timer
from .builder import STRATEGIES
from datasets.dataloader import GetHandler, Handler
from copy import deepcopy
from datasets.base_dataset import BaseDataset
from evaluation import *
from .utils import get_initialized_module, get_lr
import csv


@STRATEGIES.register_module()
class Strategy:
    def __init__(self, dataset: BaseDataset, net, args, logger, timestamp):
        self.dataset = dataset
        self.net = net
        self.args = args
        self.clf, self.optimizer, self.scheduler = None, None, None
        self.init_clf()
        self.cycle_info = None
        self.init_cycle_info()
        self.cycle = 0
        self.epoch = 0
        self.logger = logger
        self.TextLogger = TextLogger(self.clf, vars(args), logger)
        self.timer = Timer()
        self.timestamp = timestamp
        self.acc_val_list = []
        self.acc_test_list = []
        self.num_labels_list = []
        self.TextLogger._dump_log(vars(args))

    def init_clf(self):
        self.clf, self.optimizer, self.scheduler = \
            get_initialized_module(self.net, self.args.lr, self.args.momentum, self.args.weight_decay,
                                   self.args.milestones, num_classes=len(self.dataset.CLASSES))

    def init_cycle_info(self):
        self.cycle_info = [{'no': i,
                            'class': self.dataset.CLASSES[int(self.dataset.DATA_INFOS['train_full'][i]['gt_label'])],
                            'label': int(self.dataset.INDEX_LB[i]),
                            'queried': 0,
                            'score': 0}
                           for i in range(len(self.dataset.DATA_INFOS['train_full']))]

    def query(self, n):
        raise NotImplementedError

    def update(self, n, aug_args_list=None):
        if n == 0:
            return None

        if aug_args_list is None:
            if self.args.aug_ulb_evaluation_mode in ['StrengthGuidedAugment', 'RandAugment']:
                aug_args_list = self.augment_optimizer_unlab()

        idxs_q = self.query(n, aug_args_list)
        idxs_q = np.arange(len(self.dataset.DATA_INFOS['train_full']))[self.dataset.INDEX_ULB][idxs_q]
        self.dataset.update_lb(idxs_q)
        return idxs_q

    def _train(self, loader_tr, clf_group: dict, clf_name='train', soft_target=False, log_show=True):
        iter_out = self.args.out_iter_freq
        loss_list = []
        right_count_list = []
        samples_per_batch = []
        clf_group['clf'].train()
        for batch_idx, (x, y, _, _) in enumerate(loader_tr):
            x, y = x.cuda(), y.cuda()
            clf_group['optimizer'].zero_grad()
            out, _, _ = self.clf(x)
            if soft_target:
                pred_compare = out.max(1)[1]
                y_compare = y.max(1)[1]
                right_count_list.append((pred_compare == y_compare).sum().item())
                samples_per_batch.append(len(y))
            else:
                pred = out.max(1)[1]
                right_count_list.append((pred == y).sum().item())
                samples_per_batch.append(len(y))
            loss = F.cross_entropy(out, y)
            loss_list.append(loss.item())
            loss.backward()
            clf_group['optimizer'].step()
            iter_time = self.timer.since_last_check()
            if log_show:
                if (batch_idx + 1) % iter_out == 0:
                    log_dict = dict(
                        mode=clf_name,
                        epoch=self.epoch,
                        iter=batch_idx + 1,
                        lr=get_lr(clf_group['optimizer']),
                        time=iter_time,
                        acc=1.0 * np.sum(right_count_list[-iter_out:]) / np.sum(samples_per_batch[-iter_out:]),
                        loss=np.sum(loss_list[-iter_out:])
                    )
                    self.TextLogger.log(
                        log_dict=log_dict,
                        iters_per_epoch=len(loader_tr),
                        iter_count=self.epoch * len(loader_tr) + batch_idx,
                        max_iters=self.args.n_epoch * len(loader_tr),
                        interval=iter_out
                    )
        clf_group['scheduler'].step()

    def augment_optimizer_label(self, metric='loss', split_guide='val', split_train='train'):
        if self.args.aug_strength_lab is not None:
            if self.args.aug_lab_strength_mode == 'sample':
                return torch.ones([len(self.dataset.DATA_INFOS[split_train])]) * self.args.aug_strength_lab
            elif self.args.aug_lab_strength_mode == 'class':
                return torch.ones([len(self.dataset.CLASSES)]) * self.args.aug_strength_lab
            elif self.args.aug_lab_strength_mode == 'all':
                return self.args.aug_strength_lab
            else:
                raise NotImplementedError
        self.clf.train()
        dataset_tr_init = GetHandler(self.dataset, split_guide, self.dataset.default_train_transform)
        loader_tr_init = DataLoader(dataset_tr_init, shuffle=True, batch_size=self.args.batch_size,
                                    num_workers=self.args.num_workers)
        while self.epoch < self.args.n_epoch:
            self.timer.since_last_check()
            self._train(loader_tr_init, {'clf': self.clf, 'optimizer': self.optimizer, 'scheduler': self.scheduler})
            if self.epoch % self.args.save_freq == 0 and self.epoch > 0:
                pass
            self.epoch += 1
        self.epoch = 0

        sample_strength_matrix = torch.zeros([self.args.num_strength_bins_lab,
                                              len(self.dataset.DATA_INFOS[split_train])]).cuda()
        for strength in range(1, self.args.num_strength_bins_lab+1):
            dataset = GetHandler(self.dataset, split_train, self.dataset.default_val_transform, repeat_times=1,
                                 single_aug_times=2, mix_aug_times=1,
                                 aug_mode=self.args.aug_lab_training_mode,
                                 strength_mode="sample",
                                 num_strength_bins=self.args.num_strength_bins_lab,
                                 args_list=torch.ones([len(self.dataset.DATA_INFOS[split_train])]) * strength,
                                 ablation_aug_type=self.args.aug_type_lab, ablation_mix_type=self.args.mix_type_lab)
            scores = self.predict(self.clf, dataset, metric=metric, log_show=False, split_info='train_optim')
            sample_strength_matrix[strength - 1, :] = \
                torch.sum(scores.reshape([len(self.dataset.DATA_INFOS[split_train]), -1]), dim=1)
        if self.args.aug_lab_strength_mode == 'sample':
            return torch.argmin(sample_strength_matrix, dim=0).cpu() + 1
        elif self.args.aug_lab_strength_mode == 'class':
            idx_to_class = torch.tensor([int(elem['gt_label']) for elem in self.dataset.DATA_INFOS[split_train]]).cuda()
            class_sample_strength_matrix = torch.zeros(self.args.num_strength_bins_lab,
                                                       len(self.dataset.CLASSES)).cuda()
            for i in range(len(self.dataset.CLASSES)):
                class_sample_strength_matrix[:, i] += torch.sum(sample_strength_matrix[:, idx_to_class == i], dim=1)
            return torch.argmin(class_sample_strength_matrix, dim=0).cpu() + 1
        else:
            new_matrix = torch.sum(sample_strength_matrix, dim=1)
            return torch.argmin(new_matrix).cpu().item() + 1

    def augment_optimizer_unlab(self, metric='entropy'):
        if self.args.aug_strength_ulb is not None:
            if self.args.aug_ulb_strength_mode == 'sample':
                return torch.ones([len(self.dataset.DATA_INFOS['train_u'])]) * self.args.aug_strength_ulb
            elif self.args.aug_ulb_strength_mode == 'class':
                return torch.ones([len(self.dataset.CLASSES)]) * self.args.aug_strength_ulb
            elif self.args.aug_ulb_strength_mode == 'all':
                return self.args.aug_strength_ulb
            else:
                raise NotImplementedError
        sample_strength_matrix = torch.zeros([self.args.num_strength_bins_ulb,
                                              len(self.dataset.DATA_INFOS['train_u'])]).cuda()
        for strength in range(1, self.args.num_strength_bins_ulb + 1):
            dataset = GetHandler(self.dataset, 'train_u', self.dataset.default_val_transform, repeat_times=1,
                                 single_aug_times=2, mix_aug_times=1,
                                 aug_mode=self.args.aug_ulb_evaluation_mode,
                                 strength_mode="sample",
                                 num_strength_bins=self.args.num_strength_bins_ulb,
                                 args_list=torch.ones([len(self.dataset.DATA_INFOS['train_u'])]) * strength,
                                 ablation_aug_type=self.args.aug_type_ulb, ablation_mix_type=self.args.mix_type_ulb)
            scores = self.predict(self.clf, dataset, metric=metric, log_show=False, split_info='train_u_optim')
            sample_strength_matrix[strength - 1, :] = \
                torch.min(scores.reshape([len(self.dataset.DATA_INFOS['train_u']), -1]), dim=1)[0]
        if self.args.aug_ulb_strength_mode == 'sample':
            return torch.argmax(sample_strength_matrix, dim=0) + 1
        elif self.args.aug_ulb_strength_mode == 'class':
            self.get_vt_label()
            idx_to_class = torch.tensor([int(elem['vt_label']) for elem in self.dataset.DATA_INFOS['train_u']]).cuda()
            class_sample_strength_matrix = torch.zeros(self.args.num_strength_bins_ulb, len(self.dataset.CLASSES)).cuda()
            for i in range(len(self.dataset.CLASSES)):
                class_sample_strength_matrix[:, i] += torch.sum(sample_strength_matrix[:, idx_to_class == i], dim=1)
            return torch.argmax(class_sample_strength_matrix, dim=0) + 1
        else:
            new_matrix = torch.sum(sample_strength_matrix, dim=1)
            return torch.argmax(new_matrix).item() + 1

    def train(self, strength_mode=None, args_list=None):
        if args_list is None:
            if self.args.aug_lab_training_mode in ['StrengthGuidedAugment', 'RandAugment']:
                args_list = self.augment_optimizer_label()
        if strength_mode is None:
            strength_mode = self.args.aug_lab_strength_mode

        self.logger.info('Start running, host: %s, work_dir: %s',
                         f'{getuser()}@{gethostname()}', self.args.work_dir)
        self.logger.info('max: %d epochs', self.args.n_epoch)
        self.init_clf()
        dataset_tr = GetHandler(self.dataset, 'train', self.dataset.default_train_transform, 1,
                                self.args.aug_ratio_lab, self.args.mix_ratio_lab, self.args.aug_lab_training_mode,
                                strength_mode=strength_mode, args_list=args_list,
                                ablation_aug_type=self.args.aug_type_lab, ablation_mix_type=self.args.mix_type_lab)
        loader_tr = DataLoader(dataset_tr, shuffle=True, batch_size=self.args.batch_size,
                               num_workers=self.args.num_workers)

        while self.epoch < self.args.n_epoch:
            self.timer.since_last_check()
            self._train(loader_tr, {'clf': self.clf, 'optimizer': self.optimizer, 'scheduler': self.scheduler},
                        soft_target=True if (self.args.mix_ratio_lab > 0) else False)
            if self.epoch % self.args.save_freq == 0 and self.epoch > 0:
                pass
            self.epoch += 1
        self.epoch = 0

    def run(self):
        while self.cycle < self.args.n_cycle:
            active_path = os.path.join(self.args.work_dir, f'active_cycle_{self.cycle}')
            os.makedirs(active_path, mode=0o777, exist_ok=True)
            num_labels = len(self.dataset.DATA_INFOS['train'])
            self.logger.info(f'Active Round {self.cycle} with {num_labels} labeled instances')
            active_meta_log_dict = dict(
                mode='active_meta',
                cycle=self.cycle,
                num_labels=num_labels
            )
            self.TextLogger._dump_log(active_meta_log_dict)
            if not self.args.updating:
                self.init_clf()
            self.init_cycle_info()
            self.train()
            dataset_val = GetHandler(self.dataset, 'val', self.dataset.default_val_transform)
            dataset_test = GetHandler(self.dataset, 'test', self.dataset.default_val_transform)
            self.acc_val_list.append(self.predict(self.clf, dataset_val, split_info='val'))
            self.acc_test_list.append(self.predict(self.clf, dataset_test, split_info='test'))
            self.num_labels_list.append(num_labels)
            self.update(self.args.num_query)
            for idx in self.dataset.QUERIED_HISTORY[-1]:
                self.cycle_info[idx]['queried'] = 1
            self.cycle += 1
        self.record_test_accuracy()

    def predict(self, clf, dataset: Handler, metric='accuracy',
                topk=None, n_drop=None, thrs=None, dropout_split=False, log_show=True, split_info='train'):

        loader = DataLoader(dataset, shuffle=False, num_workers=self.args.num_workers)

        if isinstance(clf, torch.nn.Module):
            clf.eval()
        if n_drop is None:
            n_drop = 1
        if topk is None:
            topk = 1
        if thrs is None:
            thrs = 0.
        if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'support', 'loss']:
            self.logger.info(f"Calculating Performance with {metric}...")
            pred = []
            target = []
            with torch.no_grad():
                for x, y, _, idxs in track_iter_progress(loader):
                    x, y = x.cuda(), y.cuda()
                    if isinstance(clf, torch.nn.Module):
                        out, _, _ = clf(x)
                    else:
                        out = clf(x)
                    prob = F.softmax(out, dim=1)
                    pred.append(prob)
                    target.append(y)
            pred = torch.cat(pred).cuda()
            target = torch.cat(target).cuda()
            if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'support']:
                if metric == 'accuracy':
                    result = accuracy(pred, target, topk, thrs)
                elif metric == 'precision':
                    result = precision(pred, target, thrs=thrs)
                elif metric == 'recall':
                    result = recall(pred, target, thrs=thrs)
                elif metric == 'f1_score':
                    result = f1_score(pred, target, thrs=thrs)
                elif metric == 'support':
                    result = support(pred, target)
                else:
                    raise Exception(f"Metric {metric} not implemented!")
                if len(result) == 1:
                    result = result.item()
                else:
                    result = result.numpy().tolist()
                if log_show:
                    log_dict = dict(mode=split_info, cycle=self.cycle)
                    log_dict[metric] = result
                    self.TextLogger.log(log_dict)
            else:
                result = F.cross_entropy(pred, target, reduction='none')
        else:
            self.logger.info(f"Calculating Informativeness with {metric}...")
            if isinstance(clf, torch.nn.Module):
                clf.train()
            if dropout_split is False:
                pred = []
                for i in range(n_drop):
                    self.logger.info('n_drop {}/{}'.format(i + 1, n_drop))
                    with torch.no_grad():
                        for batch_idx, (x, _, _, _) in enumerate(track_iter_progress(loader)):
                            x = x.cuda()
                            if isinstance(clf, torch.nn.Module):
                                out, _, _ = clf(x)
                            else:
                                out = clf(x)
                            if i == 0:
                                pred.append(F.softmax(out, dim=1))
                            else:
                                pred[batch_idx] += F.softmax(out, dim=1)
                pred = torch.cat(pred).cuda()
                pred /= n_drop
                if metric == 'entropy':
                    log_pred = torch.log(pred)
                    result = - (pred * log_pred).sum(1)
                elif metric == 'lc':
                    result = 1.0 - pred.max(1)[0]
                elif metric == 'margin':
                    pred_sorted, _ = pred.sort(descending=True)
                    result = 1.0 - (pred_sorted[:, 0] - pred_sorted[:, 1])
                elif metric == 'prob':
                    result = pred
                else:
                    raise Exception(f"Metric {metric} not implemented!")
            else:
                print("No metric will be used in dropout split mode!")
                data_length = len(self.dataset)
                result = torch.zeros([n_drop, data_length, len(self.dataset.CLASSES)]).cuda()
                for i in range(n_drop):
                    self.logger.info('n_drop {}/{}'.format(i + 1, n_drop))
                    with torch.no_grad():
                        for x, _, _, idxs in track_iter_progress(loader):
                            x = x.cuda()
                            if isinstance(clf, torch.nn.Module):
                                out, _, _ = clf(x)
                            else:
                                out = clf(x)
                            result[i][idxs] += F.softmax(out, dim=1)
        return result

    def get_embedding(self, clf, dataset, embed_type='default'):
        loader = DataLoader(dataset, shuffle=False, num_workers=self.args.num_workers)

        clf.eval()
        self.logger.info(f"Extracting embedding of type {embed_type}...")
        embdim = self.get_embedding_dim()
        nlabs = len(self.dataset.CLASSES)
        if embed_type == 'default':
            embedding = []
            with torch.no_grad():
                for x, _, _, idxs in track_iter_progress(loader):
                    x = x.cuda()
                    _, e1, _ = clf(x)
                    embedding.append(e1)
            embedding = torch.cat(embedding).cuda()
        elif embed_type == 'grad':
            data_length = len(dataset)
            embedding = np.zeros([data_length, embdim * nlabs])
            for x, y, _, idxs in loader:
                x = x.cuda()
                cout, e, _ = clf(x)
                out = e.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nlabs):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embdim * c: embdim * (c + 1)] = \
                                deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embdim * c: embdim * (c + 1)] = \
                                deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
        else:
            raise Exception(f'Embedding of type {embed_type} not implemented!')
        return embedding

    def get_embedding_dim(self) -> int:
        dataset = GetHandler(self.dataset, 'train', self.dataset.default_val_transform, 1)
        loader = DataLoader(dataset, shuffle=False, batch_size=self.args.batch_size,
                            num_workers=self.args.num_workers)
        self.clf.eval()
        with torch.no_grad():
            for x, _, _, _ in loader:
                x = x.cuda()
                _, e1, _ = self.clf(x)
                return e1.shape[1]

    def get_vt_label(self):
        dataset = GetHandler(self.dataset, 'train_u', self.dataset.default_val_transform)
        loader = DataLoader(dataset, shuffle=False, num_workers=self.args.num_workers)
        self.logger.info(f"Calculating virtual labels for with unlabeled samples.")
        pred = []
        with torch.no_grad():
            for x, _, _, _ in track_iter_progress(loader):
                x = x.cuda()
                if isinstance(self.clf, torch.nn.Module):
                    out, _, _ = self.clf(x)
                else:
                    out = self.clf(x)
                pred_elem = torch.argmax(out, dim=1)
                pred.append(pred_elem)
        pred = torch.cat(pred)
        for idx, vt_label in enumerate(pred):
            self.dataset.DATA_INFOS['train_u'][idx]['vt_label'] = int(vt_label)

    def save(self):
        model_out_path = Path(os.path.join(self.args['work_dir'], f'active_round_{self.cycle}'))
        state = self.clf.state_dict(),
        if not model_out_path.exists():
            model_out_path.mkdir()
        save_target = model_out_path / f"active_round_{self.cycle}-" \
                                       f"label_num_{np.sum(self.idxs_lb).item()}-epoch_{self.epoch}.pth"
        torch.save(state, save_target)

        self.logger.info('==> save model to {}'.format(save_target))

    def record_test_accuracy(self):
        file_name = os.path.join(self.args.work_dir, 'accuracy.csv')
        header = ['num_labels', 'accuracy']
        with open(file_name, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(header)
            for i, acc in enumerate(self.acc_test_list):
                f_csv.writerow([(i + 1) * self.args.num_query, acc])
