import datetime
import os.path as osp
from collections import OrderedDict
from utils.in_out import dump

import torch


class TextLogger(object):

    def __init__(self, model, args, logger):
        self.start_iter = 0
        self.model = model
        self.args = args
        self.logger = logger
        timestamp = args['timestamp']
        self.json_log_path = osp.join(args['work_dir'],
                                      f'{timestamp}.log.json')
        self.time_sec_tot = 0

    def _get_max_memory(self):
        device = getattr(self.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        return mem_mb.item()

    def _log_info(self, log_dict, iters_per_epoch=None, max_iters=None, iter_count=None, interval=None):

        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'

            log_str = f'Epoch [{log_dict["epoch"]}]' \
                      f'[{log_dict["iter"]}/{iters_per_epoch}]\t'
            log_str += f'{lr_str}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * interval)
                time_sec_avg = self.time_sec_tot / (
                    iter_count - self.start_iter + 1)
                eta_sec = time_sec_avg * (max_iters - iter_count - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, '
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            log_str = f'Epoch({log_dict["mode"]}) \t'

        log_items = []
        for name, val in log_dict.items():
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)

        self.logger.info(log_str)

    def _dump_log(self, log_dict):
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)
        with open(self.json_log_path, 'a+') as f:
            dump(json_log, f, file_format='json')
            f.write('\n')

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def log(self, log_dict, iters_per_epoch=None, max_iters=None, iter_count=None, interval=None):

        if torch.cuda.is_available():
            log_dict['memory'] = self._get_max_memory()

        self._log_info(log_dict, iters_per_epoch, max_iters, iter_count, interval)
        self._dump_log(log_dict)
        return log_dict
