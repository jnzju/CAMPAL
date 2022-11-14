import datetime
import os
import time
import uuid
import random

import torch
import numpy as np
from datasets.builder import DATASETS
from architectures.builder import MODELS
from query_strategies.builder import STRATEGIES
from utils.config import parse_commandline_args
from utils.logger import get_logger
from utils.collect_env import collect_env
from utils.timer import Timer
import matplotlib.pyplot as plt


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(config: dict = None):
    uid = str(uuid.uuid1().hex)[:8]
    if config.work_dir is None:
        config.work_dir = os.path.join('tasks',
                                       '{}_{}_{}_no_query_{}_'
                                       'AugTypeLab_{}_AugTypeUlb_{}_Setting_{}_{}_{}'.format(
                                           config.model,
                                           config.dataset,
                                           config.strategy,
                                           config.num_query,
                                           config.aug_lab_training_mode + '_' + config.aug_lab_strength_mode,
                                           config.aug_ulb_evaluation_mode + '_' + config.aug_ulb_strength_mode,
                                           config.setting_name,
                                           datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"),
                                           uid))
    os.makedirs(config.work_dir, mode=0o777, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    config.timestamp = timestamp
    log_file = os.path.join(config.work_dir, f'{timestamp}.log')
    logger = get_logger(name='DAL', log_file=log_file)
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    if config.seed is not None:
        set_seed(config.seed)

    dataset = DATASETS.build(
        dict(type=config.dataset, initial_size=config.num_init_labels))

    n_pool = len(dataset.DATA_INFOS['train_full'])
    n_eval = len(dataset.DATA_INFOS['val'])
    n_test = len(dataset.DATA_INFOS['test'])
    logger.info('cardinality of initial labeled pool: {}'.format(config.num_init_labels))
    logger.info('cardinality of initial unlabeled pool: {}'.format(n_pool - config.num_init_labels))
    logger.info('cardinality of initial evaluation pool: {}'.format(n_eval))
    logger.info('cardinality of initial test pool: {}'.format(n_test))

    net = MODELS.build(dict(type=config.model))
    strategy = STRATEGIES.build(dict(type=config.strategy,
                                     dataset=dataset,
                                     net=net, args=config,
                                     logger=logger, timestamp=timestamp))

    logger.info('Dataset: {}'.format(config.dataset))
    logger.info('Seed: {}'.format(config.seed))
    logger.info('Strategy: {}'.format(type(strategy).__name__))

    if config.load_path is not None:
        strategy.clf.load_state_dict(torch.load(config.load_path))
        logger.info(f'Get pretrained parameters from {config.load_path}')

    strategy.run()

    plt.figure()
    plt.plot(strategy.num_labels_list, strategy.acc_test_list, 'r-*', lw=1, ms=5)
    plt.savefig(os.path.join(config.work_dir, 'acc_num_labels.png'))
    plt.clf()


if __name__ == '__main__':
    torch.set_num_threads(4)
    with Timer():
        config = parse_commandline_args()
        run(config)
