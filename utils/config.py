import argparse

__all__ = ['parse_commandline_args']


def create_parser():
    parser = argparse.ArgumentParser(description='Deep active learning args --PyTorch ')

    parser.add_argument('--work-dir', default=None, type=str, help='the dir to save logs and models')
    parser.add_argument('--save-freq', default=100, type=int, metavar='EPOCHS',
                        help='checkpoint frequency(default: 100)')

    parser.add_argument('--model', default='resnet18', metavar='MODEL')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='DATASET',
                        help='The name of the used dataset(default: cifar10)')
    parser.add_argument('--load-path', type=str, default=None, help='which pth file to preload')
    parser.add_argument('--setting-name', type=str, default=None, help='The setting name')

    parser.add_argument('--strategy', type=str, default='EntropySampling',
                        help='which sampling strategy to choose')
    parser.add_argument('--num-init-labels', default=100, type=int,
                        metavar='N', help='number of initial labeled samples(default: 100)')
    parser.add_argument('--n-cycle', default=2, type=int,
                        metavar='N', help='number of query rounds(default: 10)')
    parser.add_argument('--num-query', default=100, type=int,
                        metavar='N', help='number of query samples per epoch(default: 100)')
    parser.add_argument('--subset', default=10000, type=int,
                        metavar='N', help='the size of the unlabeled pool to query, subsampling')
    parser.add_argument('--updating', action='store_true', help='Whether to use updating or retraining')
    parser.add_argument('--n-epoch', default=2, type=int, metavar='N',
                        help='number of total training epochs(default: 100)')

    parser.add_argument('--dataset-imbalance-mode', default=None, type=str, help='Imbalance mode for the dataset used')
    parser.add_argument('--init-imbalance-mode', default=None, type=str,
                        help='Imbalance mode for the initial labeled pool')

    parser.add_argument('--batch-size', type=int, default=50, metavar='BATCH_SIZE',
                        help='Batch size in both train and test phase(default: 64)')
    parser.add_argument('--num-workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='max learning rate (default: 0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 0.0001)')
    parser.add_argument('--milestones', default=[100, 180, 240], type=int, nargs='+',
                        help='milestones of learning scheduler to use '
                             '(default: [100, 180, 240])')
    parser.add_argument('--seed', default=None, type=int, metavar='SEED', help='Random seed (default: None)')
    parser.add_argument('--out-iter-freq', default=10, type=int)

    parser.add_argument('--aug-lab-on', action='store_true', help='whether to use labeled augmentation')
    parser.add_argument('--duplicate-ratio', default=1, type=int, help='duplicate ratio of the labeled pool')
    parser.add_argument('--aug-ratio-lab', default=10, type=int, help='single-image augmentation ratio')
    parser.add_argument('--mix-ratio-lab', default=5, type=int, help='image mixing augmentation ratio')

    parser.add_argument('--aug-lab-training-mode', default='StrengthGuidedAugment', type=str,
                        help='how augmentations interfere with training 0. StrengthGuidedAugment;'
                             '1. RandAugment; 2. AutoAugment; 3. TrivialAugmentWide',)
    parser.add_argument('--aug-lab-strength-mode', default='all', type=str,
                        help='strength is optimized 0. default mode: none'
                             '1. globally: all; 2. per class: class; 3. per sample: sample')
    parser.add_argument('--num-strength-bins-lab', default=4, type=int,
                        help='The number of strengths to divide')
    parser.add_argument('--aug-type-lab', default=None, type=str, help='the augmentation type used, for ablation only')
    parser.add_argument('--mix-type-lab', default=None, type=str,
                        help='the mixing types used for ablation purpose only')
    parser.add_argument('--aug-strength-lab', default=None, type=int,
                        help='augmentation magnitude for labeled pool, '
                             'When it is None, meaning automatic optimization for magnitudes; '
                             'When it is a given number, it is fixed, no optimization.'
                             'For ablation purpose only.')

    parser.add_argument('--aug-ulb-on', action='store_true', help='whether to use labeled augmentation')
    parser.add_argument('--aug-ratio-ulb', default=10, type=int, help='single-image augmentation ratio')
    parser.add_argument('--mix-ratio-ulb', default=5, type=int, help='image mixing augmentation ratio')

    parser.add_argument('--aug-ulb-evaluation-mode', default='StrengthGuidedAugment', type=str,
                        help='how augmentations interfere with training 0. StrengthGuidedAugment;'
                             '1. RandAugment; 2. AutoAugment; 3. TrivialAugmentWide', )
    parser.add_argument('--aug-ulb-strength-mode', default='all', type=str,
                        help='strength is optimized 0. default mode: none'
                             '1. globally: all; 2. per class: class; 3. per sample: sample')
    parser.add_argument('--num-strength-bins-ulb', default=4, type=int,
                        help='The number of strengths to divide')
    parser.add_argument('--aug-metric-ulb', default='normal',
                        type=str, help='Only for augmentation-based metrics, including:')

    parser.add_argument('--aug-type-ulb', default=None, type=str, help='the augmentation types used, for ablation only')
    parser.add_argument('--mix-type-ulb', default=None, type=str, help='the mixing types used, for ablation only')
    parser.add_argument('--aug-strength-ulb', default=None, type=int,
                        help='augmentation magnitude for labeled pool, '
                             'When it is None, meaning automatic optimization for magnitudes; '
                             'When it is a given number, it is fixed, no optimization.'
                             'For ablation purpose only.')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()
