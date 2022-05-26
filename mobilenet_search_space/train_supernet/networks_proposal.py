import os
import sys
import argparse
import glob
import json
import shutil
import logging
import argparse
import random
import numpy as np
from config import config
sys.path.insert(0, '../../')
import nasbench201.utils as ig_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dset

from super_model import SuperNetwork
from mobilenet_search_space.train_supernet.init_projection import pt_project
import logging
from torch.utils.tensorboard import SummaryWriter
from sota.cnn.hdf5 import H5Dataset

parser = argparse.ArgumentParser("mobilenet")
# data related 
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet16-120', 'imagenet'], help='choose dataset')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for alpha')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')

#search space setting
parser.add_argument('--search_space', type=str, default='mobilenet')
parser.add_argument('--pool_size', type=int, default=10, help='number of model to proposed')

#system configurations
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--save', type=str, default='exp', help='experiment name')

#### common
parser.add_argument('--fast', action='store_true', default=False, help='skip loading api which is slow')

#### projection
parser.add_argument('--edge_decision', type=str, default='random', choices=['random','reverse', 'order', 'global_op_greedy', 'global_op_once', 'global_edge_greedy', 'global_edge_once', 'shrink_pt_project'], help='which edge to be projected next')
parser.add_argument('--proj_crit', type=str, default='jacob', choices=['loss', 'acc', 'jacob', 'snip', 'fisher', 'synflow', 'grad_norm', 'grasp', 'jacob_cov'], help='criteria for projection')
args = parser.parse_args()

expid = args.save
args.save = '../../experiments/mobilenet-prop-{}-{}-{}-{}-{}-{}'.format(args.save, args.seed, args.pool_size, args.dataset, args.edge_decision, args.proj_crit)
#### logging
scripts_to_save = glob.glob('*.py') + ['../../exp_scripts/{}.sh'.format(expid)]
if os.path.exists(args.save):
    if input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
        print('proceed to override saving directory')
        shutil.rmtree(args.save)
    else:
        exit(0)
ig_utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

log_path = os.path.join(args.save, 'log.txt')
logging.info('======> log filename: %s', 'log.txt')

if os.path.exists(log_path):
    if input("WARNING: {} exists, override?[y/n]".format(log_file)) == 'y':
        print('proceed to override log file directory')
    else:
        exit(0)

fh = logging.FileHandler(log_path, mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')

#### macros
if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'imagenet16-120':
    n_classes = 120
elif args.dataset == 'imagenet':
    n_classes = 1000
else:
    n_classes = 10

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    seed_torch(args.seed)

    # model
    model = SuperNetwork().cuda()

    #### data
    if args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                normalize,
        ])
        #for test
        #from nasbench201.DownsampledImageNet import ImageNet16
        #train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        #n_classes = 10
        train_data = H5Dataset(os.path.join(args.data, 'imagenet-train-256.h5'), transform=train_transform)
        #valid_data  = H5Dataset(os.path.join(args.data, 'imagenet-val-256.h5'),   transform=test_transform)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    else:
        if args.dataset == 'cifar10':
            train_transform, _ = ig_utils._data_transforms_cifar10(args)
            train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        elif args.dataset == 'cifar100':
            train_transform, _ = ig_utils._data_transforms_cifar100(args)
            train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        elif args.dataset == 'svhn':
            train_transform, _ = ig_utils._data_transforms_svhn(args)
            train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True
        )

    operations = [list(range(config.op_num)) for i in range(config.layers)]
    for i in range(len(operations)):
        if i not in config.stage_last_id and not i == 0:
            operations[i].append(-1)
    print('operations={}'.format(operations))

            # Uniform Sampling
    rng = []
    for i, ops in enumerate(operations):
        rng.append(ops)

    #format network pool diction
    networks_pool={}
    networks_pool['search_space'] = args.search_space
    networks_pool['dataset'] = args.dataset
    networks_pool['networks'] = []
    networks_pool['pool_size'] = args.pool_size 
    #### architecture selection / projection
    for i in range(args.pool_size):
        network_info={}
        logging.info('{} MODEL HAS SEARCHED'.format(i+1))
        searched_rng = pt_project(train_queue, model, rng, config, args)
        searched_rng = [str(x[0]) for x in searched_rng]
        searched_rng =  ' '.join(searched_rng)
        network_info['id'] = str(i)
        network_info['genotype'] = searched_rng
        networks_pool['networks'].append(network_info)

    with open(os.path.join(args.save,'networks_pool.json'), 'w') as save_file:
        json.dump(networks_pool, save_file)
if __name__ == '__main__':
    main()
