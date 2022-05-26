import os
import sys
import time
import glob
import copy
import numpy as np
import torch
import random
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import Network
import json

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--save', type=str, default='bench-cifar10', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--exp', type=str, default='networks_search', help='name of experiments')
args = parser.parse_args()

# args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %H:%M:%S')
save_path = '{}_{}_{}'.format(args.save, args.seed, 'log')
if not os.path.exists(save_path):
    os.mkdir(save_path)
fh = logging.FileHandler(os.path.join(save_path, '{}.log'.format(utils.get_real_arch(args.exp))))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    args.seed += 1
    logging.info("args = %s", args)

    model = Network(CIFAR_CLASSES)
    model = model.cuda()

    params = utils.count_parameters_in_MB(model)
    logging.info("param size = %fMB", params / 1e6)
    flops = utils.count_FLOPs(model)
    logging.info("FLOPs = %d", flops)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    train_transform, valid_transform = utils._data_transforms_cifar10()
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=True, num_workers=8, shuffle=True)

    arch_list = []
    pool_size = 10
    num_ops = 3
    for i in range(pool_size):
        weights = [
            [1.0/3, 1.0/3, 1.0/3],
            [1.0/3, 1.0/3, 1.0/3],
            [1.0/3, 1.0/3, 1.0/3],
            [1.0/3, 1.0/3, 1.0/3],
            [1.0/3, 1.0/3, 1.0/3],
            [1.0/3, 1.0/3, 1.0/3],
            [1.0/3, 1.0/3, 1.0/3],
            [1.0/3, 1.0/3, 1.0/3],
        ]

        weights = np.asarray(weights)
        #weights = torch.from_numpy(weights)
        best_opids = []
        validate_scorer = Jocab_Scorer(args.gpu)
        validate_scorer.setup_hooks(model, args.batch_size)
        edge_order = list(range(weights.shape[0]))
        #print(edge_order)
        random.shuffle(edge_order)
        print(edge_order)
        for activate_block in range(weights.shape[0]):
            activate_block = edge_order[activate_block]
            scores = []
            input, target = next(iter(train_queue))
            for op_id in range(num_ops):
                copy_weights = copy.deepcopy(weights)
                op_weights = np.ones_like(weights[activate_block, :])
                op_weights = op_weights * (1.0/num_ops)
                op_weights[op_id] = 0
 
                copy_weights[activate_block, :] = op_weights
                score = validate_scorer.score(model, input, target, copy_weights)
                scores.append(score)
            print(scores)
            best_opid = np.nanargmin(scores)
            best_opids.append(str(best_opid))
            weights[activate_block][best_opid] = 1
            #print(weights)
        best_opids = ''.join(best_opids)
        arch_list.append(best_opids)
        print(best_opids)
        
        query_results = True
        if query_results:
            import json
            with open('../data/nas-bench-macro_cifar10.json', 'r') as f:
                data = json.load(f)
                print(data[best_opids])
                
            

    #format network pool diction
    networks_pool={}
    networks_pool['dataset'] = 'CIFAR10'
    networks_pool['networks'] = arch_list
    networks_pool['pool_size'] = pool_size
    networks_pool['seed'] = args.seed

    import pickle as pkl
    with open(os.path.join(save_path,'networks_pool.pickle'), 'wb') as save_file:
        pkl.dump(networks_pool, save_file, protocol=pkl.HIGHEST_PROTOCOL)


class Jocab_Scorer:
    def __init__(self, gpu):
        self.gpu = gpu
        print('Jacob score init')

    def score(self, model, input, target, weights):
        #print(weights)
        batch_size = input.shape[0]
        model.K = torch.zeros(batch_size, batch_size).cuda()
        weights = torch.from_numpy(weights).cuda()
        input = input.to(torch.device('cuda', self.gpu))
        with torch.no_grad():
            model(input, weights)
        score = self.hooklogdet(model.K.cpu().numpy())

        return score

    def setup_hooks(self, model, batch_size):
        #initalize score 
        model = model.to(torch.device('cuda', self.gpu))
        model.eval()
        model.K = torch.zeros(batch_size, batch_size).cuda()
        def counting_forward_hook(module, inp, out):
            try:
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                x = (inp > 0).float()
                K = x @ x.t()
                K2 = (1.-x) @ (1.-x.t())
                model.K = model.K + K + K2
            except:
                pass

        for name, module in model.named_modules():
            if 'ReLU' in str(type(module)):
                module.register_forward_hook(counting_forward_hook)

    def hooklogdet(self, K, labels=None):
        s, ld = np.linalg.slogdet(K)
        return ld

if __name__ == '__main__':
    main()
