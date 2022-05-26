import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time
import random
import tensorflow as tf
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
sys.path.insert(0, '../../../')
from nasbench import api
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.nn.functional as F
from nasbench_analysis import eval_darts_one_shot_model_in_nasbench as naseval
from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from optimizers.darts import utils
from optimizers.darts.architect import Architect
from optimizers.darts.model_proj import Network
import copy
from optimizers.darts.genotypes import PRIMITIVES
from nasbench_analysis.utils import get_top_k, INPUT, OUTPUT, CONV1X1, NasbenchWrapper, natural_keys
import heapq
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the darts corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=9, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP_ORI', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random_ws seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training darts')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--output_weights', type=bool, default=True, help='Whether to use weights on the output nodes')
parser.add_argument('--search_space', choices=['1', '2', '3'], default='1')
parser.add_argument('--debug', action='store_true', default=False, help='run only for some batches')
parser.add_argument('--warm_start_epochs', type=int, default=0,
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--pool_size', type=int, default=10, help='pool_size')
args = parser.parse_args()

args.save = 'experiments/zc_prop/search_space_{}/search-{}-{}-{}-{}-{}'.format(args.search_space, args.save,
                                                                          time.strftime("%Y%m%d-%H%M%S"), args.seed,
                                                                          args.search_space, args.pool_size)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# Dump the config of the run
with open(os.path.join(args.save, 'config.json'), 'w') as fp:
    json.dump(args.__dict__, fp)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():

    #nasbench = NasbenchWrapper(dataset_file='../../data/nasbench_full.tfrecord')
    nasbench=None
    # Select the search space to search in
    if args.search_space == '1':
        search_space = SearchSpace1()
    elif args.search_space == '2':
        search_space = SearchSpace2()
    elif args.search_space == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, output_weights=args.output_weights,
                    steps=search_space.num_intermediate_nodes, search_space=search_space)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    mixed_op_alpha = model.arch_parameters()[0]
    output_alpha = model.arch_parameters()[1]
    input_alpha = model.arch_parameters()[2:]



    edges, ops = mixed_op_alpha.size()
    output_alpha_size = list(output_alpha.size())[1]
    #print(output_alpha_size, output_alpha.size())
    num_input_alpha = len(input_alpha)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.batch_size*args.pool_size*(edges+num_input_alpha+1)))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    ori_arch_params = copy.deepcopy(model.arch_parameters())

    arch_list = []
    for i in range(args.pool_size):

        score_output = Variable(torch.zeros(1, edges + 1).cuda(), requires_grad=False)

        # Initialize the weights for the inputs to each choice block.
        if type(model.search_space) == SearchSpace1:
            begin = 3
        else:
            begin = 2
        score_inputs = [Variable(torch.zeros(1, n_inputs).cuda(), requires_grad=False) for n_inputs in range(begin, edges + 1)]

        validate_scorer = Jocab_Scorer(args.gpu)
        validate_scorer.setup_hooks(model, args.batch_size)

        node_list = []
        op_weights = F.softmax(model._arch_parameters[0], dim=-1)
        output_weights = F.softmax(model._arch_parameters[1], dim=-1)
        input_weights = [F.softmax(params, dim=-1) for params in model._arch_parameters[2:]]
        print(input_weights)
        #pt ops
        edge_order = list(range(edges))
        random.shuffle(edge_order)
        logging.info(edge_order)

        for i in range(len(edge_order)):
            #get data 
            input, target = next(iter(train_queue))

            scores = []
            e = edge_order[i]
            for o in range(ops):
                temp_value = float(op_weights[e, o])
                op_weights[e, o] = 0
                score = validate_scorer.score(model, input, target, op_weights=op_weights)
                op_weights[e, o] = temp_value
                scores.append(score) 
       
            best_op = np.nanargmin(scores)
            logging.info(scores)
            node_list.append(PRIMITIVES[best_op])
            for o in range(ops):
                op_weights[e, o] = 0
            op_weights[e, best_op] = 1

        save_node_list = copy.deepcopy(node_list)
        if args.search_space == '1':
            search_space = SearchSpace1()
            num_inputs = list(search_space.num_parents_per_node.values())[3:-1]
            parents = {
                '0': [],
                '1': [0],
                '2': [0, 1],
                '3': [0, 1, 2],
                '4': [0, 1, 2, 3],
                '5': [0, 1, 2, 3, 4]
            }

            inputs_candidate = ['3', '4']
            outputs_candidate = ['5']
            node_list = [INPUT, *node_list, CONV1X1, OUTPUT]

        elif args.search_space  == '2':
            search_space = SearchSpace2()
            num_inputs = list(search_space.num_parents_per_node.values())[2:]
            parents = {
                '0': [],
                '1': [0],
                '2': [0, 1],
                '3': [0, 1, 2],
                '4': [0, 1, 2, 3],
                '5': [0, 1, 2, 3, 4]
            }

            inputs_candidate = ['2', '3', '4']
            outputs_candidate = ['5']
            node_list = [INPUT, *node_list, CONV1X1, OUTPUT]

        elif args.search_space  == '3':
            num_inputs = list(search_space.num_parents_per_node.values())[2:]
            parents = {
                '0': [],
                '1': [0],
                '2': [0, 1],
                '3': [0, 1, 2],
                '4': [0, 1, 2, 3],
                '5': [0, 1, 2, 3, 4],
                '6': [0, 1, 2, 3, 4, 5]
            }
            inputs_candidate = ['2', '3', '4', '5']
            outputs_candidate = ['6']
            node_list = [INPUT, *node_list, OUTPUT]

        random.shuffle(inputs_candidate)
        logging.info(inputs_candidate)
        for input_node_id in range(len(inputs_candidate)):
            input_node = inputs_candidate[input_node_id]
            input, target = next(iter(train_queue))
            scores = []
            for i in range(len(parents[input_node])):
                copy_parents = copy.deepcopy(parents)
                copy_parents[input_node].pop(i)
                adjacency_matrix = search_space.create_nasbench_adjacency_matrix(copy_parents)
                arch = (adjacency_matrix, save_node_list)
                print(copy_parents)
                arch_parameters = get_weights_from_arch(arch, model)
                #print(arch_parameters)
                model._arch_parameters = arch_parameters
                score = validate_scorer.score(model, input, target)
                scores.append(score*-1)
            print(scores)
            parents[input_node] = heapq.nlargest(num_inputs[input_node_id], range(len(scores)), scores.__getitem__)
        
        output_node = outputs_candidate[0]
        input, target = next(iter(train_queue))
        output_scores = []
        for i in range(len(parents[output_node])):
            copy_parents = copy.deepcopy(parents)
            copy_parents[output_node].pop(i)
            adjacency_matrix = search_space.create_nasbench_adjacency_matrix(copy_parents)
            arch = (adjacency_matrix, save_node_list)
            print(copy_parents)
            arch_parameters = get_weights_from_arch(arch, model)
            model._arch_parameters = arch_parameters
            score = validate_scorer.score(model, input, target)
            output_scores.append(score*-1)
        print(output_scores)
        parents[output_node] = heapq.nlargest(num_inputs[-1], range(len(output_scores)), output_scores.__getitem__)

            
        print(parents, save_node_list)
        adjacency_matrix = search_space.create_nasbench_adjacency_matrix(parents)
        arch = (adjacency_matrix, save_node_list)
        arch_list.append(arch)

    #format network pool diction
    networks_pool={}
    networks_pool['search_space'] = args.search_space
    networks_pool['dataset'] = 'CIFAR10'
    networks_pool['networks'] = arch_list
    networks_pool['pool_size'] = args.pool_size
    networks_pool['seed'] = args.seed

    import pickle as pkl
    with open(os.path.join(args.save,'networks_pool.pickle'), 'wb') as save_file:
        pkl.dump(networks_pool, save_file, protocol=pkl.HIGHEST_PROTOCOL)

def get_weights_from_arch(arch, model):
    adjacency_matrix, node_list = arch
    #print(adjacency_matrix)
    if args.search_space == '1' or args.search_space == '2':
        adjacency_matrix = np.delete(adjacency_matrix, 5, 0)
        #print(adjacency_matrix)
        adjacency_matrix = np.delete(adjacency_matrix, 5, 1)
    #print(adjacency_matrix)
    num_ops = len(PRIMITIVES)

    # Assign the sampled ops to the mixed op weights.
    # These are not optimized
    alphas_mixed_op = Variable(torch.zeros(model._steps, num_ops).cuda(), requires_grad=False)
    for idx, op in enumerate(node_list):
        alphas_mixed_op[idx][PRIMITIVES.index(op)] = 1

    # Set the output weights
    alphas_output = Variable(torch.zeros(1, model._steps + 1).cuda(), requires_grad=False)
    for idx, label in enumerate(list(adjacency_matrix[:, -1][:-1])):
        alphas_output[0][idx] = label

    # Initialize the weights for the inputs to each choice block.
    if type(model.search_space) == SearchSpace1:
        begin = 3
    else:
        begin = 2
    alphas_inputs = [Variable(torch.zeros(1, n_inputs).cuda(), requires_grad=False) for n_inputs in
                     range(begin, model._steps + 1)]
    for alpha_input in alphas_inputs:
        connectivity_pattern = list(adjacency_matrix[:alpha_input.shape[1], alpha_input.shape[1]])
        for idx, label in enumerate(connectivity_pattern):
            alpha_input[0][idx] = label

    # Total architecture parameters
    arch_parameters = [
        alphas_mixed_op,
        alphas_output,
        *alphas_inputs
    ]
    return arch_parameters

class Jocab_Scorer:
    def __init__(self, gpu):
        self.gpu = gpu
        print('Jacob score init')

    def score(self, model, input, target, op_weights=None, input_weights=None, output_weights=None):
        batch_size = input.shape[0]
        model.K = torch.zeros(batch_size, batch_size).cuda()

        input = input.cuda()
        with torch.no_grad():
            #print(input_weights, output_weights)
            model(input, proj_op_weights=op_weights, proj_output_weights=output_weights, proj_input_weights=input_weights)
        score = self.hooklogdet(model.K.cpu().numpy())

        #print(score)
        return score

    def setup_hooks(self, model, batch_size):
        #initalize score 
        model = model.to(torch.device('cuda', self.gpu))
        model.eval()
        model.K = torch.zeros(batch_size, batch_size).cuda()
        def counting_forward_hook(module, inp, out):
            try:
                # if not module.visited_backwards:
                #     return
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
                has_hook = True
                for k, v in module._forward_hooks.items():
                    #print(v.__name__)
                    if v.__name__ == 'counting_forward_hook':
                        #print('avoid rehook')
                        has_hook = False
                if has_hook:
                    module.register_forward_hook(counting_forward_hook)
                #module.register_backward_hook(counting_backward_hook)

    def hooklogdet(self, K, labels=None):
        s, ld = np.linalg.slogdet(K)
        return ld


if __name__ == '__main__':
    main()