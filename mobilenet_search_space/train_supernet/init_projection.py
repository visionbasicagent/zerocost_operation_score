import os
import sys
import numpy as np
import torch
sys.path.insert(0, '../../')
import logging
import copy
from collections import OrderedDict
from foresight.pruners import *

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

# fixed [reverse, order]: discretizes the edges in a fixed order, where in our experiments we discretize from the222input towards the output of the cell struct
# random: discretizes the edges in a random order (DARTS-PT)
# NOTE: Only this methods allows use other zero-cost proxy metrics 
def pt_project(proj_queue, model, init_rngs, config, args):
    rngs = copy.deepcopy(init_rngs)
    candidate_flags = np.ones((config.layers,), dtype=int)
    def project(model, args):
        ## macros

        ## select an edge
        remain_eids = np.nonzero(candidate_flags)[0]
        #print(remain_eids)
        if args.edge_decision == "random":
            selected_eid = np.random.choice(remain_eids, size=1)[0]
        elif args.edge_decision == "reverse":
            selected_eid = remain_eids[-1]
        else:
            selected_eid = remain_eids[0]

        ## select the best operation
        compare = lambda x, y: x < y

        if args.dataset == 'cifar100':
            n_classes = 100
        elif args.dataset == 'imagenet16-120':
            n_classes = 120
        elif args.dataset == 'imagenet':
            n_classes = 1000
        else:
            n_classes = 10

        best_opid = 0
        crit_list = []
        op_ids = []
        num_op = len(rngs[selected_eid])
        input, target = next(iter(proj_queue))
        for opid in range(num_op):
            pt_rngs = copy.deepcopy(rngs)
            pt_rngs[selected_eid].pop(opid)
            ## proj evaluation
            if args.proj_crit == 'jacob':
                crit = Jocab_Score(model, input,  target, pt_rngs)
            else:
                model.rngs = pt_rngs
                measures = predictive.find_measures(model,
                                    proj_queue,
                                    ('random', 1, n_classes), 
                                    torch.device("cuda"),
                                    measure_names=[args.proj_crit])
                model.rngs = None
                crit = measures[args.proj_crit]

            crit_list.append(crit)
            op_ids.append(opid)
            
        best_opid = op_ids[np.nanargmin(crit_list)]

        logging.info('best opid %d', best_opid)
        logging.info('current edge id %d', selected_eid)
        logging.info(crit_list)
        return selected_eid, best_opid
        
    num_edges = len(rngs)
    for epoch in range(num_edges):
        logging.info('epoch %d', epoch)        
        logging.info('project')
        selected_eid, best_opid = project(model, args)
        rngs[selected_eid] = [rngs[selected_eid][best_opid]]
        candidate_flags[selected_eid] = 0
    print(rngs)
    return rngs

def Jocab_Score(ori_model, input, target, pt_rngs):
    model = copy.deepcopy(ori_model)
    model.eval()

    batch_size = input.shape[0]
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
    
    input = input.cuda()

    with torch.no_grad():
        model(input, pt_rngs)
    score = hooklogdet(model.K.cpu().numpy())

    del model
    return score

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld