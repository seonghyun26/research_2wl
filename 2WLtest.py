from math import e
from scipy.sparse import data
from sklearn import utils
import random
import numpy as np
from model import LocalWLNet, WLNet, FWLNet, LocalFWLNet, Net_cora
from datasets import load_dataset, dataset
from impl import train
import torch
from torch.optim import Adam
from ogb.linkproppred import Evaluator
import yaml

import warnings
warnings.filterwarnings("ignore")

import wandb
from datetime import datetime

TESTNUM = 9

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate_hits(pos_pred, neg_pred, K):
    results = {}
    evaluator = Evaluator(name='ogbl-collab')
    evaluator.K = K
    hits = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })[f'hits@{K}']

    results[f'Hits@{K}'] = hits

    return results

def testparam(device="cpu", dsname="Celegans", subgraph="path"):  # mod_params=(32, 2, 1, 0.0), lr=3e-4
    device = torch.device(device)
    bg = load_dataset(dsname, args.pattern)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    max_degree = torch.max(bg.x[2])

    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))
    if trn_ds.na != None:
        print("use node feature")
        trn_ds.na = trn_ds.na.to(device)
        val_ds.na = val_ds.na.to(device)
        tst_ds.na = tst_ds.na.to(device)
        use_node_attr = True
    else:
        use_node_attr = False


    def valparam(**kwargs):
        lr = kwargs.pop('lr')
        epoch = kwargs.pop('epoch')
        if args.pattern == '2wl':
            mod = WLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2wl_l':
            mod = LocalWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2fwl':
            mod = FWLNet(max_degree, use_node_attr, trn_ds.na, subgraph=subgraph, **kwargs).to(device)
        elif args.pattern == '2fwl_l':
            mod = LocalFWLNet(max_degree, use_node_attr, trn_ds.na, subgraph=subgraph, **kwargs).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train.train_routine(args.dataset, mod, opt, trn_ds, val_ds, tst_ds, epoch, verbose=True)

    with open(f"config/{args.pattern}/{args.dataset}.yaml") as f:
        params = yaml.safe_load(f)

    valparam(**(params))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pattern', type=str, default="2fwl_l")
    parser.add_argument('--dataset', type=str, default="USAir")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--path', type=str, default="opt/")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--check', action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--subgraph', type=str, default="path")
    # path, cycle, incoming, outgoing
    
    args = parser.parse_args()
    if args.device < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)
    print(args.device)
    started = datetime.now()
    for i in range(10):
        run = wandb.init(
            project = "2wl",
            entity = "eddy26",
            resume = False,
            name = args.pattern + "_" + args.dataset + "_test" + str(TESTNUM) + "_seed"+str(i),
            job_type = "eval",
            config = {
                # NOTE: HERE
                # "subgraph": args.subgraph,
                "pattern": args.pattern,
                "seed": args.seed,
                "i": i,
                "started": started,
                "dataset": args.dataset
            }
        )
        set_seed(i + args.seed)
        print(f"<<--- {i}th seed {i+args.seed} Test Start --->>")
        testparam(args.device, args.dataset, args.subgraph)
        print(f"<<--- {i}th seed {i+args.seed} Test Done  --->>\n")
        run.finish()
        