import argparse
import os
import sys
import json
import time
from datetime import datetime
import logging
from functools import partial
import torch
import torch.distributed as dist
import torch.distributed.nn as distnn
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim import Adam
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from torch.nn.utils import clip_grad_norm_

from clasp import CLASP, Transformer, tokenize, basic_sampler, basic_rand_sampler, basic_aa_tokenizer, CLASPDataset

import numpy as np
import pandas as pd

# multi-GPU training script based on https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

# Example command to start a inferenceinference:
# python inference_multigpu_ezpred_test.py --path-weights "results/run24/model/2021-05-08_13:35:03_step00140000.pt"
# python inference_multigpu_ezpred_test.py --path-weights "results/testrun10big_multigpusim/model/2021-04-26_16:07:50_step00220000.pt" --tenc-dim 512 --bsenc-dim 512 --tenc-depth 6 --bsenc-depth 6


def get_args():
    """Get all parsed arguments."""
    parser = argparse.ArgumentParser(description="CLASP training loop")

    # data
    parser.add_argument("--id", type=str,
                        help="run id")
    parser.add_argument("--path-data", type=str,
                        help="path preprocessed csv file for training")
    parser.add_argument("--path-offsd", type=str,
                        help="path preprocessed offset dictionary json file for training")
    parser.add_argument("--path-results", type=str, default="results",
                        help="path to the results data, i.e., logs, model weights, etc. (default: results)")
    parser.add_argument("--path-weights", type=str, default=None,
                        help="path to weights for reloading (default: None)")
    parser.add_argument("--numw", type=int, default=0,
                        help="number of workers for pytorch dataloader (default: 0)")

    # training
    parser.add_argument("--world-size", type=int, default=2,
                        help="training world size (default: 2)")
    parser.add_argument("--bs", type=int, default=128,
                        help="batch size (default: 128)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="epochs (default: 2)")
    parser.add_argument("--dryrun", action="store_true", default=False,
                        help="Dry run for the setup runs only 4 steps in each epoch, use to test your setup (default: False)")

    # model
    # text encoder
    parser.add_argument("--tenc-ntok", type=int, default=49408,
                        help="text encoder num_tokens (default: 49408)")
    parser.add_argument("--tenc-dim", type=int, default=768,
                        help="text encoder dim (default: 768)")
    parser.add_argument("--tenc-depth", type=int, default=12,
                        help="text encoder depth (default: 12)")
    parser.add_argument("--tenc-seq-len", type=int, default=1024,
                        help="text encoder seq_len (default: 1024)")
    parser.add_argument("--tenc-rev", action="store_true", default=True,
                        help="text encoder reversibility (default: True)")

    # bioseq encoder
    parser.add_argument("--bsenc-ntok", type=int, default=23,
                        help="bioseq encoder num_tokens (default: 23)")
    parser.add_argument("--bsenc-dim", type=int, default=768,
                        help="bioseq encoder dim (default: 768)")
    parser.add_argument("--bsenc-depth", type=int, default=12,
                        help="bioseq encoder depth (default: 12)")
    parser.add_argument("--bsenc-seq-len", type=int, default=512,
                        help="bioseq encoder seq_len (default: 512)")
    parser.add_argument("--bsenc-rev", action="store_true", default=True,
                        help="bioseq encoder reversibility (default: True)")
    parser.add_argument("--bsenc-sparse-attn", action="store_true", default=False,
                        help="bioseq encoder sparse_attn (default: False)")

    # logging and saving
    parser.add_argument("--save-interval-epoch", type=int, default=1,
                        help="save interval epoch (default: 1")
    parser.add_argument("--save-interval-step", type=int, default=4_000,
                        help="save interval step (default: 4_000")

    # inference
    parser.add_argument("--inf-rank", type=int, default=0,
                        help="inference rank (default: 0")

    args = parser.parse_args()
    args.cmd = " ".join("\""+arg+"\"" if " " in arg else arg for arg in sys.argv)
    return args


def set_requires_grad(model, option=False):
    for param in model.parameters():
        param.requires_grad = option


class ECPredEnzymePredDataset(Dataset):
    def __init__(self, path, bioseq_sampler, bioseq_tok):
        super().__init__()
        self.path = path
        df = pd.read_pickle(path)
        df["ec_str"] = df.ec.apply(lambda x: str(x))
        self.df = df.drop_duplicates(subset=['sequence']) # remove duplicated sequences

        self.bioseq_sampler = bioseq_sampler
        self.bioseq_tok = bioseq_tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        bioseq = self.df.iloc[idx].sequence
        bioseq = self.bioseq_sampler(bioseq)
        bioseq, bioseq_mask = self.bioseq_tok(bioseq)

        #print(self.df.iloc[idx].ec)
        label = torch.tensor(self.df.iloc[idx].ec_str != "[]") # if enzyme true

        return bioseq, bioseq_mask, label


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(func, world_size):
    mp.spawn(func,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def inferer(rank, world_size):

    # get args
    args = get_args()
    args.rank = rank
    args.world_size = world_size
    
    if args.rank == args.inf_rank:
        print(f"{datetime.now()} rank: {args.rank} torch version: {torch.__version__}")
        # model setup
        text_enc = Transformer(
            num_tokens = args.tenc_ntok,
            dim = args.tenc_dim,
            depth = args.tenc_depth,
            seq_len = args.tenc_seq_len,
            reversible = args.tenc_rev
        )

        bioseq_enc = Transformer(
            num_tokens = args.bsenc_ntok,
            dim = args.bsenc_dim,
            depth = args.bsenc_depth,
            seq_len = args.bsenc_seq_len,
            reversible = args.bsenc_rev,
            sparse_attn = args.bsenc_sparse_attn
        )

        model = CLASP(
            text_encoder = text_enc,
            bioseq_encoder = bioseq_enc
        )

    #if args.rank == 0:
        if args.path_weights:
            if not(args.path_weights.startswith("randinit")):
                # TO DO: Check if this setup is really needed due to ddp.
                print(f"{datetime.now()} rank: {args.rank} path weights: {args.path_weights}")
                ckpt = torch.load(args.path_weights, map_location="cpu")
                new_ckpt = OrderedDict() 
                for k, v in ckpt.items():
                    name = k[7:] # remove "module."
                    new_ckpt[name] = v
                print(f"{datetime.now()} rank: {args.rank} reloaded checkpoint weights keys count {len(model.state_dict().keys())}")
                print(f"{datetime.now()} rank: {args.rank} processed checkpoint weights keys count {len(new_ckpt.keys())}")
                model.load_state_dict(new_ckpt)
                print(f"{datetime.now()} rank: {args.rank} reloaded model weights from {args.path_weights}")

        model.eval()
        set_requires_grad(model, option=False)
        print(f"{datetime.now()} rank: {args.rank} model set to eval mode and set requires grad for all parameters to false")

        text_sampler = partial(basic_sampler, sample_len=1024)
        text_tok = partial(tokenize, context_length=1024, return_mask=True)

        bioseq_sampler = partial(basic_sampler, sample_len=512)
        bioseq_tok = partial(basic_aa_tokenizer, context_length=512, return_mask=True)
        print(f"{datetime.now()} rank: {args.rank} samplers and tokenizer setup")

        # ECPred data from: https://github.com/nstrodt/UDSMProt/blob/master/git_data/ECPred_data.pkl
        path_ecpred = "/home/mmp/projects/UDSMProt/ECPred_data.pkl"
        ds = ECPredEnzymePredDataset(path_ecpred, bioseq_sampler, bioseq_tok)
        dl = DataLoader(ds, args.bs, shuffle=False)
        print(f"{datetime.now()} rank: {args.rank} created dataloader")

        print(f"{datetime.now()} rank: {args.rank} start inference")
        tp = time.time()
        model.to(args.rank)
        bioseq_outputs = []
        labels = []
        len_dl = len(dl)
        for i, b in enumerate(dl):
            #print(f"batch {i+1}/{len_dl}")
            bioseq, bioseq_mask, label = b
            bioseq = bioseq.to(args.rank)
            bioseq_mask = bioseq_mask.to(args.rank)
            bioseq_out = model.bioseq_encoder(bioseq, mask=bioseq_mask)
            bioseq_outputs.append(bioseq_out.detach().cpu().clone()) 
            labels.append(label.detach().cpu().clone())
            #if i == 10:
            #    break
        print(f"{datetime.now()} rank: {args.rank} inference carried out in {time.time() - tp:.3f} s")

        bioseq_outputs = torch.cat(bioseq_outputs)
        labels = torch.cat(labels)

        file_out_base = args.path_weights.split(".pt")[0].replace("/","_")
        np.save(f"{file_out_base}_bioseq_outputs.npy", bioseq_outputs.numpy())
        np.save(f"{file_out_base}_labels.npy", labels.numpy())

        #text, text_mask = text_tok(text_orig)
        #text = text.to(args.rank)
        #text_mask = text_mask.to(args.rank)
        #text_out = model.text_encoder(text, mask=text_mask)

        # EC classes descriptions from: https://en.wikipedia.org/wiki/Enzyme_Commission_number#Top_level_codes
        text_orig = [
                     "oxidoreductase to catalyze oxidation/reduction reactions; transfer of h and o atoms or electrons from one substance to another",
                     "transferase transfer of a functional group from one substance to another. the group may be methyl-, acyl-, amino- or phosphate group",
                     "hydrolass formation of two products from a substrate by hydrolysis",
                     "lyase non-hydrolytic addition or removal of groups from substrates. c-c, c-n, c-o or c-s bonds may be cleaved",
                     "isomerase intramolecule rearrangement, i.e. isomerization changes within a single molecule",
                     "ligase join together two molecules by synthesis of new c-o, c-s, c-n or c-c bonds with simultaneous breakdown of atp",
                     ]
        text_ = [text_tok(text_sampler(t)) for t in text_orig]
        text, text_mask = list(zip(*text_))
        text            = torch.cat([t for t in text]).to(args.rank)
        text_mask       = torch.cat([t for t in text_mask]).to(args.rank)

        text_out = model.text_encoder(text, mask=text_mask)

        np.save(f"{file_out_base}_text_out.npy", text_out.detach().cpu().numpy())

        text_out_norm = F.normalize(text_out, p=2, dim =-1).cpu()
        bioseq_out_norm = F.normalize(bioseq_outputs, p=2, dim =-1).cpu()

        sim = einsum('n d, m d -> n m', bioseq_out_norm, text_out_norm) * model.temperature.exp()

        df_ecpred = pd.read_pickle(path_ecpred)
        df_ecpred["label"] = df_ecpred.ec.apply(lambda x: str(x)[2:3])
        class_labels = df_ecpred[df_ecpred.label != ""].drop_duplicates(subset=['sequence']).label.apply(lambda x: int(x)).tolist()
        #print("len(class_labels):",len(class_labels))
        class_idx = (df_ecpred.drop_duplicates(subset=['sequence']).label != "").tolist()
        acc = (sim[class_idx].argmax(dim=-1)+1 == torch.tensor(class_labels)).float().mean().item()
        print(f"Accuracy: {acc:.6f}")
        with open(f"{args.path_weights.split('/')[1]}_ecpred.txt", "a") as f:
            f.write(f"{file_out_base},{acc}")
            f.write("\n")


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"#gpus: {n_gpus}")
    if n_gpus < 2:
      print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
    else:
      run(inferer, n_gpus)

