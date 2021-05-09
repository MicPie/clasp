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

from clasp import CLASP, Transformer, tokenize, basic_rand_sampler, basic_aa_tokenizer, CLASPDataset

import requests

# multi-GPU training script based on https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

# Example command to start a inferenceinference:
# python inference_multigpu_simple_query_test.py --id TEST --path-data "data/uniprot_sprot.csv" --path-offsd "data/uniprot_sprot_offset_dict.json" --save-interval-step 2 --path-weights "results/run24/model/2021-05-08_13:35:03_step00140000.pt"


# TO DO: Setup for GPU inference.


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
    parser.add_argument("--bs", type=int, default=24,
                        help="batch size (default: 24)")
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

    args = parser.parse_args()
    args.cmd = " ".join("\""+arg+"\"" if " " in arg else arg for arg in sys.argv)
    return args


def set_requires_grad(model, option=False):
    for param in model.parameters():
        param.requires_grad = option


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
    
    if args.rank == 0:
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

        text_sampler = partial(basic_rand_sampler, sample_len=1024)
        text_tok = partial(tokenize, context_length=1024, return_mask=True)

        bioseq_sampler = partial(basic_rand_sampler, sample_len=512)
        bioseq_tok = partial(basic_aa_tokenizer, context_length=512, return_mask=True)
        print(f"{datetime.now()} rank: {args.rank} samplers and tokenizer setup")

        prot_urls = ["https://www.uniprot.org/uniprot/B2RUZ4.fasta",
                     "https://www.uniprot.org/uniprot/P38649.fasta",
                     
                     "https://www.uniprot.org/uniprot/Q9VGE8.fasta",
                     "https://www.uniprot.org/uniprot/Q16650.fasta",
                     
                     "https://www.uniprot.org/uniprot/P20963.fasta",
                     "https://www.uniprot.org/uniprot/P60568.fasta",
                     
                     "https://www.uniprot.org/uniprot/P01215.fasta",
                     "https://www.uniprot.org/uniprot/O97385.fasta",
                     
                     "https://www.uniprot.org/uniprot/P04298.fasta",
                     "https://www.uniprot.org/uniprot/Q8TDQ0.fasta",
                    
                     "https://www.uniprot.org/uniprot/P01599.fasta",
                     "https://www.uniprot.org/uniprot/P01700.fasta"]

        bioseq_orig =["".join(requests.get(url).text.split("\n")[1:]) for url in prot_urls]
        print(f"{datetime.now()} rank: {args.rank} downloaded sequences")
        #print(bioseq_orig)

        bioseq_ = [bioseq_tok(bioseq_sampler(s)) for s in bioseq_orig]
        bioseq, bioseq_mask = list(zip(*bioseq_))
        bioseq      = torch.cat([s.unsqueeze(0) for s in bioseq])
        bioseq_mask = torch.cat([s.unsqueeze(0) for s in bioseq_mask])

        bioseq_out = model.bioseq_encoder(bioseq, mask=bioseq_mask)

        #text_orig = "protein that is present in the blood"
        #text_orig = "blood cells blood stream nutrition"
        #text_orig = "brain neurology nerve cells"
        #text_orig = "proteins that play a role in the blood transport"
        #text_orig = "hormone regulation and similar functions"
        text_orig = "immune system and antigen processing"
        #text_orig = "viral infection and replication"
        #text_orig = "antibody immune reaction"

        text, text_mask = text_tok(text_orig)

        text_out = model.text_encoder(text, mask=text_mask)

        text_out_norm = F.normalize(text_out, p=2, dim =-1)
        bioseq_out_norm = F.normalize(bioseq_out, p=2, dim =-1)

        sim = einsum('n d, n d -> n', text_out_norm, bioseq_out_norm)
        print(f"{datetime.now()} rank: {args.rank} similarities for text query: '{text_orig}':")
        for s in sim.tolist():
            print(f"{s:>20.6f}")


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"#gpus: {n_gpus}")
    if n_gpus < 2:
      print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
    else:
      run(inferer, n_gpus)

