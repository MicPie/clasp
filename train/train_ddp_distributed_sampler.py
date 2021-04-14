import os
import sys
import time
from functools import partial
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from clasp import CLASP, Transformer, tokenize, basic_rand_sampler, basic_aa_tokenizer, CLASPDataset


# multi-GPU training script based on https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

path_data = "~/projects/clasp/data/uniprot_100_reduced.csv"
#path_data = "~/hdd1/ProTexCLIP/uniprot_sprot_reduced.csv"
epochs = 2
bs = 4
world_size = 2


# data setup

text_sampler = partial(basic_rand_sampler, sample_len=1024)
text_tok     = partial(tokenize, context_length=1024, return_mask=True)

bioseq_sampler = partial(basic_rand_sampler, sample_len=512)
bioseq_tok     = partial(basic_aa_tokenizer, context_length=512, return_mask=True)


# model setup

text_enc = Transformer(
    num_tokens = 49408,
    dim = 512,
    depth = 6,
    seq_len = 1024,
    reversible = True
)

bioseq_enc = Transformer(
    num_tokens = 23,
    dim = 512,
    depth = 6,
    seq_len = 512,
    reversible = True
    #    sparse_attn = True
)

clasp = CLASP(
    text_encoder = text_enc,
    bioseq_encoder = bioseq_enc
)


# optimizer

opt = Adam(clasp.parameters(), lr = 3e-4)


# training

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# From https://github.com/rwightman/pytorch-image-models/blob/779107b693010934ac87c8cecbeb65796e218488/timm/utils/distributed.py#L11
def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    #print(f"rt/n: {rt}/{n}")
    rt /= n
    return rt


def train_ddp(rank, world_size, model=clasp, optimizer=opt, epochs=epochs):

    print(f"rank: {rank:<5}world_size: {world_size}.")

    setup(rank, world_size)

    # set this up so each rank gets only a subset of the file, so the entire file is only loaded one time!
    ds_train = CLASPDataset(path=path_data,
                      text_sampler=text_sampler,
                      bioseq_sampler=bioseq_sampler,
                      text_tok=text_tok,
                      bioseq_tok=bioseq_tok)

    sampler = DistributedSampler(ds_train,
                                 num_replicas=world_size,
                                 rank=rank, # By default, rank is retrieved from the current distributed group.
                                 shuffle=True, # May be True
                                 seed=42)

    dl_train = DataLoader(ds_train,
                          batch_size=bs,
                          shuffle=False,  # Must be False!
                          num_workers=0,
                          sampler=sampler,
                          #sampler=None,
                          pin_memory=True)

    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    #time.sleep(30)

    def one_epoch(model, optimizer, dl, epoch, rank, train=True):
        time_epoch_start = time.time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        if train:
            model.train()
        else:
            model.eval()

        tp = time.time()
        print(f"rank: {rank:<5}len dl: {len(dl)}")
        for i, b in enumerate(dl):
            optimizer.zero_grad()

            dt = time.time() - tp
            dt = torch.tensor(dt).to(rank)
            dt = reduce_tensor(dt, world_size)
            data_time.update(dt)

            text, text_mask, bioseq, bioseq_mask = b

            text        = text.to(rank).squeeze(1)
            text_mask   = text_mask.to(rank).squeeze(1)
            bioseq      = bioseq.to(rank)
            bioseq_mask = bioseq_mask.to(rank)

            loss = clasp(
                text,
                bioseq,
                text_mask = text_mask,
                bioseq_mask = bioseq_mask,
                return_loss = True # set return loss to True
            )
            #print("loss: ",loss)
            reduced_loss = reduce_tensor(loss.data, world_size)
            #print("reduced_loss: ",reduced_loss)

            #losses.update(loss.item())
            losses.update(reduced_loss.item())

            if train:
                loss.backward()
                optimizer.step()

            bt = time.time() - tp
            bt = torch.tensor(bt).to(rank)
            bt = reduce_tensor(bt, world_size)
            batch_time.update(bt)

            tp = time.time()

        time_epoch_end = time.time()
        et = time_epoch_end - time_epoch_start
        et = torch.tensor(et).to(rank)
        epoch_time = reduce_tensor(et, world_size)

        if rank == 0:
            print(f"rank: {rank:<5}epoch: {epoch:<5}et: {epoch_time:<10.3f}bt: {batch_time.avg:<10.3f}dt: {data_time.avg:<10.3f}{'train' if train else 'valid'} loss: {losses.avg:<10.3f}")

        return model, optimizer

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        ddp_model, optimizer = one_epoch(ddp_model, optimizer, dl_train, epoch, rank, train=True)

    cleanup()


def run(func, world_size):
    mp.spawn(func,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"#gpus: {n_gpus}")
    if n_gpus < 2:
      print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
    else:
      run(train_ddp, world_size)

