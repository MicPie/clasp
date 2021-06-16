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

from clasp import CLASP, Transformer, tokenize, basic_rand_sampler, basic_aa_tokenizer, CLASPRankSplitDataset


# multi-GPU training script based on https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

# Example command to start a training run:
# python train_multigpusim.py --id TEST --path-data-train "data/uniprot_full_valid-ood.csv" --path-offsd-train "data/uniprot_full_valid-ood_offsetdict.json" --path-data-valid-id "data/uniprot_full_valid-ood.csv" --path-offsd-valid-id "data/uniprot_full_valid-ood_offsetdict.json" --path-data-valid-ood "data/uniprot_full_valid-ood.csv" --path-offsd-valid-ood "data/uniprot_full_valid-ood_offsetdict.json" --save-interval-step 2 --dryrun


def get_args():
    """Get all parsed arguments."""
    parser = argparse.ArgumentParser(description="CLASP training loop")

    # data
    parser.add_argument("--id", type=str,
                        help="run id")

    parser.add_argument("--path-data-train", type=str,
                        help="path preprocessed csv file for training")
    parser.add_argument("--path-offsd-train", type=str,
                        help="path preprocessed offset dictionary json file for training")

    parser.add_argument("--path-data-valid-id", type=str,
                        help="path preprocessed csv file for valid id")
    parser.add_argument("--path-offsd-valid-id", type=str,
                        help="path preprocessed offset dictionary json file for valid id")

    parser.add_argument("--path-data-valid-ood", type=str,
                        help="path preprocessed csv file for valid ood")
    parser.add_argument("--path-offsd-valid-ood", type=str,
                        help="path preprocessed offset dictionary json file for valid ood")

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
    parser.add_argument("--tenc-dim", type=int, default=512,
                        help="text encoder dim (default: 512)")
    parser.add_argument("--tenc-depth", type=int, default=6,
                        help="text encoder depth (default: 6)")
    parser.add_argument("--tenc-seq-len", type=int, default=1024,
                        help="text encoder seq_len (default: 1024)")
    parser.add_argument("--tenc-rev", action="store_true", default=True,
                        help="text encoder reversibility (default: True)")

    # bioseq encoder
    parser.add_argument("--bsenc-ntok", type=int, default=23,
                        help="bioseq encoder num_tokens (default: 23)")
    parser.add_argument("--bsenc-dim", type=int, default=512,
                        help="bioseq encoder dim (default: 512)")
    parser.add_argument("--bsenc-depth", type=int, default=6,
                        help="bioseq encoder depth (default: 6)")
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


def create_logger(path_log, file_name):
    file_path = os.path.join(path_log, file_name)

    # file handler for logging to file
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)

    # console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if file_path is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


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


# From https://github.com/rwightman/pytorch-image-models/blob/779107b693010934ac87c8cecbeb65796e218488/timm/utils/distributed.py#L11
def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    #print(f"rt/n: {rt}/{n}")
    rt /= n
    return rt


#def log_info_rank0(logger, rank, output):
#    if rank == 0:
#        logger.info(f"{datetime.now()} {output}")


def train_ddp(args, model, optimizer, dl_train, dl_valid_id, dl_valid_ood, epochs, logger=None, writer=None):

    # Based on: https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    torch.cuda.set_device(args.rank)
    torch.cuda.empty_cache()

    step = 0

    logger.info(f"{datetime.now()} rank: {args.rank} world_size: {args.world_size}")
    setup(args.rank, args.world_size)
    logger.info(f"{datetime.now()} rank: {args.rank} ddp setup")
    model.to(args.rank)
    logger.info(f"{datetime.now()} rank: {args.rank} model moved to rank {args.rank}")
    ddp_model = DDP(model, device_ids=[args.rank])
    logger.info(f"{datetime.now()} rank: {args.rank} created ddp model")

    def validate(model, dl, step, logid="id"):
        losses     = AverageMeter()
        accuracies = AverageMeter()

        model.eval()

        with torch.no_grad():
            for j, b in enumerate(dl):
                text, text_mask, bioseq, bioseq_mask = b

                text        = text.to(args.rank).squeeze(1)
                text_mask   = text_mask.to(args.rank).squeeze(1)
                bioseq      = bioseq.to(args.rank)
                bioseq_mask = bioseq_mask.to(args.rank)

                text_latents, bioseq_latents, temp = model(
                    text,
                    bioseq,
                    text_mask = text_mask,
                    bioseq_mask = bioseq_mask,
                    return_loss = False,
                    return_latents_temp = True
                )

                all_text_latents   = [torch.zeros_like(text_latents)   for _ in range(dist.get_world_size())]
                all_bioseq_latents = [torch.zeros_like(bioseq_latents) for _ in range(dist.get_world_size())]
                dist.all_gather(all_text_latents, text_latents)
                dist.all_gather(all_bioseq_latents, bioseq_latents)

                all_text_latents   = torch.cat(all_text_latents, dim=0)
                all_bioseq_latents = torch.cat(all_bioseq_latents, dim=0)

                sim_text   = (einsum('i d, j d -> i j', text_latents, all_bioseq_latents) * temp)
                sim_bioseq = (einsum('i d, j d -> i j', bioseq_latents, all_text_latents) * temp)

                labels = torch.arange(args.rank*args.bs, (args.rank+1)*args.bs).to(args.rank)

                loss = ((F.cross_entropy(sim_text, labels) + F.cross_entropy(sim_bioseq, labels)) / 2).mean()

                acc_text   = ((sim_text.argmax(0) == labels.argmax(0)).float()).mean()
                acc_bioseq = ((sim_bioseq.argmax(0) == labels.argmax(0)).float()).mean()
                acc        = (acc_text + acc_bioseq).mean()

                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses.update(reduced_loss.item())

                reduced_acc = reduce_tensor(acc.data, args.world_size)
                accuracies.update(reduced_acc.item())

                if args.rank == 0:
                    writer.add_scalars("1 loss/1 step", {f"valid {logid}": reduced_loss.item()}, step)
                    writer.add_scalars("2 accuracy/1 step", {f"valid {logid}": reduced_acc.item()}, step)

    def one_epoch(args, model, optimizer, dl_train, dl_valid_id, dl_valid_ood, epoch, step):
        time_epoch_start = time.time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        model.train()

        tp = time.time()
        for i, b in enumerate(dl_train):

            if args.dryrun:
                if i == 4:
                    break

            optimizer.zero_grad()

            dt = time.time() - tp
            dt = torch.tensor(dt).to(args.rank)
            dt = reduce_tensor(dt, args.world_size)
            data_time.update(dt)

            text, text_mask, bioseq, bioseq_mask = b

            text        = text.to(args.rank).squeeze(1)
            text_mask   = text_mask.to(args.rank).squeeze(1)
            bioseq      = bioseq.to(args.rank)
            bioseq_mask = bioseq_mask.to(args.rank)

            text_latents, bioseq_latents, temp = model(
                text,
                bioseq,
                text_mask = text_mask,
                bioseq_mask = bioseq_mask,
                return_loss = False,
                return_latents_temp = True
            )

            all_text_latents   = [torch.zeros_like(text_latents)   for _ in range(dist.get_world_size())]
            all_bioseq_latents = [torch.zeros_like(bioseq_latents) for _ in range(dist.get_world_size())]
            dist.all_gather(all_text_latents, text_latents)
            dist.all_gather(all_bioseq_latents, bioseq_latents)

            all_text_latents   = torch.cat(all_text_latents, dim=0)
            all_bioseq_latents = torch.cat(all_bioseq_latents, dim=0)

            sim_text   = (einsum('i d, j d -> i j', text_latents, all_bioseq_latents) * temp)
            sim_bioseq = (einsum('i d, j d -> i j', bioseq_latents, all_text_latents) * temp)

            model.temperature.data.clamp_(-torch.log(torch.tensor(100.)), torch.log(torch.tensor(100.)))

            labels = torch.arange(args.rank*args.bs, (args.rank+1)*args.bs).to(args.rank)

            loss = ((F.cross_entropy(sim_text, labels) + F.cross_entropy(sim_bioseq, labels)) / 2).mean()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            acc_text   = ((sim_text.argmax(0) == labels.argmax(0)).float()).mean()
            acc_bioseq = ((sim_bioseq.argmax(0) == labels.argmax(0)).float()).mean()
            acc        = (acc_text + acc_bioseq).mean()

            reduced_loss = reduce_tensor(loss.data, args.world_size)
            losses.update(reduced_loss.item())

            reduced_acc = reduce_tensor(acc.data, args.world_size)
            accuracies.update(reduced_acc.item())

            if args.rank == 0:
                writer.add_scalars("1 loss/1 step", {"train": reduced_loss.item()}, step)
                writer.add_scalars("2 accuracy/1 step", {"train": reduced_acc.item()}, step)

            if (step % args.save_interval_step == 0) and (step != 0):
                if args.rank == 0:
                    path_save = os.path.join(args.path_model, f"{'_'.join(str(datetime.now()).split('.')[0].split(' '))}_step{step:08d}.pt")
                    torch.save(ddp_model.state_dict(), path_save)

                validate(model, dl_valid_id, step, logid="id")
                validate(model, dl_valid_ood, step, logid="ood")

            bt = time.time() - tp
            bt = torch.tensor(bt).to(args.rank)
            bt = reduce_tensor(bt, args.world_size)
            batch_time.update(bt)
            if args.rank == 0:
                writer.add_scalars("3 timings/1 step", {"dt": dt, "bt": bt}, step)
                if (step % args.save_interval_step == 0) and (step != 0):
                    logger.info(f"{datetime.now()} epoch: {epoch:>4} step: {step:>8} bt: {batch_time.avg:<10.3f}dt: {data_time.avg:<10.3f}{'train' if train else 'valid'} loss: {losses.avg:<10.3f} acc: {accuracies.avg:<10.3f}")
                step += 1

            tp = time.time()

        if args.rank == 0:
            if epoch % args.save_interval_epoch == 0:
                path_save = os.path.join(args.path_model, f"{'_'.join(str(datetime.now()).split('.')[0].split(' '))}_epoch{epoch:03d}.pt")
                torch.save(ddp_model.state_dict(), path_save)

        time_epoch_end = time.time()
        et = time_epoch_end - time_epoch_start
        et = torch.tensor(et).to(args.rank)
        epoch_time = reduce_tensor(et, args.world_size)

        if args.rank == 0:
            logger.info(f"{datetime.now()} epoch: {epoch:>4} et: {epoch_time:<11.3f}bt: {batch_time.avg:<10.3f}dt: {data_time.avg:<10.3f}{'train' if train else 'valid'} loss: {losses.avg:<10.3f} acc: {accuracies.avg:<10.3f}")
            writer.add_scalars("1 loss/2 epoch", {"train": losses.avg}, epoch)
            writer.add_scalars("2 accuracy/2 epoch", {"train": accuracies.avg}, epoch)
            writer.add_scalars("3 timings/2 step", {"dt": data_time.avg, "bt": batch_time.avg}, epoch)
            writer.add_scalars("3 timings/3 epoch", {"et": epoch_time}, epoch)

        return model, optimizer, step

    logger.info(f"{datetime.now()} rank: {args.rank} start training")
    for epoch in range(args.epochs):
        ddp_model, optimizer, step = one_epoch(args, ddp_model, optimizer, dl_train, dl_valid_id, dl_valid_ood, epoch, step=step, train=True)

    cleanup()
    #logger.info(f"{datetime.now()} rank: {args.rank} ddp cleanup")


def run(func, world_size):
    mp.spawn(func,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def trainer(rank, world_size):

    # get args
    args = get_args()
    args.rank = rank
    args.world_size = world_size
    
    # setup paths
    args.path_log = os.path.join(args.path_results, args.id)
    args.path_tb = os.path.join(args.path_log,"tb")
    args.path_model = os.path.join(args.path_log,"model")
    os.makedirs(args.path_log, exist_ok=True)
    os.makedirs(args.path_tb, exist_ok=True)
    os.makedirs(args.path_model, exist_ok=True)

    # setup loggers
    # TO DO: Revisit file name when process are more than 1min apart.
    fn_log = f"clasp_train_{'_'.join(str(datetime.now()).split('.')[0].split(' '))}.log"
    logger = create_logger(args.path_log, file_name=fn_log)
    logger.info(f"{datetime.now()} rank: {args.rank} start logging")
    if args.rank == 0:
        writer = SummaryWriter(log_dir=args.path_tb, flush_secs=2)

    if args.rank == 0:
        # show args
        for k in args.__dict__.keys():
            logger.info(f"{k:>20}: {args.__dict__[k]}")

    # data setup
    text_sampler = partial(basic_rand_sampler, sample_len=1024)
    text_tok     = partial(tokenize, context_length=1024, return_mask=True)

    bioseq_sampler = partial(basic_rand_sampler, sample_len=512)
    bioseq_tok     = partial(basic_aa_tokenizer, context_length=512, return_mask=True)
    logger.info(f"{datetime.now()} rank: {args.rank} created samplers and tokenizers")

    logger.info(f"{datetime.now()} rank: {args.rank} data setup")
    with open(args.path_offsd_train, "r", encoding='utf-8') as offsd:
        offset_dict_train = json.load(offsd)
    logger.info(f"{datetime.now()} rank: {args.rank} loaded train offset dict")

    with open(args.path_offsd_valid_id, "r", encoding='utf-8') as offsd:
        offset_dict_valid_id = json.load(offsd)
    logger.info(f"{datetime.now()} rank: {args.rank} loaded valid id offset dict")

    with open(args.path_offsd_valid_ood, "r", encoding='utf-8') as offsd:
        offset_dict_valid_ood = json.load(offsd)
    logger.info(f"{datetime.now()} rank: {args.rank} loaded valid ood offset dict")

    ds_train = CLASPRankSplitDataset(file_path=args.path_data_train,
                           offset_dict=offset_dict_train,
                           rank=args.rank,
                           world_size=args.world_size,
                           logger=logger,
                           text_sampler=text_sampler,
                           bioseq_sampler=bioseq_sampler,
                           text_tok=text_tok,
                           bioseq_tok=bioseq_tok)
    logger.info(f"{datetime.now()} rank: {args.rank} created train dataset")

    dl_train = DataLoader(ds_train,
                          batch_size=args.bs,
                          shuffle=True if not(args.dryrun) else False,
                          num_workers=args.numw,
                          pin_memory=True)
    logger.info(f"{datetime.now()} rank: {args.rank} created train dataloader with length {len(dl_train)}")

    ds_valid_id = CLASPRankSplitDataset(file_path=args.path_data_valid_id,
                           offset_dict=offset_dict_valid_id,
                           rank=args.rank,
                           world_size=args.world_size,
                           logger=logger,
                           text_sampler=text_sampler,
                           bioseq_sampler=bioseq_sampler,
                           text_tok=text_tok,
                           bioseq_tok=bioseq_tok)
    logger.info(f"{datetime.now()} rank: {args.rank} created valid id dataset")

    dl_valid_id = DataLoader(ds_valid_id,
                          batch_size=args.bs,
                          shuffle=False,
                          num_workers=args.numw,
                          pin_memory=True)
    logger.info(f"{datetime.now()} rank: {args.rank} created valid id dataloader with length {len(dl_valid_id)}")

    ds_valid_ood = CLASPRankSplitDataset(file_path=args.path_data_valid_ood,
                           offset_dict=offset_dict_valid_ood,
                           rank=args.rank,
                           world_size=args.world_size,
                           logger=logger,
                           text_sampler=text_sampler,
                           bioseq_sampler=bioseq_sampler,
                           text_tok=text_tok,
                           bioseq_tok=bioseq_tok)
    logger.info(f"{datetime.now()} rank: {args.rank} created valid ood dataset")

    dl_valid_ood = DataLoader(ds_valid_ood,
                          batch_size=args.bs,
                          shuffle=False,
                          num_workers=args.numw,
                          pin_memory=True)
    logger.info(f"{datetime.now()} rank: {args.rank} created train dataloader with length {len(dl_train)}")

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

    if args.path_weights:
        # TO DO: Check if this setup is really needed due to ddp.
        ckpt = torch.load(args.path_weights, map_location="cpu")
        new_ckpt = OrderedDict() 
        for k, v in ckpt.items():
            name = k[7:] # remove "module."
            new_ckpt[name] = v
        model.load_state_dict(new_ckpt)
        logger.info(f"{datetime.now()} rank: {args.rank} reloaded model weights from {args.path_weights}")

    logger.info(f"{datetime.now()} rank: {args.rank} created clasp model")

    # optimizer
    opt = Adam(model.parameters(), lr = 3e-4)
    logger.info(f"{datetime.now()} rank: {args.rank} created optimizer")

    # training
    if args.rank == 0:
        train_ddp(args, model=model, optimizer=opt,
                dl_train=dl_train, dl_valid_id=dl_valid_id, dl_valid_ood=dl_valid_ood, 
                epochs=args.epochs, logger=logger, writer=writer)
    else:
        train_ddp(args, model=model, optimizer=opt,
                dl_train=dl_train, dl_valid_id=dl_valid_id, dl_valid_ood=dl_valid_ood, 
                epochs=args.epochs, logger=logger)
    logger.info(f"{datetime.now()} rank: {args.rank} training finished")


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"#gpus: {n_gpus}")
    if n_gpus < 2:
      print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
    else:
      run(trainer, n_gpus)

