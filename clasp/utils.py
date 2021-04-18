import torch
from torch.utils.data import Dataset, DataLoader
import random
import time
from datetime import datetime

AA_VOCAB = "ACDEFGHIKLMNOPQRSTUVWY"
AA_DICT = {a: i for i, a in enumerate(AA_VOCAB)}


def basic_rand_sampler(seq, sample_len):
    """
    Basic random text sampler.
    If sample_len is greater than the length of the seq, the seq is returned.
    """
    seq_len   = len(seq)
    if seq_len > sample_len:
        start_idx = random.randint(0, min(seq_len,seq_len - sample_len))
        end_idx   = start_idx+sample_len
        return seq[start_idx:end_idx]
    else:
        return seq


identity_sampler = lambda x: x


def basic_aa_tokenizer(seq, context_length, return_mask=True):
    """
    Maps a number between 0 and 21 to each 21 proteogenic aminoacids.
    Unknown char input gets mapped to 22.
    """
    seq_len = len(seq)
    seq = torch.tensor([d[a] if a in aa else 22 for a in seq] + \
                       [0] * (context_length - len(seq)), dtype=torch.long)
    if return_mask:
        mask = torch.zeros_like(seq).bool()
        mask[0:seq_len] = True
        return seq, mask
    else:
        return seq


class CLASPDataset(Dataset):
    """
    Basic CLASP dataset that loads the preprocessed csv file into RAM.
        path: path to the csv file
    """
    def __init__(self, path, text_sampler, bioseq_sampler, text_tok, bioseq_tok):
        super().__init__()
        self.path = path

        tp = time.time()
        with open(path, "r") as reader:
            self.data = reader.readlines()
        print(f"Load data time: {time.time() - tp:.3f} s")

        self.cols = self.data.pop(0).split(",")
        self.len  = len(self.data)

        self.text_sampler   = text_sampler
        self.bioseq_sampler = bioseq_sampler

        self.text_tok   = text_tok
        self.bioseq_tok = bioseq_tok

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.data[idx][:-2] # without "\n"
        sample = sample.split(",")
        sample = [x for x in sample if len(x) > 0]

        text   = " ".join(sample[:-2])
        bioseq = sample[-1]

        text   = self.text_sampler(text)
        bioseq = self.bioseq_sampler(bioseq)

        text, text_mask = self.text_tok(text)
        bioseq, bioseq_mask = self.bioseq_tok(bioseq)

        return text, text_mask, bioseq, bioseq_mask


class RankSplitDataset(Dataset):
    def __init__(self, file_path, offset_dict, rank, world_size, logger=None):
        self.file_path        = file_path
        self.offset_dict      = offset_dict
        self.total_len        = len(offset_dict.keys())
        self.rank_len         = self.total_len // world_size
        self.rank_line_offset = self.rank_len * rank
        self.rank_byte_offset = self.offset_dict[str(self.rank_line_offset)] # because json keys are strings after it is saved

        if logger:
            logger.info(f"{datetime.now()} rank: {rank} dataset information:\n{'total len':>20}: {self.total_len}\n{'rank len':>20}: {self.rank_len}\n{'rank line offset':>20}: {self.rank_line_offset}\n{'rank byte offset':>20}: {self.rank_byte_offset}")
        else:
            print(f"{datetime.now()} rank: {rank} dataset information:\n{'total len':>20}: {self.total_len}\n{'rank len':>20}: {self.rank_len}\n{'rank line offset':>20}: {self.rank_line_offset}\n{'rank byte offset':>20}: {self.rank_byte_offset}")

        tp = time.time()
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.rank_byte_offset) # move to the line for the specific rank
            lines = []
            for i in range(self.rank_len): # load all the lines for the rank
                line = f.readline()
                if line != "":
                    lines.append(line)

        self.data = lines

        if logger:
            logger.info(f"{datetime.now()} rank: {rank} dataset load data time: {time.time() - tp:.3f} s")
            logger.info(f"{datetime.now()} rank: {rank} dataset len: {len(self.data)}")
        else:
            print(f"{datetime.now()} rank: {rank} dataset load data time: {time.time() - tp:.3f} s")
            print(f"{datetime.now()} rank: {rank} dataset len: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CLASPRankSplitDataset(RankSplitDataset):
    """
    CLASP rank split dataset that loads equally sized pieces for each rank
    of the preprocessed csv file into RAM.
        path: path to the csv file
    """
    def __init__(self, file_path, offset_dict, rank, world_size, logger,
                 text_sampler, bioseq_sampler, text_tok, bioseq_tok):
        super().__init__(file_path, offset_dict, rank, world_size, logger)

        self.text_sampler   = text_sampler
        self.bioseq_sampler = bioseq_sampler

        self.text_tok   = text_tok
        self.bioseq_tok = bioseq_tok

    def __getitem__(self, idx):
        sample = self.data[idx][:-1] # without "\n"
        sample = sample.split(",")
        sample = [x for x in sample if len(x) > 0]

        text   = " ".join(sample[:-1])
        bioseq = sample[-1]

        text   = self.text_sampler(text)
        bioseq = self.bioseq_sampler(bioseq)

        text, text_mask = self.text_tok(text)
        bioseq, bioseq_mask = self.bioseq_tok(bioseq)

        return text, text_mask, bioseq, bioseq_mask

