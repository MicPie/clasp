import torch
from torch.utils.data import Dataset, DataLoader
import random
import time


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
    aa = "ACDEFGHIKLMNOPQRSTUVWY"
    d = {a: i for i, a in enumerate(aa)}
    seq_len = len(seq)
    seq_empty = torch.zeros(context_length - len(seq), dtype=torch.long)
    seq_tok   = torch.tensor([d[a] if a in aa else 22 for a in seq], dtype=torch.long)
    seq = torch.cat([seq_tok, seq_empty], dim=0)
    if return_mask:
        mask = torch.zeros_like(seq).bool()
        mask[0:seq_len+1] = True
        return seq, mask
    else:
        return seq


class CLASPDataset(Dataset):
    """
    Basic CLASP dataset that loads preprocess csv file into RAM.
        path: path to the csv file
    """
    def __init__(self, path, text_sampler, bioseq_sampler, text_tok, bioseq_tok):
        self.path = path

        tp = time.time()
        with open(path, "r") as reader:
            self.data = reader.readlines()
        print(f"Load data time: {time.time() - tp:.3f} s")

        self.cols = self.data.pop(0).split(",")
        self.len = len(self.data)

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

