import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from clasp import CLASP, Transformer, tokenize, basic_rand_sampler, basic_aa_tokenizer, CLASPDataset
from functools import partial


# setup data

text_sampler = partial(basic_rand_sampler, sample_len=1024)
text_tok     = partial(tokenize, context_length=1024, return_mask=True)

bioseq_sampler = partial(basic_rand_sampler, sample_len=512)
bioseq_tok     = partial(basic_aa_tokenizer, context_length=512, return_mask=True)

ds = CLASPDataset(path="/mnt/ssd-cluster/clasp/data/uniprot_sprot.csv",
                  text_sampler=text_sampler,
                  bioseq_sampler=bioseq_sampler,
                  text_tok=text_tok,
                  bioseq_tok=bioseq_tok)

dl = DataLoader(ds, 32, True, num_workers=8, drop_last=True)


# instantiate the attention models for text and bioseq

text_enc = Transformer(
    num_tokens = 49408,
    dim = 512,
    depth = 6,
    seq_len = 1024
)

bioseq_enc = Transformer(
    num_tokens = 23,
    dim = 512,
    depth = 6,
    seq_len = 512,
#    sparse_attn = True
)


# clasp (CLIP) trainer

clasp = CLASP(
    text_encoder = text_enc,
    bioseq_encoder = bioseq_enc
)


opt = Adam(clasp.parameters(), lr = 3e-4)

device = torch.device("cuda:0")
clasp.to(device)

for b in dl:
    text, text_mask, bioseq, bioseq_mask = b
    text = text.to(device).squeeze(1)
    text_mask = text_mask.to(device).squeeze(1)
    bioseq.to(device)
    bioseq_mask.to(device)

    print("text.shape: ",text.shape)
    print("text_mask.shape: ",text_mask.shape)
    print("bioseq.shape: ",bioseq.shape)
    print("bioseq_mask.shape: ",bioseq_mask.shape)

    loss = clasp(
        text,
        bioseq,
        text_mask = text_mask,
        bioseq_mask = bioseq_mask,
        return_loss = True               # set return loss to True
    )

    loss.backward()

    print(loss)
