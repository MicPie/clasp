import torch
from torch import nn, einsum
import torch.nn.functional as F

class CLASP(nn.Module):
    def __init__(
        self,
        *,
        text_encoder,
        bioseq_encoder
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.bioseq_encoder = bioseq_encoder
        self.temperature = nn.Parameter(torch.tensor(1.))

    def forward(
        self,
        text,
        bioseq,
        text_mask = None,
        bioseq_mask = None,
        return_loss = False
    ):
        b, device = text.shape[0], text.device

        text_latents = self.text_encoder(text, mask = text_mask)
        bioseq_latents = self.bioseq_encoder(bioseq, mask = bioseq_mask)

        text_latents, bioseq_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (text_latents, bioseq_latents))

        temp = self.temperature.exp()

        if not return_loss:
#            sim = einsum('n d, n d -> n', text_latents, bioseq_latents) * temp
#            return sim
            return text_latents, bioseq_latents, temp

        sim = einsum('i d, j d -> i j', text_latents, bioseq_latents) * temp
        labels = torch.arange(b, device = device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss.mean()
