## CLASP - Contrastive Language-Aminoacid Sequence Pretraining

Repository for creating models pretrained on language and aminoacid sequences similar to [ConVIRT](https://arxiv.org/abs/2010.00747), [CLIP](https://openai.com/blog/clip/), and [ALIGN](https://arxiv.org/abs/2102.05918).

Work in progress - more updates soon!

## Requirements

You can install the requirements with the following

```bash
$ python setup.py install --user
```

Then, you must bear the installation process for Microsoft's superb deepspeed sparse attention CUDA kernel.

Firstly

```bash
$ sh install_deepspeed.sh
```

Next, you need to install the pip package `triton`

```bash
$ pip install triton
```

If both of the above succeeded, now you can train your long biosequences efficiently!

## Usage

```python
import torch

from clasp import CLASP
from clasp.transformer import Transformer

# instantiate the attention models for text and bioseq

text_enc = Transformer(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    seq_len = 1024
)

bioseq_enc = Transformer(
    num_tokens = 21,
    dim = 512,
    depth = 6,
    seq_len = 512,
    sparse_attn = True
)

# clasp (CLIP) trainer

clasp = CLASP(
    text_encoder = text_enc,
    bioseq_encoder = bioseq_enc
)

# data

text = torch.randint(0, 20000, (1, 1024))
text_mask = torch.ones_like(text).bool()

bioseq = torch.randint(0, 21, (1, 512))
bioseq_mask = torch.ones_like(bioseq).bool()

# do the below with large batch sizes for many many iterations

loss = clasp(
    text,
    bioseq,
    text_mask = text_mask,
    bioseq_mask = bioseq_mask
)

loss.backward()
```

## Resources

See [interesting resources](https://github.com/MicPie/clasp/blob/main/resources.md) (feel free to add interesting material that could be useful).


## Citations

```bibtex
@article{zhang2020contrastive,
  title={Contrastive learning of medical visual representations from paired images and text},
  author={Zhang, Yuhao and Jiang, Hang and Miura, Yasuhide and Manning, Christopher D and Langlotz, Curtis P},
  journal={arXiv preprint arXiv:2010.00747},
  year={2020}
}
```

[OpenAI blog post "CLIP: Connecting Text and Images"](https://openai.com/blog/clip/)

```bibtex
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
```

```bibtex
@article{jia2021scaling,
  title={Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision},
  author={Jia, Chao and Yang, Yinfei and Xia, Ye and Chen, Yi-Ting and Parekh, Zarana and Pham, Hieu and Le, Quoc V and Sung, Yunhsuan and Li, Zhen and Duerig, Tom},
  journal={arXiv preprint arXiv:2102.05918},
  year={2021}
}
```
