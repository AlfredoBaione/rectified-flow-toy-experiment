import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import torch
from torch.nn import Linear, Embedding
from torch.nn.functional import silu


class GroupNorm(torch.nn.Module):
    # from https://github.com/NVlabs/edm/blob/main/training/networks.py#L96C1-L106C17
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=1, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype),
                                           bias=self.bias.to(x.dtype), eps=self.eps)
        return x


class PositionalEmbedding(torch.nn.Module):
    # from https://github.com/NVlabs/edm/blob/main/training/networks.py#L193
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.outer(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class Block(torch.nn.Module):
    # adapted from:
    # https://github.com/NVlabs/edm/blob/main/training/networks.py#L134C1-L187C17
    # and
    # https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L372
    def __init__(self,
                 in_channels, model_channels, out_channels,
                 num_labels=0,
                 eps=1e-5,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        emb_channels = model_channels
        self.out_channels = out_channels
        self.empty_label = num_labels

        # sigma embedding
        self.t_embedder = PositionalEmbedding(num_channels=model_channels)
        self.n_dense0 = Linear(in_features=emb_channels, out_features=model_channels * 2, **init)

        # class label embedding
        if num_labels:
            self.c_embeddings = Embedding(num_labels + 1, embedding_dim=emb_channels)  # +1 because void label
            self.c_dense0 = Linear(in_features=emb_channels, out_features=model_channels * 2, **init)
        else:
            self.c_embeddings = None
            self.c_dense0 = None

        # data
        self.dense0 = Linear(in_features=in_channels, out_features=model_channels, **init)
        self.norm1 = GroupNorm(num_channels=model_channels, eps=eps)
        self.dense1 = Linear(in_features=model_channels, out_features=model_channels * 2, **init)
        self.norm2 = GroupNorm(num_channels=model_channels * 2, eps=eps)
        self.dense2 = Linear(in_features=model_channels * 2, out_features=model_channels, **init)
        self.norm3 = GroupNorm(num_channels=model_channels, eps=eps)
        self.dense3 = Linear(in_features=model_channels, out_features=out_channels, **init)

    def forward(self, x, t, c_labels=None):

        # noise embedding
        emb = self.t_embedder(t)
        emb = silu(self.n_dense0(emb))
        # class conditioning
        if self.c_embeddings:
            if c_labels is None:
                c_labels = torch.full((x.shape[0],), self.empty_label)
            c_emb = self.c_embeddings(c_labels)
            c_emb = silu(self.c_dense0(c_emb))
            emb = emb + c_emb
        # data
        x = self.dense0(x)
        x = self.dense1(silu(self.norm1(x)))
        emb = emb.to(x.dtype)
        x = self.dense2(silu(self.norm2(x.add_(emb))))
        x = self.dense3(silu(self.norm3(x)))
        return x


class ToyModel(torch.nn.Module):
    """ Adapted from:
    https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L632
    """
    def __init__(self,
                 kind='b'
    ):
        super().__init__()
        self.kind = kind
        self.block = Block(in_channels=2, model_channels=128, out_channels=2)

    def forward(self, x, t):
        x = x.to(torch.float32)
        t = t.to(torch.float32).flatten()  #.reshape(1, 1)
        return self.block(x, t)
