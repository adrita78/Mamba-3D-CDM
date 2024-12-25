import argparse
import os.path as osp
from typing import Any, Dict, Optional
import math
import torch as th
import egnn_clean as eg

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
import gc
from torch_geometric.nn import global_add_pool
import inspect
from typing import Any, Dict, Optional
from utils.GPSConv import GPSConv

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential
from torch_geometric.utils import to_dense_batch

from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index
import torch_sparse


class GraphModel(torch.nn.Module):
    def __init__(self, channels: int, in_node_nf: int, out_node_nf: int, in_edge_nf: int, hidden_nf: int, pe_dim: int, context_dim: int, time_embed_dim: int, num_layers: int, model_type: str, shuffle_ind: int, d_state: int, d_conv: int, order_by_degree: False):
        super().__init__()

        self.node_emb = Linear(64, channels - pe_dim - time_embed_dim - context_dim)
        self.context_emb = Linear(64, context_dim)
        self.context_dim = context_dim
        self.time_embed_dim = time_embed_dim
        self.pe_lin = Linear(20, pe_dim)    # PE_encoder = linear, DeepSet
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Linear(4, channels-time_embed_dim)
        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        self.convs = ModuleList()

        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            if self.model_type == 'egnn':
                conv = eg.EGNN(in_node_nf, hidden_nf, out_node_nf, in_edge_nf)

            if self.model_type == 'mamba':
                conv = GPSConv(channels, eg.EGNN(in_node_nf, hidden_nf, out_node_nf, in_edge_nf), heads=4, attn_dropout=0.5,
                               att_type='mamba',
                               shuffle_ind=self.shuffle_ind,
                               order_by_degree=self.order_by_degree,
                               d_state=d_state, d_conv=d_conv)

            if self.model_type == 'transformer':
                conv = GPSConv(channels, eg.EGNN(in_node_nf, hidden_nf, out_node_nf, in_edge_nf), heads=4, attn_dropout=0.5, att_type='transformer')
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )

    def forward(self, h, pe, x, t, context, edges, edge_index, edge_attr, batch):

        h_pe = self.pe_norm(pe)        # positional_encoding = RWSE-20, LapPE-8, LapPE-16
        time_emb = timestep_embedding(t, self.time_embed_dim)
        context = self.context_emb(context)
        h = torch.cat((self.node_emb(h.squeeze(-1)), self.pe_lin(h_pe), time_emb.squeeze(1), context), dim=1)

        edge_attr = self.edge_emb(edge_attr.squeeze(-1))
        required_repeats = edge_attr.size(0) // time_emb.size(0)
        remainder = edge_attr.size(0) % time_emb.size(0)
        time_emb_edge = time_emb.repeat_interleave(required_repeats, dim=0)

        if remainder > 0:
          extra_time_emb = time_emb[:remainder]
        time_emb_edge = torch.cat((time_emb_edge, extra_time_emb), dim=0)

        assert time_emb_edge.size(0) == edge_attr.size(0), "Final sizes still do not match!"
        edge_attr = torch.cat((edge_attr, time_emb_edge.squeeze(1)), dim=-1)

        for conv in self.convs:
            if self.model_type == 'egnn':
                h = conv(h, x, edges=edges, edge_attr=edge_attr)
            else:
                h = conv(h,x, edge_index, batch, edges=edges, edge_attr=edge_attr)
        h = global_add_pool(h, batch)    # graph_pool = sum, mean
        return self.mlp(h)



def timestep_embedding(timesteps, dim, max_period=10000):
      """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
      half = dim // 2
      freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
      ).to(device=timesteps.device)
      args = timesteps[:, None].float() * freqs[None]
      embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
      if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
      return embedding
