import argparse
import os.path as osp
from typing import Any, Dict, Optional
import math
import torch as th

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
import inspect
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj

class GPSConv(torch.nn.Module):

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = 'relu',
        att_type: str = 'transformer',
        order_by_degree: bool = False,
        shuffle_ind: int = 0,
        d_state: int = 16,
        d_conv: int = 4,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree

        assert (self.order_by_degree==True and self.shuffle_ind==0) or (self.order_by_degree==False), f'order_by_degree={self.order_by_degree} and shuffle_ind={self.shuffle_ind}'
        if self.att_type == 'transformer':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                dropout=attn_dropout,
                batch_first=True,
            )
        if self.att_type == 'mamba':
            self.self_attn = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=1
            )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            y,x = self.conv(h, x, **kwargs)
            y = F.dropout(h, p=self.dropout, training=self.training)
            y = y + h
            if self.norm1 is not None:
                if self.norm_with_batch:
                    y = self.norm1(y, batch=batch)
                else:
                    y = self.norm1(y)
            hs.append(y)
            print("shape of hs from MPNN:", h.shape)

        ### Global attention transformer-style model.
        if self.att_type == 'transformer':
            y, mask = to_dense_batch(h, batch)
            y, _ = self.attn(y, y, y, key_padding_mask=~mask, need_weights=False)
            h = h[mask]

        if self.att_type == 'mamba':

            if self.order_by_degree:
                deg = degree(edge_index[0], h.shape[0]).to(torch.long)
                order_tensor = torch.stack([batch, deg], 1).T
                _, h = sort_edge_index(order_tensor, edge_attr=x)
            if self.shuffle_ind == 0:
                y, mask = to_dense_batch(h, batch)
                y = self.self_attn(y)[mask]

            else:
                mamba_arr = []
                for _ in range(self.shuffle_ind):
                    y_ind_perm = permute_within_batch(h, batch)
                    y_i, mask = to_dense_batch(h[y_ind_perm], batch)
                    y_i = self.self_attn(y_i)[mask][y_ind_perm]
                    mamba_arr.append(y_i)
                y = sum(mamba_arr) / self.shuffle_ind
                print("Shape of h from mamba:", y.shape)

                # Averages the results from all shuffled versions to produce the final node representations.
        ###

        y = F.dropout(y, p=self.dropout, training=self.training)
        y = y + h  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                y = self.norm2(y, batch=batch)
            else:
                y = self.norm2(y)
        hs.append(y)
        print("shape of hs from attention:", h.shape)

        out = sum(hs)  # Combine local and global outputs.
        print("Shape of out:", out.shape)

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')
