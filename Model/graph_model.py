import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
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
from torch_geometric.utils import to_dense_batch

from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index
import torch_sparse


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
        # order_by_degree flag is set to True, this section reorders the nodes based on their degrees
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
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        ### Global attention transformer-style model.
        if self.att_type == 'transformer':
            h, mask = to_dense_batch(x, batch)
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
            h = h[mask]

        if self.att_type == 'mamba':

            if self.order_by_degree:
              # degree() function: edge_index[0]= source nodes of the edges, x.shape[0]= number of nodes)
                deg = degree(edge_index[0], x.shape[0]).to(torch.long)
                order_tensor = torch.stack([batch, deg], 1).T
                # creates a tensor by stacking the 'batch' tensor(batch_index of each node) and the degree of each node. making it easier to reorder nodes
                _, x = sort_edge_index(order_tensor, edge_attr=x)
                # reorders the feature tensor x based on the node degree and batch information, reordering x as well.

            if self.shuffle_ind == 0:
              # no shuffling occurs, and dense batch repr. is created from x
                h, mask = to_dense_batch(x, batch)
                h = self.self_attn(h)[mask]
                # the result is masked to remove invalid entries (padded entries)
                #
            else:
                mamba_arr = []
                # list to store the results of multiple attention passes over shuffled node features
                for _ in range(self.shuffle_ind):
                  # loops over the number of shuffles(shuffle_ind)
                    h_ind_perm = permute_within_batch(x, batch)

                    h_i, mask = to_dense_batch(x[h_ind_perm], batch)

                    h_i = self.self_attn(h_i)[mask][h_ind_perm]
                    # The shuffled dense batch is passed through the self-attention mechanism, and the result is reordered back to the original node order (h_ind_perm).
                    mamba_arr.append(h_i)
                h = sum(mamba_arr) / self.shuffle_ind

                # Averages the results from all shuffled versions to produce the final node representations.
        ###

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

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

class GraphModel(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, model_type: str, shuffle_ind: int, d_state: int, d_conv: int, order_by_degree: False, in_node_features: int, in_edge_features: int,
                 sh_lmax: int, ns: int, nv: int, num_conv_layers: int, max_radius: int, radius_embed_dim: int, use_second_order_repr=True, batch_norm=True, residual=True):
        super().__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.time_embed_dim = time_embed_dim
        self.num_layers = num_layers
        self.channels = channels
        self.pe_dim = pe_dim
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.d_state = d_state
        self.d_conv = d_conv
        self.order_by_degree = order_by_degree
        self.max_radius = max_radius
        self.radius_embed_dim = radius_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv


        self.node_emb = Embedding(in_node_features + time_embed_dim, channels - pe_dim)
        self.edge_emb = Embedding(in_edge_features + time_embed_dim, channels)

        # Graph Convolution layers
        self.convs = ModuleList()
        # A list of graph convolutional layers

        # Creates a two-layer neural network (nn), used within each GINEConv layer
        #for _ in range(num_layers):
            #nn = Sequential(
                #Linear(channels, channels),
                #ReLU(),
                #Linear(channels, channels),
            #)
            if self.model_type == 'e3nn':
                conv = TensorProductEquivariantModel(in_node_features: int, in_edge_features: int,sh_lmax: int, ns: int, nv: int,
                                                     num_conv_layers: int, max_radius: int, radius_embed_dim: int,
                                                     use_second_order_repr=True, batch_norm=True, residual=True)

            if self.model_type == 'mamba':
                conv = GPSConv(channels, TensorProductEquivariantModel(in_node_features: int, in_edge_features: int, sh_lmax: int, ns: int, nv: int, num_conv_layers: int, max_radius: int,
                                                                       radius_embed_dim: int, use_second_order_repr=True, batch_norm=True, residual=True),
                                                                       heads=4, attn_dropout=0.5, att_type='mamba',shuffle_ind=self.shuffle_ind,
                                                                       order_by_degree=self.order_by_degree, d_state=d_state, d_conv=d_conv)

            if self.model_type == 'transformer':
                conv = GPSConv(channels, TensorProductEquivariantModel(in_node_features: int, in_edge_features: int, sigma_embed_dim: int,
                                                                       sh_lmax: int, ns: int, nv: int, num_conv_layers: int, max_radius: int,
                                                                       radius_embed_dim: int, use_second_order_repr=True,
                                                                       batch_norm=True, residual=True), heads=4, attn_dropout=0.5,
                                                                       att_type='transformer')

            # conv = GINEConv(nn)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )

    def forward(self, x, pe, t, edge_index, edge_attr, batch):
      # x: Node input features
      # pe: Positional encoding for nodes
      # edge_index: edge indices of the graph
      # edge_attr: edge attributes
      # batch
        x_pe = self.pe_norm(pe)
        # Batch normalization to positional encodings

        time_emb = timestep_embediing(t, self.time_embed_dim)

        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        # embedding applied to the node features
        # linear transformation applied to x_pe
        # concatenate along feature dimension

        edge_attr = self.edge_emb(edge_attr)
        # edge embedding applied to edge attributes, transforms into the same dim. as nodes

        for conv in self.convs:
            if self.model_type == 'gine':
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x = global_add_pool(x, batch)
        # node features are pooled at the graph level using global pooling
        return self.mlp(x)
        

def permute_within_batch(x, batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices

# Example usage
batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
x = torch.tensor([0, 10, 20, 30, 40, 50, 60])

# Get permuted indices
permuted_indices = permute_within_batch(x, batch)

# Use permuted indices to get the permuted tensor
permuted_x = x[permuted_indices]

print("Original x:", x)
print("Permuted x:", permuted_x)
print("Permuted indices:", permuted_indices)

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
