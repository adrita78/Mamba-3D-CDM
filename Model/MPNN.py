import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter
import numpy as np
from e3nn.nn import BatchNorm
from utils.MPNN_utils import TensorProductConvLayer, GaussianSmearing


class MPNNModel(torch.nn.Module):
    def __init__(self, channels= 64, in_node_features=64, in_edge_features= 64, sh_lmax=2, ns=32, nv=8, num_conv_layers=4, max_radius=5, radius_embed_dim=50,use_second_order_repr=True, batch_norm=True, residual=True
                 ):
        super(MPNNModel, self).__init__()
                     
        """
        Part of this is code taken from torsional-diffusion https://github.com/gcorso/torsional-diffusion

        """
                     
        self.max_radius = max_radius
        self.radius_embed_dim = radius_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.num_conv_layers = num_conv_layers
        self.channels = channels
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features


        self.node_embedding = nn.Sequential(
                nn.Linear(in_node_features, self.ns),
                nn.ReLU(),
                nn.Linear(self.ns, self.ns)
            )

        self.edge_embedding = nn.Sequential(
                nn.Linear( in_edge_features + self.radius_embed_dim, self.ns),
                nn.ReLU(),
                nn.Linear(self.ns, self.ns)
            )


        self.distance_expansion = GaussianSmearing(0.0, max_radius, radius_embed_dim)
        conv_layers = []

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                residual=residual,
                batch_norm=batch_norm
            )
            conv_layers.append(layer)

        self.conv_layers = nn.ModuleList(conv_layers)

        self.final_linear = None

    def forward(self, data):
      node_attr, edge_index, edge_attr, edge_sh = self.build_conv_graph(data)
      src, dst = edge_index
      node_attr = self.node_embedding(node_attr)
      edge_attr = self.edge_embedding(edge_attr)


      for layer in self.conv_layers:
        edge_attr_ = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
        node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, reduce='mean')
          
      out = node_attr
      final_input_dim = node_attr.size(-1)
      if self.final_linear is None:
            self.final_linear = nn.Sequential(
                nn.Linear(final_input_dim, self.channels, bias=False),
                nn.Tanh(),
                nn.Linear(self.channels, self.channels, bias=False)
            )

      out = self.final_linear(node_attr)
      return out



    def build_conv_graph(self, data):
      #in_edge_features = data.edge_attr.size(-1)
      radius_edges = radius_graph(data.pos, self.max_radius, data.batch)
      edge_index = torch.cat([data.edge_index, radius_edges], dim=1).long()

      if isinstance(data.edge_attr, list):
        data.edge_attr = torch.tensor(data.edge_attr)
      edge_attr = torch.cat([
        data.edge_attr,
        torch.zeros(radius_edges.shape[-1], self.in_edge_features, device=data.x.device)
    ], dim=0)

      node_attr = data.x

    # Edge vectors and their lengths
      src, dst = edge_index
      edge_vec = data.pos[dst.long()] - data.pos[src.long()]
      edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1))

    # Concatenate edge features with edge length embeddings
      edge_attr = torch.cat([edge_attr, edge_length_emb], dim=1)

    # Compute spherical harmonics for directional information
      edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

      return node_attr, edge_index, edge_attr, edge_sh

