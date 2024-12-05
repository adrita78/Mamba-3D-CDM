import torch

class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`."""
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, adj, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class Regressor(torch.nn.Module):
    def __init__(self, max_node_num, max_feat_num, depth, nhid, dropout):
        super().__init__()

        self.linears = torch.nn.ModuleList([torch.nn.Linear(max_feat_num, nhid)])
        for _ in range(depth - 1):
            self.linears.append(torch.nn.Linear(nhid, nhid))

        self.convs = torch.nn.ModuleList(
            [DenseGCNConv(nhid, nhid) for _ in range(depth)]
        )

        dim = max_feat_num + depth * nhid
        dim_out = nhid

        self.sigmoid_linear = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_out), torch.nn.Sigmoid()
        )
        self.tanh_linear = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_out), torch.nn.Tanh()
        )

        self.final_linear = [
            torch.nn.Linear(dim_out, nhid),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(nhid, 2),
        ]
        self.final_linear = torch.nn.Sequential(*self.final_linear)


    def bernoulli_likelihood(mean_output, optimal_value):
      return torch.exp(mean_output - optimal_value)

    def compute_score_function(x_t, model, goal_value):
      x_t = x_t.requires_grad_(True)
      mean_output, _ = model(x_t)
      likelihood = bernoulli_likelihood(mean_output, goal_value)
      log_likelihood = torch.log(likelihood + 1e-6)
      log_likelihood.sum().backward()
      return x_t.grad
