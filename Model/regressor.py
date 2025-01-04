import torch


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

    def compute_score_function(x_t, model, optimal_value):
      x_t = x_t.requires_grad_(True)
      mean_output, _ = model(x_t)
      likelihood = bernoulli_likelihood(mean_output, optimal_value)
      log_likelihood = torch.log(likelihood + 1e-6)
      log_likelihood.sum().backward()
      return x_t.grad
