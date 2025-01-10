import torch
import torch.nn as nn
import torch.nn.functional as F
from sde import VPSDE
from torch_geometric.nn import GCNConv

class Regressor(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, dropout):
        super().__init__()

        self.linears = torch.nn.ModuleList([torch.nn.Linear(max_feat_num, nhid)])
        for _ in range(depth - 1):
            self.linears.append(torch.nn.Linear(nhid, nhid))

        self.convs = torch.nn.ModuleList(
            [GCNConv(nhid, nhid) for _ in range(depth)]
        )

        dim = max_feat_num + depth * nhid
        dim_out = nhid

        self.sigmoid_linear = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_out), torch.nn.Sigmoid()
        )
        self.tanh_linear = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_out), torch.nn.Tanh()
        )

        self.final_linear = torch.nn.Sequential(
            torch.nn.Linear(dim_out, nhid),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(nhid, 2),
        )

    def forward(self, x, edge_index, edge_attr=None):
        xs = [x]
        out = x
        for lin, conv in zip(self.linears, self.convs):
            out = lin(out)
            if edge_attr is not None:
                out = conv(out, edge_index)
            else:
                out = conv(out, edge_index)
            out = torch.tanh(out)
            xs.append(out)
        out = torch.cat(xs, dim=-1)

        sigmoid_out = self.sigmoid_linear(out)
        tanh_out = self.tanh_linear(out)
        out = torch.mul(sigmoid_out, tanh_out).sum(dim=1)
        embeds = torch.tanh(out)
        preds = self.final_linear(embeds)
        preds[:, 0] = torch.sigmoid(preds[:, 0])

        return preds, embeds

class RegressorEnsemble(torch.nn.Module):
    def __init__(self,max_feat_num, depth, nhid, dropout, ensemble_size):
        super().__init__()
        self.regressors = torch.nn.ModuleList(
            [
                Regressor(max_feat_num, depth, nhid, dropout)
                for _ in range(ensemble_size)
            ]
        )

    def forward(self,x,edge_index,edge_attr=None):

        preds = []
        embeds = []
        for regressor in self.regressors:
            pred, embed = regressor(x, edge_index, edge_attr=None)
            preds.append(pred)
            embeds.append(embed)

        return preds, embeds


def get_regressor_fn(sde, model):
    if not isinstance(sde, VPSDE):
        raise ValueError(f"Expected VPSDE, but got {sde.__class__.__name__}")

    def regressor_fn(x, edge_index, t):
        pred, embed = model(x, edge_index)
        return pred, embed

    return regressor_fn

class GuidanceScore(torch.nn.Module):
    def __init__(self, sde, Regressor, betas, N=1000, T=1):
        super().__init__()
        self.sde = sde
        self.regressor = get_regressor_fn(sde, Regressor)
        self.betas = betas
        self.N = N
        self.T = T

    def forward(self, x, edge_index, t ):
        """
        Compute the integral for the provided inputs.

        Args:
            x: Node features (torch.Tensor).
            edge_index: Graph edges (torch.Tensor).
            optimal_value: Optimal target value for likelihood computation.

        Returns:
            integral: Approximated value of the integral.
        """
        with torch.enable_grad():
            x_para = torch.nn.Parameter(x)
        dt = 1 / self.N  
        integral = 0.0  
        timestep = (t * (self.N - 1) / self.T).long()  
        beta = self.betas[timestep] 
        F, _ = self.regressor(x_para, edge_index, t)
        if isinstance(F, list):
          F = torch.stack(F, dim=0).mean(dim=0)
        likelihood = bernoulli_likelihood(F, optimal_value)
        log_likelihood = torch.log(likelihood + 1e-6)
        log_likelihood.sum().backward()
        grad_log_prob = x_para.grad.clone() 
        x_para.grad.zero_() 
        integral += beta * grad_log_prob * dt

        return integral


def bernoulli_likelihood(mean_value, optimal_value):
      
      output = torch.exp(mean_value - optimal_value)
      return output / (1 + output)
