from torch_scatter import scatter
import torch_scatter
import torch

def mean_flat_graph(tensor, batch):
    """
    Take the mean over all non-batch dimensions for graph data.

    Args:
        tensor (torch.Tensor): Node or edge features of shape [num_nodes, feature_dim].
        batch (torch.Tensor): Batch vector of shape [num_nodes], associating elements to graphs.

    Returns:
        torch.Tensor: Mean value per graph, shape [num_graphs].
    """
    # Aggregate sum per graph using the batch index
    sum_per_graph = torch_scatter.scatter(tensor.sum(dim=-1), batch, dim=0, reduce="sum")

    # Count elements per graph
    count_per_graph = torch_scatter.scatter(torch.ones_like(tensor[:, 0]), batch, dim=0, reduce="sum")

    # Compute mean per graph
    mean_per_graph = sum_per_graph / count_per_graph
    return mean_per_graph








class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  # mean, std of the perturbation kernel
  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape).triu(1)#.abs()
    return x + x.transpose(-1,-2)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    # f = torch.sqrt(alpha)[:, None, None, None] * x - x
    f = torch.sqrt(alpha)[:, None, None] * x - x
    G = sqrt_beta
    return f, G

  def change_discreteization_steps(self, k):
    self.N = int(self.N * k)
    self.discrete_betas = torch.linspace(self.beta_0 / self.N, self.beta_1 / self.N, self.N)
    self.alphas = 1. - self.discrete_betas

  # def transition(self, x, t, dt):
  #   # negative timestep dt
  #   log_mean_coeff = 0.25 * dt * (2*self.beta_0 + (2*t + dt)*(self.beta_1 - self.beta_0) )
  #   # mean = torch.exp(log_mean_coeff[:, None, None]) * x
  #   mean = torch.exp(-log_mean_coeff[:, None, None]) * x
  #   std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
  #   return mean, std
