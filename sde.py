import math
import torch
import torch.nn.functional as F
import requests
import gc
import numpy as np
import torch as th
from torchinfo import summary

from time import time
class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000, schedule_name="cosine"):
        super().__init__(N)

        # Define beta schedule
        self.betas = get_named_beta_schedule(schedule_name, N, beta_min, beta_max)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        """
        x_start = x_start.float()

        if noise is None:
            noise = th.randn_like(x_start)

        # Ensure broadcastable shapes
        broadcast_shape = x_start.shape
        sqrt_alpha_tensor = _extract_into_tensor(self.sqrt_alphas_cumprod, t, broadcast_shape)
        sqrt_one_minus_alpha_tensor = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, broadcast_shape)

        # Compute noisy x_t
        perturbed_x = sqrt_alpha_tensor * x_start + sqrt_one_minus_alpha_tensor * noise
        return perturbed_x

    def marginal_prob(self, x, t):
        """
        Compute the marginal mean and std.
        """
        log_mean_coeff = -0.5 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = th.exp(log_mean_coeff[:, None, None]) * x
        std = th.sqrt(1.0 - th.exp(2.0 * log_mean_coeff))
        return mean, std
