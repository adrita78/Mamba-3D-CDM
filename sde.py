import math
import torch
import torch.nn.functional as F
import requests
import gc
import numpy as np
import torch as th

import enum
import math

import numpy as np
import torch as th

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps,beta_start,beta_end):
    """
    scheduler for VP model
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        
        beta_start = beta_start / num_diffusion_timesteps
        beta_end = beta_end / num_diffusion_timesteps
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class VP_Diffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    
    """

    def __init__(
        self,
        betas,
        loss_type,
        c = 0.0,
        beta_start=0.1, 
        beta_max=0.999, 
        N=1000, 
        schedule_name="cosine"
    ):
       
        self.loss_type = loss_type
        self.c = c

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)    

     def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A perturbed version of x_start.
        """
        x_start = x_start.float()

        if noise is None:
            noise = th.randn_like(x_start)

        broadcast_shape = x_start.shape
        sqrt_alpha_tensor = _extract_into_tensor(self.sqrt_alphas_cumprod, t, broadcast_shape)
        sqrt_one_minus_alpha_tensor = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, broadcast_shape)

        perturbed_x = sqrt_alpha_tensor * x_start + sqrt_one_minus_alpha_tensor * noise
        return perturbed_x    


     def marginal_prob(self, x, t):
        """
        Compute the marginal mean and std.
        """
        log_mean_coeff = -0.5 * t ** 2 * (self.beta_max - self.beta_start) - 0.5 * t * self.beta_start
        mean = th.exp(log_mean_coeff[:, None, None]) * x
        std = th.sqrt(1.0 - th.exp(2.0 * log_mean_coeff))
        return mean, std
