import enum
import math

import numpy as np
import torch as th

def q_sample(x_start, t, noise=None):
    """
    Diffuse the data for a given number of diffusion steps.
    """
    x_start = x_start.float()

    if noise is None:
        noise = torch.randn_like(x_start)
    assert noise.shape == x_start.shape, f"Noise shape {noise.shape} does not match x_start shape {x_start.shape}."

    broadcast_shape = x_start.shape
    sqrt_alphas_tensor = _extract_into_tensor(sqrt_alphas_cumprod, t, broadcast_shape)
    sqrt_one_minus_alphas_tensor = _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, broadcast_shape)
    noisy_x_start = (
        sqrt_alphas_tensor * x_start +
        sqrt_one_minus_alphas_tensor * noise
    )
    return noisy_x_start

"""Creating Beta Schedule"""

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

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from `arr` based on `timesteps`, expanding to `broadcast_shape`.
    """

    if timesteps.ndimension() == 0:
        timesteps = timesteps.unsqueeze(0)

    if len(broadcast_shape) > 1:
        broadcast_shape = (timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1)
    arr = torch.tensor(arr, dtype=torch.float32)
    try:
        extracted = arr[timesteps.long()]
        if extracted.numel() == 1:
            extracted = extracted.view(1)
        else:
            extracted = extracted.view(*broadcast_shape)
    except Exception as e:
        raise e

    return extracted

# define beta schedule
betas = get_named_beta_schedule(schedule_name = "cosine" , num_diffusion_timesteps= 1000 ,beta_start =  0.0001 , beta_end=0.02 )
betas = np.array(betas, dtype=np.float64)
assert len(betas.shape) == 1, "betas must be 1-D"
assert (betas > 0).all() and (betas <= 1).all()

num_timesteps = int(betas.shape[0])
print(num_timesteps)

alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)


alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)
assert alphas_cumprod_prev.shape == (num_timesteps,)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
