import numpy as np
import torch.nn.functional as F
import torch as th

def ph_loss(input,target,c):
    assert input.shape == target.shape, "Input and target must have the same shape"
    return th.sqrt((input - target) ** 2 + c**2) - c
