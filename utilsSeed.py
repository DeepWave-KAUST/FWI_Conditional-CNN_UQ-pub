import random
import numpy as np
import torch

def mask(velocity, water_velocity=1.5, device="cpu"):
    """
    Create a mask for the velocity model.

    Args:
        velocity: velocity model
        water_velocity: water velocity
    Returns:
        Mask
    """
    msk = torch.zeros_like(velocity)
    msk[velocity >= water_velocity] = 1
    return msk.to(device)

def set_seed(seed):
    """Set all random seeds to a fixed value and take out any
    randomness from cuda kernels

    Parameters
    ----------
    seed : :obj:`int`
        Seed number

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True