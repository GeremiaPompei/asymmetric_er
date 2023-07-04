import random

import numpy as np
import torch


def fix_seed(seed: int):
    """
    Function able to set manually the seed for different libraries.
    @param seed: Value of the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)