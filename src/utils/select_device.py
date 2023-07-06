import torch
from src.utils.logging import log


def select_device() -> str:
    """
    Function able to select accelerator device.
    @return: String related to de accelerator.
    """
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'
    log.info(f'Selected device is {device}')
    return device
