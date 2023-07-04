import torch


def scale_by_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Function able to scale by its norm a certain tensor.
    @param x: Tensor to normalize.
    @return: Normalized tensor.
    """
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    return x / (x_norm + 1e-05)
