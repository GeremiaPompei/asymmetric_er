from abc import ABC, abstractmethod

import torch


class FeaturesMapModel(ABC):
    """
    Interface implemented by all models that have a feature extractor.
    """

    @abstractmethod
    def return_hidden(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method able to retrieve the latent versio of a certain input.
        @param X: Input items.
        @return: Latent items.
        """
        pass
