from abc import ABC, abstractmethod

import torch


class FeaturesMapModel(ABC):

    @abstractmethod
    def return_hidden(self, X: torch.Tensor) -> torch.Tensor:
        pass
