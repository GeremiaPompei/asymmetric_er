import torch
from avalanche.evaluation import PluginMetric


class BatchNormTracker(PluginMetric[list]):

    def __init__(self):
        super().__init__()
        self.layers = []

    def reset(self) -> None:
        pass

    def result(self) -> list:
        return self.layers

    def after_training_epoch(self, strategy: "SupervisedTemplate"):
        model = strategy.model
        self.layers.append({
            i: self.__extract_from_bn_layer__(layer)
            for i, layer in enumerate(model.modules())
            if type(layer) is torch.nn.BatchNorm2d
        })

    def __extract_from_bn_layer__(self, bn_layer: torch.nn.BatchNorm2d):
        return dict(
            running_mean=torch.norm(bn_layer.running_mean).item(),
            running_var=torch.norm(bn_layer.running_var).item(),
            weight=torch.norm(bn_layer.weight).item(),
            bias=torch.norm(bn_layer.bias).item()
        )

    def __str__(self):
        return "BatchNormTracker"
