from avalanche.evaluation import PluginMetric


class BatchNormTracker(PluginMetric[dict]):

    def __init__(self):
        super().__init__()
        self.running_mean, self.running_var, self.weight, self.bias = [], [], [], []

    def reset(self) -> None:
        pass

    def result(self) -> dict:
        return dict(
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias
        )

    def after_training_epoch(self, strategy: "SupervisedTemplate"):
        model = strategy.model
        self.running_mean.append(model.bn1.running_mean.mean().item())
        self.running_var.append(model.bn1.running_var.mean().item())
        self.weight.append(model.bn1.weight.mean().item())
        self.bias.append(model.bn1.bias.mean().item())

    def __str__(self):
        return "BatchNormTracker"
