import torch
from avalanche.evaluation import PluginMetric
from avalanche.training.templates import SupervisedTemplate


class MonitorLayer:

    def __init__(self, model):
        self.layers = dict(
            buffer=[],
            new=[]
        )
        self.active = False
        self.data_label = ''
        for i, layer in enumerate(model.modules()):
            if type(layer) is torch.nn.BatchNorm2d:
                layer.register_forward_hook(lambda m, inputs, o: self.monitor_layer(inputs))

    def monitor_layer(self, inputs: torch.Tensor):
        if self.active:
            self.layers[self.data_label][-1].append(
                (
                    inputs[0].detach().mean().item(),
                    inputs[0].detach().std().item()
                )
            )

    def monitor(self, model, inputs, data_label):
        self.data_label = data_label
        self.layers[data_label].append([])
        model(inputs)


class BatchNormTracker(PluginMetric[dict]):
    """
    Batch normalization tracker able to collect statistics according batch normalization layers.
    """

    def __init__(self):
        """
        Batch normalization tracker constructor.
        """
        super().__init__()
        self.monitor_layer: MonitorLayer = None

    def reset(self) -> None:
        pass

    def result(self) -> dict:
        """
        Method able to return statistics collected during training.
        @return: Statistics collected.
        """
        return self.monitor_layer.layers

    def before_training(self, strategy: SupervisedTemplate):
        """
        Method able to initialize the monitor layer object before training.
        @param strategy:
        @return:
        """
        if self.monitor_layer is None:
            self.monitor_layer = MonitorLayer(strategy.model)

    def before_training_iteration(self, strategy: SupervisedTemplate):
        """
        Monitor bn layer before training iteration.
        @param strategy: Strategy tracked.
        """
        try:
            if strategy.mb_buffer_x is not None:
                strategy.model.eval()
                self.monitor_layer.active = True
                self.monitor_layer.monitor(strategy.model, strategy.mb_x, 'new')
                self.monitor_layer.monitor(strategy.model, strategy.mb_buffer_x, 'buffer')
                self.monitor_layer.active = False
                strategy.model.train()
        except:
            pass

    def __str__(self):
        """
        String representation of this object.
        @return: String representation.
        """
        return "BatchNormTracker"
