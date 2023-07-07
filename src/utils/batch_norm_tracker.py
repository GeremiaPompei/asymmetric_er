import torch
from avalanche.evaluation import PluginMetric
from avalanche.training.templates import SupervisedTemplate


class MonitorModelBNLayers:
    """
    Class able to provide a way to monitor and store batch normalization activations mean and std for each model bn layer.
    """

    def __init__(self, model):
        """
        MonitorModelBNLayers constructor.
        @param model: Model to monitor.
        """
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
        """
        Method able to monitor a single layer.
        @param inputs: Input activations.
        """
        if self.active:
            self.layers[self.data_label][-1].append(
                (
                    inputs[0].detach().mean().item(),
                    inputs[0].detach().std().item()
                )
            )

    def monitor(self, model, inputs: torch.Tensor, data_label: str):
        """
        Method able to monitor each layer.
        @param model: Model to monitor used here to launch the inference.
        @param inputs: input activations.
        @param data_label: String that represents the label of data type to store statistics. They could be 'new' of 'buffer'.
        """
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
        self.monitor_layer: MonitorModelBNLayers = None

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
        @param strategy: Strategy where retrieve the model.
        """
        if self.monitor_layer is None:
            self.monitor_layer = MonitorModelBNLayers(strategy.model)

    def before_training_iteration(self, strategy: SupervisedTemplate):
        """
        Monitor bn layers before training iteration.
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
