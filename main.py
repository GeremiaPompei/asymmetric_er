from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, StreamConfusionMatrix
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive#, ER_ACE
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from src.er_ace.er_ace import ER_ACE
from src.er_aml.er_aml import ER_AML
from src.model.resnet18 import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    num_workers = 4
    n_classes = 100
    benchmark = SplitCIFAR100(n_experiences=20, seed=0)

    # MODEL CREATION
    model = ResNet18(n_classes).to(device)

    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True),
        loggers=[interactive_logger]
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = ER_AML(
        model,
        SGD(model.parameters(), lr=0.1, momentum=0.9),
        CrossEntropyLoss(),
        train_mb_size=10,
        eval_mb_size=10,
        train_epochs=1,
        evaluator=eval_plugin,
        device=device,
        mem_size=200,
        batch_size_mem=10
    )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience, num_workers=num_workers)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(benchmark.test_stream, num_workers=num_workers))


if __name__ == '__main__':
    main()
