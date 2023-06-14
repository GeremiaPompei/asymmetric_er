from avalanche.benchmarks.classic import SplitCIFAR100
import torch

from src.er_ace import ER_ACE
from src.er_aml import ER_AML
from src.gridsearch.gridsearch import gridsearch
from src.model.resnet18 import ResNet18


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = [
        (
            ER_AML,
            dict(
                strategy_train_mb_size=[10],
                strategy_eval_mb_size=[10],
                strategy_train_epochs=[1],
                strategy_mem_size=[100],
                strategy_batch_size_mem=[35],
                strategy_temp=[0.1, 0.2],
                sgd_lr=[0.1, 0.01, 0.001],
                sgd_momentum=[0.9]
            )
        ),
        (
            ER_ACE,
            dict(
                strategy_train_mb_size=[10],
                strategy_eval_mb_size=[10],
                strategy_train_epochs=[1],
                strategy_mem_size=[100],
                strategy_batch_size_mem=[35],
                sgd_lr=[0.1, 0.01, 0.001],
                sgd_momentum=[0.9]
            ),
        ),
    ]

    results = []
    for strategy_builder, hyperparams_list in configs:
        validation_results, test_results = gridsearch(
            strategy_builder=strategy_builder,
            benchmark_builder=SplitCIFAR100,
            validation_size=0.05,
            model_builder=ResNet18,
            hyperparams_list=hyperparams_list,
            num_workers=2,
            device=device,
            verbose=True,
            seed=0,
        )
        results.append((strategy_builder.__name__, *test_results))
    print(results)


if __name__ == '__main__':
    main()
