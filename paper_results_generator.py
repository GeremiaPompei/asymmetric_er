import torch
from avalanche.training import Naive, GDumb
from avalanche.training import ER_ACE as ER_ACE_AVALANCHE

from src.benchmark import split_cifar100
from src.er_ace import ER_ACE
from src.er_aml import ER_AML
from src.gridsearch import gridsearch
from src.model import ResNet18
from src.utils import log


def main():
    """
    Main function of paper results generator file able to compute results related to the gridsearch function run for
    each strategy to compare.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = [
        (
            'Naive',
            Naive,
            dict(
                strategy_train_mb_size=[10],
                strategy_eval_mb_size=[10],
                strategy_train_epochs=[1],
                sgd_lr=[0.1, 0.01, 0.001],
                sgd_momentum=[0]
            )
        ),
        (
            'GDumb',
            GDumb,
            dict(
                strategy_train_mb_size=[10],
                strategy_eval_mb_size=[10],
                strategy_train_epochs=[1],
                strategy_mem_size=[100 * 100],
                sgd_lr=[0.1, 0.01, 0.001],
                sgd_momentum=[0]
            )
        ),
        (
            'ER_ACE_AVALANCHE',
            ER_ACE_AVALANCHE,
            dict(
                strategy_train_mb_size=[10],
                strategy_eval_mb_size=[10],
                strategy_train_epochs=[1],
                strategy_mem_size=[100 * 100],
                strategy_batch_size_mem=[10],
                sgd_lr=[0.1, 0.01, 0.001],
                sgd_momentum=[0]
            ),
        ),
        (
            'ER_AML',
            ER_AML,
            dict(
                strategy_train_mb_size=[10],
                strategy_eval_mb_size=[10],
                strategy_train_epochs=[1],
                strategy_mem_size=[100 * 100],
                strategy_batch_size_mem=[10],
                strategy_temp=[0.1, 0.2],
                sgd_lr=[0.1, 0.01, 0.001],
                sgd_momentum=[0]
            )
        ),
        (
            'ER_ACE',
            ER_ACE,
            dict(
                strategy_train_mb_size=[10],
                strategy_eval_mb_size=[10],
                strategy_train_epochs=[1],
                strategy_mem_size=[100 * 100],
                strategy_batch_size_mem=[10],
                sgd_lr=[0.1, 0.01, 0.001],
                sgd_momentum=[0]
            ),
        ),
    ]

    for strategy_name, strategy_builder, hyperparams_list in configs:
        log.info(f'STRATEGY "{strategy_name}"')
        gridsearch(
            strategy_builder=strategy_builder,
            benchmark_builder=split_cifar100,
            validation_size=0.1,
            model_builder=ResNet18,
            hyperparams_list=hyperparams_list,
            num_workers=1,
            device=device,
            verbose=True,
            seed=0,
            file_to_save='paper_results.json',
            name=strategy_name,
        )


if __name__ == '__main__':
    main()
