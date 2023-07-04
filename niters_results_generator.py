import json

import torch

from src.benchmark import split_cifar100
from src.er_ace import ER_ACE
from src.er_aml import ER_AML
from src.gridsearch.strategy_runner import run_strategy
from src.model import ResNet18
from src.utils import log
from src.utils.batch_norm_tracker import BatchNormTracker
from src.utils.fs import save_record_in_file, read_file


def main():
    """
    Main function of niters results generator file able to compute results related to the run of strategies ER_AML and
    ER_ACE with different niters param to study the behavior of them. The main focus is related to the change of batch
    normalization features.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    file_to_save = 'niters_results.json'

    configs = [
        *[
            (
                ER_AML,
                dict(
                    strategy_train_mb_size=10,
                    strategy_eval_mb_size=10,
                    strategy_train_epochs=1,
                    strategy_mem_size=100 * 100,
                    strategy_batch_size_mem=10,
                    strategy_temp=0.1,
                    strategy_n_iters=i,
                    sgd_lr=0.1,
                    sgd_momentum=0
                )
            ) for i in [1, 2, 4]
        ],
        *[
            (
                ER_ACE,
                dict(
                    strategy_train_mb_size=10,
                    strategy_eval_mb_size=10,
                    strategy_train_epochs=1,
                    strategy_mem_size=100 * 100,
                    strategy_batch_size_mem=10,
                    strategy_n_iters=i,
                    sgd_lr=0.1,
                    sgd_momentum=0
                )
            ) for i in [1, 2, 4]
        ]
    ]

    try_read_file = read_file(file_to_save)
    for strategy_builder, hyperparams in configs:
        strategy_name = f'{strategy_builder.__name__} [niters={hyperparams["strategy_n_iters"]}]'
        log.info(f'STRATEGY "{strategy_name}"')
        if try_read_file is not None and strategy_name in try_read_file:
            continue
        bn_tracker = BatchNormTracker()
        benchmark = split_cifar100(n_experiences=20, seed=0)
        AAA, accuracy, info = run_strategy(
            strategy_builder=strategy_builder,
            train_stream=benchmark.train_stream,
            eval_stream=benchmark.test_stream,
            model_builder=ResNet18,
            hyperparams=hyperparams,
            n_classes=benchmark.n_classes,
            num_workers=0,
            plugins=[bn_tracker],
            device=device
        )
        results = dict(
            AAA=AAA,
            accuracy=accuracy,
            hyperparams=json.dumps(hyperparams),
            info=info,
            bn_tracker=bn_tracker.result()
        )
        save_record_in_file(file_to_save, strategy_name, results)
        bn_tracker.result()


if __name__ == '__main__':
    main()
