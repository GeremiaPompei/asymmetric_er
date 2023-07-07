import json

from src.benchmark import split_cifar100
from src.er_ace import ER_ACE
from src.er_aml import ER_AML
from src.generators.generators import Generator
from src.gridsearch.strategy_runner import run_strategy
from src.model import ResNet18
from src.utils import log
from src.utils.fs import save_record_in_file, read_file

from src.utils.select_device import select_device


class NItersGenerator(Generator):
    """
    NIters generator class able to train ER_ACE and ER_AML strategies with different niters hyperparams.
    """

    def __init__(self):
        super(NItersGenerator, self).__init__('NITERS')

    def __call__(self, file_to_save=None):
        if file_to_save is None:
            file_to_save = 'data/niters_results.json'
        device = select_device()

        configs = [
            *[
                (
                    'ER_ACE',
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
            ],
            *[
                (
                    'ER_AML',
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
        ]

        try_read_file = read_file(file_to_save)
        for strategy_name, strategy_builder, hyperparams in configs:
            if "strategy_n_iters" in hyperparams:
                strategy_name += f' [niters={hyperparams["strategy_n_iters"]}]'
            log.info(f'STRATEGY "{strategy_name}"')
            if try_read_file is not None and strategy_name in try_read_file:
                continue
            benchmark = split_cifar100(n_experiences=20, seed=0)
            AAA, accuracy, info = run_strategy(
                strategy_builder=strategy_builder,
                train_stream=benchmark.train_stream,
                eval_stream=benchmark.test_stream,
                model_builder=ResNet18,
                hyperparams=hyperparams,
                n_classes=benchmark.n_classes,
                num_workers=0,
                device=device
            )
            results = dict(
                AAA=AAA,
                accuracy=accuracy,
                hyperparams=json.dumps(hyperparams),
                info=info
            )
            save_record_in_file(file_to_save, strategy_name, results)


niters_generator = NItersGenerator()
