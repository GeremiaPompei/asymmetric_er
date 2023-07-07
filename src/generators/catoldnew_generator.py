import json

from src.benchmark import split_cifar100
from src.er_ace import ER_ACE
from src.er_aml import ER_AML
from src.generators.generators import Generator
from src.gridsearch.strategy_runner import run_strategy
from src.model import ResNet18
from src.utils import log
from src.utils.batch_norm_tracker import BatchNormTracker
from src.utils.fs import save_record_in_file, read_file

from src.utils.select_device import select_device


class CatOldNewGenerator(Generator):
    """
    CatOldNew generator class able to train ER_ACE and ER_AML strategies with different cat_old_new hyperparam.
    """

    def __init__(self):
        super(CatOldNewGenerator, self).__init__('CAT_OLD_NEW')

    def __call__(self, file_to_save=None):
        if file_to_save is None:
            file_to_save = 'data/catoldnew_results.json'
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
                        strategy_cat_old_new=i,
                        sgd_lr=0.1,
                        sgd_momentum=0
                    )
                ) for i in [False, True]
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
                        strategy_cat_old_new=i,
                        sgd_lr=0.1,
                        sgd_momentum=0
                    )
                ) for i in [False, True]
            ],
        ]

        try_read_file = read_file(file_to_save)
        for strategy_name, strategy_builder, hyperparams in configs:
            if "strategy_cat_old_new" in hyperparams:
                strategy_name += f' [cat_old_new={hyperparams["strategy_cat_old_new"]}]'
            log.info(f'STRATEGY "{strategy_name}"')
            if try_read_file is not None and strategy_name in try_read_file:
                continue
            benchmark = split_cifar100(n_experiences=20, seed=0)
            bn_tracker = BatchNormTracker()
            AAA, accuracy, info = run_strategy(
                strategy_builder=strategy_builder,
                train_stream=benchmark.train_stream,
                eval_stream=benchmark.test_stream,
                model_builder=ResNet18,
                hyperparams=hyperparams,
                n_classes=benchmark.n_classes,
                num_workers=0,
                device=device,
                plugins=[bn_tracker]
            )
            results = dict(
                AAA=AAA,
                accuracy=accuracy,
                hyperparams=json.dumps(hyperparams),
                info=info,
                bn_info=bn_tracker.result()
            )
            save_record_in_file(file_to_save, strategy_name, results)


catoldnew_generator = CatOldNewGenerator()
