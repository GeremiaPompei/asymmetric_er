from avalanche.training import Naive, GDumb
from avalanche.training import ER_ACE as ER_ACE_AVALANCHE

from src.benchmark import split_cifar100
from src.er_ace import ER_ACE
from src.er_aml import ER_AML
from src.generators.generators import Generator
from src.gridsearch import gridsearch
from src.model import ResNet18
from src.utils import log
from src.utils.select_device import select_device


class PaperGenerator(Generator):
    """
    Paper generator class able to provide a way to do the paper analysis and store results in a json file. In this
    analysis are run different strategies: Naive, GDumb, ER_ACE_AVALANCHE, ER_AML and ER_ACE.
    """

    def __init__(self):
        super(PaperGenerator, self).__init__('PAPER')

    def __call__(self, file_to_save=None):
        if file_to_save is None:
            file_to_save = 'data/paper_results.json'
        device = select_device()

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
                file_to_save=file_to_save,
                name=strategy_name,
            )


paper_generator = PaperGenerator()
