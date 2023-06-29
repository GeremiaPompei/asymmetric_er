from src.gridsearch.strategy_runner import run_strategy
from src.model import ResNet18
from src.benchmark import split_cifar100
from src.er_ace import ER_ACE
from src.er_aml import ER_AML
from avalanche.training.supervised.er_ace import ER_ACE as ER_ACE_AVALANCHE

benchmark = split_cifar100(n_experiences=20, seed=0)
run_strategy(
    strategy_builder=ER_AML,
    train_stream=benchmark.train_stream,
    eval_stream=benchmark.test_stream,
    model_builder=ResNet18,
    hyperparams=dict(
        strategy_train_mb_size=10,
        strategy_eval_mb_size=10,
        strategy_train_epochs=1,
        strategy_mem_size=100 * 100,  # mem_size * num_classes
        strategy_batch_size_mem=10,
        model_dist_linear=True,
        sgd_lr=0.1,
        sgd_momentum=0
    ),
    n_classes=benchmark.n_classes,
    num_workers=0,
    verbose=True
)