from typing import Callable

from avalanche.benchmarks import NCStream
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm


def __extract_hyperparams(general_hyperparams: dict, key: str) -> dict:
    """
    Function able to extract hyperparams with a certain prefix.
    @param general_hyperparams: General hyperparams dictonary.
    @param key: Prefix related to the key of hyperparams.
    @return: Hyperparams related to the selected key.
    """
    prefix = f'{key}_'
    return {k.replace(prefix, ''): v for k, v in general_hyperparams.items() if k.startswith(prefix)}


def run_strategy(
        strategy_builder: Callable,
        train_stream: NCStream,
        eval_stream: NCStream,
        model_builder: Callable,
        hyperparams: dict,
        n_classes: int,
        num_workers: int = 2,
        device: str = 'cpu',
        verbose: bool = True,
        plugins: list = [],
        loggers: list = [],
) -> tuple[float, float, list]:
    """
    Function able to run a certain strategy.
    @param strategy_builder: Function able to build a strategy.
    @param train_stream: Training stream of experiences.
    @param eval_stream: Evaluation stream of experiences.
    @param model_builder: Function able to create a model.
    @param hyperparams: Hyperparams dictionary.
    @param n_classes: Number of classes.
    @param num_workers: Number of workers used in training and evaluation phase.
    @param device: Accelerator type.
    @param verbose: Flag of verbose logging.
    @param plugins: List of additional plugins.
    @param loggers: List of additional loggers.
    @return: Tuple of AAA and accuracy performances and results of metrics for each experience.
    """
    sgd_params = __extract_hyperparams(hyperparams, 'sgd')
    strategy_params = __extract_hyperparams(hyperparams, 'strategy')
    model_params = __extract_hyperparams(hyperparams, 'model')
    model = model_builder(n_classes, **model_params).to(device)
    cl_strategy = strategy_builder(
        model,
        SGD(model.parameters(), **sgd_params),
        CrossEntropyLoss(),
        evaluator=EvaluationPlugin(
            accuracy_metrics(experience=True),
            *plugins,
            loggers=loggers,
        ),
        device=device,
        **strategy_params,
    )

    anytime_accuracies = []
    AAA, anytime_accuracy = None, None
    results = []
    pbar = tqdm(train_stream)
    for experience in pbar:
        model.train()
        cl_strategy.train(experience, num_workers=num_workers, drop_last=True)
        model.eval()
        res = cl_strategy.eval(eval_stream[:experience.current_experience + 1], num_workers=num_workers)
        results.append(res)
        accuracies = [v for k, v in res.items() if 'Top1_Acc_Exp' in k]
        anytime_accuracy = sum(accuracies) / len(accuracies)
        anytime_accuracies.append(anytime_accuracy)
        AAA = sum(anytime_accuracies) / len(anytime_accuracies)
        if verbose:
            pbar.set_description(f'AAA={AAA}, Accuracy={anytime_accuracy}')

    return AAA, anytime_accuracy, results
