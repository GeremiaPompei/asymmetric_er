import itertools
import random

import numpy as np
import torch
from avalanche.benchmarks import benchmark_with_validation_stream

from src.gridsearch.strategy_runner import run_strategy


def __fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def gridsearch(
        strategy_builder,
        benchmark_builder,
        model_builder,
        hyperparams_list,
        validation_size=0.2,
        n_experiences=20,
        num_workers=2,
        device='cpu',
        verbose=True,
        seed=0,
) -> tuple[
    dict[float, tuple[any, float]],
    tuple[float, float, dict[str, any]]
]:
    if isinstance(hyperparams_list, dict):
        hyperparams_list = [
            dict(zip(hyperparams_list.keys(), t))
            for t in itertools.product(*hyperparams_list.values())
        ]

    __fix_seed(seed)
    benchmark_test = benchmark_builder(
        seed=seed,
        n_experiences=n_experiences
    )
    n_classes = benchmark_test.n_classes
    benchmark_validation = benchmark_with_validation_stream(
        benchmark_test,
        validation_size=validation_size
    )

    validation_results = {}
    if verbose:
        print('Start of validation...')
    for i, hyperparams in enumerate(hyperparams_list):
        print(f'Hyperparams config number {i + 1}/{len(hyperparams_list)}: {hyperparams}')
        AAA, accuracy = run_strategy(
            strategy_builder=strategy_builder,
            train_stream=benchmark_validation.train_stream,
            eval_stream=benchmark_validation.valid_stream,
            model=model_builder(n_classes).to(device),
            hyperparams=hyperparams,
            num_workers=num_workers,
            device=device,
            verbose=verbose,
        )
        validation_results[AAA] = (hyperparams, accuracy)

        break  # only dev

    validation_results = dict(sorted(validation_results.items(), reverse=True))
    best_hyperparams, _ = list(validation_results.values())[0]
    if verbose:
        print(f'Best hyperparams: {best_hyperparams}')
        print('End of validation...')

    AAA, accuracy = run_strategy(
        strategy_builder=strategy_builder,
        train_stream=benchmark_test.train_stream,
        eval_stream=benchmark_test.test_stream,
        model=model_builder(n_classes).to(device),
        hyperparams=best_hyperparams,
        num_workers=num_workers,
        device=device,
        verbose=verbose,
    )
    test_results = AAA, accuracy, best_hyperparams
    if verbose:
        print(f'Test results: Accuracy={accuracy}, AAA={AAA}')

    return validation_results, test_results
