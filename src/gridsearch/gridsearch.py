import itertools
import json
import os.path
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
        file_to_save=None,
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
    
    if file_to_save is not None:
        if os.path.exists(file_to_save):
            with open(file_to_save) as file:
                validation_results = json.load(file)
        
    if verbose:
        print('Start of validation...')
    for i, hyperparams in enumerate(hyperparams_list):
        print(f'Hyperparams config number {i + 1}/{len(hyperparams_list)}: {hyperparams}')

        if hyperparams in validation_results:
            continue

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
        if verbose:
            print(f'AAA: {AAA}, accuracy: {accuracy}')

        validation_results[hyperparams] = (AAA, accuracy)

        if file_to_save is not None:
            with open(file_to_save, 'w') as file:
                json.dump(validation_results, file)

    best_hyperparams = max({v[0]: k for k, v in validation_results.items()}.items())[1]
    
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
