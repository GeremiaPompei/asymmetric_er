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


def __read_record_in_file(file_to_save, strategy_key):
    if file_to_save is not None:
        if os.path.exists(file_to_save):
            with open(file_to_save) as file:
                results = json.load(file)
                if strategy_key in results:
                    return results[strategy_key]


def __save_record_in_file(file_to_save, strategy_key, strategy_value):
    results = __read_record_in_file(file_to_save)
    results[strategy_key] = strategy_value
    with open(file_to_save, 'w') as file:
        json.dump(results, file)


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
) -> dict:
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

    try_to_read = __read_record_in_file(file_to_save, strategy_builder.__class__)
    results = dict(
        validation={},
        test=None,
    ) if try_to_read is None else try_to_read

    if verbose:
        print('Start of validation...')
    for i, hyperparams in enumerate(hyperparams_list):
        print(f'Hyperparams config number {i + 1}/{len(hyperparams_list)}: {hyperparams}')

        if hyperparams in results['validation']:
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

        results['validation'][hyperparams] = (AAA, accuracy)
        __save_record_in_file(file_to_save, strategy_builder.__class__, results)

    best_hyperparams = max({v[0]: k for k, v in results['validation'].items()}.items())[1]

    if verbose:
        print(f'Best hyperparams: {best_hyperparams}')
        print('End of validation...')

    if results['test'] is None:
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
        results['test'] = (AAA, accuracy, best_hyperparams)
        __save_record_in_file(file_to_save, strategy_builder.__class__, results)

    AAA, accuracy, best_hyperparams = results['test']

    if verbose:
        print(f'Test results: Accuracy={accuracy}, AAA={AAA}')

    return results
