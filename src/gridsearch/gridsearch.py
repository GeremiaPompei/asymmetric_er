import itertools
import json
import os.path
from ctypes import Union
from typing import Callable

from avalanche.benchmarks import benchmark_with_validation_stream

from src.gridsearch.strategy_runner import run_strategy
from src.utils import log
from src.utils.fs import save_record_in_file, read_file


def gridsearch(
        strategy_builder: Callable,
        benchmark_builder: Callable,
        model_builder: Callable,
        hyperparams_list: Union[list, dict],
        validation_size: float = 0.2,
        n_experiences: int = 20,
        num_workers: int = 2,
        device: str = 'cpu',
        verbose: bool = True,
        seed: int = 0,
        file_to_save: Union[None, str] = None,
        name: Union[None, str] = None,
        plugins: list = [],
) -> dict:
    """
    Gridsearch function able to retrain also the model on test set.
    @param strategy_builder: Function able to construct a strategy.
    @param benchmark_builder: Function able to construct a benchmark.
    @param model_builder: Function able to construct a model.
    @param hyperparams_list: Hyperparams list of dict or dict of list to use in validation phase for model selection of best ones.
    @param validation_size: Validation set percentage size.
    @param n_experiences: Number of experiences for benchmark streams.
    @param num_workers: Number of workers for train and validation phase.
    @param device: Accelerator type.
    @param verbose: Flag able to switch on and of the verbose printing of logs.
    @param seed: Seed value used to replicate tests.
    @param file_to_save: Name of file to save results. If is None results are not saved.
    @param name: Name of strategy used to save results in the file.
    @param plugins: Plugins used during learning phase to monitor it.
    @return: Gridsearch results of validation and test phase. In particular there are values of AAA and accuracy metrics,
    hyperparams and metrics saved for each experiences.
    """
    strategy_name = strategy_builder.__name__ if name is None else name
    if isinstance(hyperparams_list, dict):
        hyperparams_list = [
            dict(zip(hyperparams_list.keys(), t))
            for t in itertools.product(*hyperparams_list.values())
        ]

    benchmark_test = benchmark_builder(n_experiences=n_experiences, seed=seed)
    n_classes = benchmark_test.n_classes
    benchmark_validation = benchmark_with_validation_stream(
        benchmark_test,
        validation_size=validation_size
    )

    results = dict(
        validation={},
        test=None,
    )
    try_to_read = read_file(file_to_save)
    if try_to_read is not None and strategy_name in try_to_read:
        results = try_to_read[strategy_name]

    if len(hyperparams_list) > 1:

        if verbose:
            log.info('Start of validation...')
        for i, hyperparams in enumerate(hyperparams_list):
            log.info(f'Hyperparams config number {i + 1}/{len(hyperparams_list)}: {hyperparams}')

            stored_hyperparams_hash = [hash(k) for k in results['validation']]
            current_hash = hash(json.dumps(hyperparams))
            if current_hash in stored_hyperparams_hash:
                continue

            AAA, accuracy, info = run_strategy(
                strategy_builder=strategy_builder,
                train_stream=benchmark_validation.train_stream,
                eval_stream=benchmark_validation.valid_stream,
                model_builder=model_builder,
                hyperparams=hyperparams,
                n_classes=n_classes,
                num_workers=num_workers,
                device=device,
                verbose=verbose,
                plugins=plugins,
            )
            if verbose:
                log.info(f'AAA: {AAA}, accuracy: {accuracy}')

            results['validation'][json.dumps(hyperparams)] = dict(AAA=AAA, accuracy=accuracy, info=info)
            save_record_in_file(file_to_save, strategy_name, results)

        best_hyperparams = json.loads(max(results['validation'], key=lambda x: results['validation'][x]['AAA']))

        if verbose:
            log.info(f'Best hyperparams: {best_hyperparams}')
            log.info('End of validation...')

    else:

        best_hyperparams = hyperparams_list[0]

    if results['test'] is None:
        AAA, accuracy, info = run_strategy(
            strategy_builder=strategy_builder,
            train_stream=benchmark_test.train_stream,
            eval_stream=benchmark_test.test_stream,
            model_builder=model_builder,
            hyperparams=best_hyperparams,
            n_classes=n_classes,
            num_workers=num_workers,
            device=device,
            verbose=verbose,
            plugins=plugins,
        )
        results['test'] = dict(AAA=AAA, accuracy=accuracy, hyperparams=json.dumps(best_hyperparams), info=info)
        save_record_in_file(file_to_save, strategy_name, results)

    AAA, accuracy, best_hyperparams, _ = tuple(results['test'].values())

    if verbose:
        log.info(f'Test results: Accuracy={accuracy}, AAA={AAA}')

    return results
