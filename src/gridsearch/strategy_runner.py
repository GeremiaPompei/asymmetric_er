from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import SGD


def run_strategy(
        strategy_builder,
        train_stream,
        eval_stream,
        model,
        hyperparams=dict(
            train_mb_size=10,
            eval_mb_size=10,
            train_epochs=1,
            mem_size=200,
            batch_size_mem=10,
            lr=0.1,
            momentum=0.9
        ),
        num_workers=2,
        device='cpu',
        verbose=True,
) -> tuple[float, float]:
    sgd_params = {k.replace('sgd_', ''): v for k, v in hyperparams.items() if k.startswith('sgd_')}
    strategy_params = {k.replace('strategy_', ''): v for k, v in hyperparams.items() if k.startswith('strategy_')}
    cl_strategy = strategy_builder(
        model,
        SGD(model.parameters(), **sgd_params),
        CrossEntropyLoss(),
        evaluator=EvaluationPlugin(accuracy_metrics(experience=True), loggers=InteractiveLogger() if verbose else None),
        device=device,
        **strategy_params,
    )

    anytime_accuracy = []
    for experience in train_stream:
        if verbose:
            print(f'Experience {experience.current_experience + 1}/{len(train_stream)}')
        cl_strategy.train(experience, num_workers=num_workers)
        res = cl_strategy.eval(eval_stream, num_workers=num_workers)
        accuracy_res = {int(k[-3:]): v for k, v in res.items() if 'Top1_Acc_Exp' in k}
        accuracies = [accuracy_res[e] for e in range(experience.current_experience + 1)]
        anytime_accuracy.append(sum(accuracies) / len(accuracies))
        AAA = sum(anytime_accuracy) / len(anytime_accuracy)
        if verbose:
            print(f'AAA: {AAA}')

    return AAA, anytime_accuracy[-1]
