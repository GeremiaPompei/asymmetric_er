from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm


def run_strategy(
        strategy_builder,
        train_stream,
        eval_stream,
        model,
        hyperparams,
        num_workers=2,
        device='cpu',
        verbose=True,
) -> tuple[float, float, list]:
    sgd_params = {k.replace('sgd_', ''): v for k, v in hyperparams.items() if k.startswith('sgd_')}
    strategy_params = {k.replace('strategy_', ''): v for k, v in hyperparams.items() if k.startswith('strategy_')}
    cl_strategy = strategy_builder(
        model,
        SGD(model.parameters(), **sgd_params),
        CrossEntropyLoss(),
        evaluator=EvaluationPlugin(accuracy_metrics(experience=True)),
        device=device,
        **strategy_params,
    )

    AAA = 0
    results = []
    pbar = tqdm(train_stream)
    for experience in pbar:
        cl_strategy.train(experience, num_workers=num_workers)
        res = cl_strategy.eval(eval_stream, num_workers=num_workers)
        results.append(res)
        accuracy_res = {int(k[-3:]): v for k, v in res.items() if 'Top1_Acc_Exp' in k}
        accuracies = [accuracy_res[e] for e in range(experience.current_experience + 1)]
        AAA = sum(accuracies) / len(accuracies)
        if verbose:
            pbar.set_description(f'AAA={AAA}')

    return AAA, accuracies[-1], results
