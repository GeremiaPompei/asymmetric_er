from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm


def __extract_hyperparams(general_hyperparams, key):
    prefix = f'{key}_'
    return {k.replace(prefix, ''): v for k, v in general_hyperparams.items() if k.startswith(prefix)}


def run_strategy(
        strategy_builder,
        train_stream,
        eval_stream,
        model_builder,
        hyperparams,
        n_classes,
        num_workers=2,
        device='cpu',
        verbose=True,
        loggers=[]
) -> tuple[float, float, list]:
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
