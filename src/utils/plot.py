import json
import math

import matplotlib.pyplot as plt
import pandas as pd


def extract_accuracy_table(history: dict) -> dict:
    return {
        k: [
            [
                x[f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{str(i).zfill(3)}'] if i < len(x) else 0
                for i in range(len(v['info']))
            ]
            for x in v['info']
        ]
        for k, v in history.items()
    }


def plot_general_results(history: dict):
    to_perc = lambda x: f'{round(x * 100, 2)} %'
    return pd.DataFrame({
        k: dict(
            AAA=to_perc(v['AAA']),
            accuracy=to_perc(v['accuracy']),
            **json.loads(v['hyperparams'])
        )
        for k, v in history.items()
    })


def plot_over_experiences(history: dict, xlabel: str, ylabel: str):
    plt.figure(figsize=(20, 10))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_forgetting(history: dict):
    accuracy_table = extract_accuracy_table(history)
    forgetting = {}

    for k, acc_table in accuracy_table.items():
        f = [acc_table[i][i] - acc_table[-1][i] for i in range(len(acc_table))]
        forgetting[k] = sum(f) / len(f)

    plt.figure(figsize=(20, 10))
    plt.bar(*zip(*forgetting.items()))
    plt.grid(True)
    plt.show()


def plot_accuracy_tables(history: dict):
    accuracy_table = extract_accuracy_table(history)
    for name, data in accuracy_table.items():
        plt.figure(figsize=(20, 10))
        plt.title(name)
        plt.imshow(data)
        for i in range(len(data)):
            for j in range(len(data[i])):
                plt.annotate(round(data[i][j], 2), xy=(j, i), ha='center', va='center')
        plt.colorbar()
        plt.xlabel('Experiences')
        plt.ylabel('Time')
        plt.show()


def plot_bn_over_epochs(results: dict, layers: list[str], bn_feature: str):
    layer_info = {k: v['bn_tracker'] for k, v in results.items()}
    y = 2
    x = math.ceil(len(layers) / y)
    fig, ax = plt.subplots(x, y, figsize=(20, 7 * x))
    for i in range(x * y):
        axis = ax[i // y, i % y]
        if i >= len(layers):
            axis.remove()
        else:
            layer = layers[i]
            axis.set_title(f'{layer} bn layer')
            for str_name, bn_info in layer_info.items():
                axis.plot([v[layer][bn_feature] for v in bn_info], label=str_name)
            axis.set_xlabel('epochs')
            axis.set_ylabel(f'{bn_feature} norm')
            axis.grid(True)
            axis.legend()
    plt.show()
