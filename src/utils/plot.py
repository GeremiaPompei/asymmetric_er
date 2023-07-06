import json
import math

import matplotlib.pyplot as plt
import pandas as pd


def extract_accuracy_table(history: dict) -> dict:
    """
    Function able to extract from a dictionary of (strategy_name, strategy_results) a dictionary of (strategy_name,
    accuracy_table). The accuracy table is a square matrix (list of list) composed by accuracies of each experience
    computed each time that a new experience is learned.
    @param history: History of strategies results.
    @return: Dictionary of accuracy tables.
    """
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


def plot_general_results(history: dict) -> pd.DataFrame:
    """
    Function able to extract a dataframe with general information related to strategies results.
    @param history: History of strategies results.
    @return: Dataframe with general info.
    """
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
    """
    Function able to plot data over experiences.
    @param history: Dictionary of data to plot.
    @param xlabel: Name of x label.
    @param ylabel: Name of y label.
    """
    plt.figure(figsize=(20, 10))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_forgetting(history: dict):
    """
    Function able to plot an histogram of forgetting for each experience..
    @param history: Dictionary of data to use to compute the forgetting.
    """
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
    """
    Function able to plot accuracy tables for each strategy.
    @param history: Dictionary of data to use to extract accuracy table.
    """
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


def plot_bn_over_epochs(history: dict, n_layer: int, initial_offset: int = 0):
    """
    Function able to plot a batch normalization feature over epochs.
    @param history: Dictionary of results where extract dn features.
    @param n_layer: Index of bn layer to plot.
    @param initial_offset: Parameter able to discard previous noisy values.
    """
    new_data = {k: v['bn_tracker']['new'] for k, v in history.items()}
    buffer_data = {k: v['bn_tracker']['buffer'] for k, v in history.items()}
    fig, ax = plt.subplots(len(new_data), 2, figsize=(20, 7 * len(new_data)))
    for row, strategy_name in enumerate(new_data):
        for i_feature, bn_feature in enumerate(['mean', 'std']):
            axis = ax[row, i_feature] if len(new_data) > 1 else ax[i_feature]
            if row >= len(new_data):
                axis.remove()
            else:
                axis.set_title(f'{strategy_name} - {n_layer} bn layer - {bn_feature}')
                axis.plot([v[n_layer][i_feature] for v in new_data[strategy_name]][initial_offset:], label=f'new data')
                axis.plot([v[n_layer][i_feature] for v in buffer_data[strategy_name]][initial_offset:], label=f'buffer data')
                axis.set_xlabel('iterations')
                axis.set_ylabel(bn_feature)
                axis.grid(True)
                axis.legend()
    plt.show()
