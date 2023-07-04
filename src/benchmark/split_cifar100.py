import numpy as np
from avalanche.benchmarks import SplitCIFAR100, NCScenario
from torchvision.transforms import transforms, ToTensor

from src.utils import fix_seed


def split_cifar100(n_experiences: int = 20, seed: int = 0) -> NCScenario:
    """
    Function able to load split cifar100 benchmark.
    @param n_experiences: Number of experience for splitting.
    @param seed: Seed to guarantee the replication of experiments.
    @return: Split cifar100 benchmark.
    """
    fix_seed(seed)
    fixed_class_order = np.arange(100)
    unique_transform = transforms.Compose(
        [
            ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
            ),
        ]
    )
    return SplitCIFAR100(
        seed=seed,
        n_experiences=n_experiences,
        train_transform=unique_transform,
        eval_transform=unique_transform,
        fixed_class_order=fixed_class_order,
        return_task_id=False,
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
    )
