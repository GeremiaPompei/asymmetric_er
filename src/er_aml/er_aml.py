from typing import Callable, List, Optional, Union

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.core import SupervisedPlugin
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.evaluation import (
    EvaluationPlugin,
    default_evaluator,
)
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate

from src.er_aml.er_aml_criterion import AMLCriterion
from src.model.features_map import FeaturesMapModel


class ER_AML(SupervisedTemplate):

    def __init__(
            self,
            model: Union[Module, FeaturesMapModel],
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(),
            temp: float = 0.1,
            base_temp: float = 0.07,
            n_iters: int = 1,
            mem_size: int = 200,
            batch_size_mem: int = 10,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = 1,
            device: Union[str, torch.device] = "cpu",
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: Union[
                EvaluationPlugin,
                Callable[[], EvaluationPlugin]
            ] = default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.aml_criterion = AMLCriterion(model=model, temp=temp, base_temp=base_temp, device=device)
        self.n_iters = n_iters

        self.mb_buffer_x = None
        self.mb_buffer_y = None
        self.mb_buffer_tid = None
        self.mb_buffer_out = None

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            for i in range(self.n_iters):
                available_buffer = len(self.storage_policy.buffer) >= self.batch_size_mem
                if available_buffer:
                    batch = next(
                        iter(
                            torch.utils.data.DataLoader(
                                self.storage_policy.buffer,
                                batch_size=self.batch_size_mem,
                                shuffle=True,
                                drop_last=True,
                            )
                        )
                    )
                    self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = (v.to(self.device) for v in batch)

                self.optimizer.zero_grad()
                self.loss = self._make_empty_loss()

                # Forward
                self._before_forward(**kwargs)
                self.mb_output = avalanche_forward(
                    self.model, self.mb_x, self.mb_task_id
                )
                if available_buffer:
                    self.mb_buffer_out = avalanche_forward(
                        self.model, self.mb_buffer_x, self.mb_buffer_tid
                    )
                self._after_forward(**kwargs)

                # Loss & Backward
                if not available_buffer or self.experience.current_experience == 0:
                    self.loss += self.criterion()
                else:
                    self.loss += self.aml_criterion(
                        self.mb_x,
                        self.mb_y,
                        self.mb_buffer_out,
                        self.mb_buffer_y,
                        list(self.storage_policy.buffer),
                    )

                self._before_backward(**kwargs)
                self.backward()
                self._after_backward(**kwargs)

                # Optimization step
                self._before_update(**kwargs)
                self.optimizer_step()
                self._after_update(**kwargs)

            self.storage_policy.update(self, **kwargs)
            self._after_training_iteration(**kwargs)
