import torch
import torch.nn.functional as F


class ACECriterion:

    def __init__(self):
        self.seen_so_far = set()

    def __call__(
            self,
            output_in,
            target_in,
            output_buffer,
            target_buffer
    ):
        loss_buffer = F.cross_entropy(output_buffer, target_buffer)
        mask = torch.ones_like(output_in)
        self.seen_so_far = {*list(self.seen_so_far), *target_in.unique().tolist()}
        mask[:, list(self.seen_so_far)] = 0
        mask[:, target_in.unique().tolist()] = 1
        loss_in = F.cross_entropy(output_in * mask, target_in)
        return loss_in + loss_buffer
