import torch
import torch.nn.functional as F
from avalanche.training import ReservoirSamplingBuffer, ClassBalancedBuffer

from src.model.features_map import FeaturesMapModel


def normalize(x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    return x / (x_norm + 1e-05)


class AMLCriterion:

    def __init__(self, model: FeaturesMapModel, temp=0.1, base_temp=0.07, device='cpu'):
        self.device = device
        self.model = model
        self.temp = temp
        self.base_temp = base_temp

    def __compute_pos_neg(self, x_in, y_in, x_buffer, y_buffer):
        x_all = torch.cat((x_buffer, x_in))
        y_all = torch.cat((y_buffer, y_in))
        indexes = torch.arange(y_all.shape[0]).to(self.device)

        same_x = indexes[-x_in.shape[0]:].reshape(1, -1) == indexes.reshape(-1, 1)
        same_y = y_in.reshape(1, -1) == y_all.reshape(-1, 1)

        valid_pos = same_y & ~same_x
        valid_neg = ~same_y

        has_valid_pos = valid_pos.sum(0) > 0
        has_valid_neg = valid_neg.sum(0) > 0
        invalid_idx = ~has_valid_pos | ~has_valid_neg
        is_invalid = torch.zeros_like(y_in).bool()
        is_invalid[invalid_idx] = 1
        if invalid_idx.sum() > 0:
            # avoid operand fail
            valid_pos[:, invalid_idx] = 1
            valid_neg[:, invalid_idx] = 1

        pos_idx = torch.multinomial(valid_pos.float().T, 1).squeeze(1)
        neg_idx = torch.multinomial(valid_neg.float().T, 1).squeeze(1)

        pos_x = x_all[pos_idx]
        pos_y = y_all[pos_idx]
        neg_x = x_all[neg_idx]
        neg_y = y_all[neg_idx]

        return (pos_x, pos_y), (neg_x, neg_y), is_invalid

    def __sup_con_loss(self, anchor_features, features, anchor_targets, targets):
        pos_mask = (anchor_targets.reshape(-1, 1) == targets.reshape(1, -1)).float().to(self.device)
        similarity = anchor_features @ features.T / self.temp
        similarity -= similarity.max(dim=1)[0].detach()
        log_prob = similarity - torch.log(torch.exp(similarity).sum(1))
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        loss = - (self.temp / self.base_temp) * mean_log_prob_pos.mean()
        return loss

    def __call__(
            self,
            input_in,
            target_in,
            output_buffer,
            target_buffer,
            reservoir_sampling_data
    ):
        x_buffer, y_buffer, _ = reservoir_sampling_data
        (pos_x, pos_y), (neg_x, neg_y), is_invalid = self.__compute_pos_neg(input_in, target_in, x_buffer, y_buffer)
        loss_buffer = F.cross_entropy(output_buffer, target_buffer)
        hidden_in = normalize(self.model.return_hidden(input_in)[~is_invalid])

        hidden_pos_neg = normalize(self.model.return_hidden(torch.cat((pos_x, neg_x))))
        pos_h, neg_h = hidden_pos_neg.reshape(2, pos_x.shape[0], -1)[:, ~is_invalid]

        loss_in = self.__sup_con_loss(
            anchor_features=hidden_in.repeat(2, 1),
            features=torch.cat((pos_h, neg_h)),
            anchor_targets=target_in[~is_invalid].repeat(2),
            targets=torch.cat((pos_y[~is_invalid], neg_y[~is_invalid])),
        )
        return loss_in + loss_buffer
