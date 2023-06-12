import torch
import torch.nn.functional as F

from src.model.features_map import FeaturesMapModel


class AMLCriterion:

    def __init__(self, model: FeaturesMapModel, device='cpu'):
        self.device = device
        self.model = model
        self.pos_h = None
        self.pos_y = None
        self.neg_h = None
        self.neg_y = None

    def __compute_pos_neg(self, x_in, y_in, x_buffer, y_buffer):
        x_all = torch.cat((x_buffer, x_in))
        y_all = torch.cat((y_buffer, y_in))
        indexes = torch.arange(y_all.shape[0]).to(self.device)

        same_x = indexes[-x_in.shape[0]:].reshape(1, -1) == indexes.reshape(-1, 1)
        same_y = y_in.reshape(1, -1) == y_all.reshape(-1, 1)

        valid_pos = same_y & ~same_x
        valid_neg = ~same_y

        pos_idx = torch.multinomial(valid_pos.float().T, 1).squeeze(1)
        neg_idx = torch.multinomial(valid_neg.float().T, 1).squeeze(1)

        pos_h = self.model.return_hidden(x_all[pos_idx])
        pos_y = y_all[pos_idx]
        neg_h = self.model.return_hidden(x_all[neg_idx])
        neg_y = y_all[neg_idx]

        return (pos_h, pos_y), (neg_h, neg_y)

    def __compute_loss_in(self, hidden_in, pos, neg):
        (pos_h, pos_y), (neg_h, neg_y) = pos, neg
        loss = (
                torch.exp(F.cosine_similarity(hidden_in, pos_h)) /
                torch.exp(F.cosine_similarity(hidden_in, neg_h)).sum()
        ).log().mean()
        return -loss

    def __call__(
            self,
            input_in,
            target_in,
            output_buffer,
            target_buffer,
            buffer: list
    ):
        x_buffer, y_buffer, _ = zip(*buffer)
        xx_buffer = torch.stack(x_buffer)
        y_buffer = torch.Tensor(y_buffer)
        pos, neg = self.__compute_pos_neg(input_in, target_in, xx_buffer, y_buffer)
        loss_buffer = F.cross_entropy(output_buffer, target_buffer)
        hidden_in = self.model.return_hidden(input_in)
        loss_in = self.__compute_loss_in(hidden_in, pos, neg)
        return loss_in + loss_buffer
