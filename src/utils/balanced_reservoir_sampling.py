import torch


def __empty__():
    return torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0,))


class BalancedReservoirSampling:
    def __init__(self, mem_size: int):
        self.mem_size = mem_size
        self.__buffer_x_class = {}
        self.iteration = 0

    @property
    def n_classes(self):
        return len(self.__buffer_x_class)

    @property
    def size_x_class(self):
        return self.mem_size // self.n_classes

    def __balance(self):
        for k, (x, y, t) in self.__buffer_x_class.items():
            self.__buffer_x_class[k] = x[:self.size_x_class], y[:self.size_x_class], t[:self.size_x_class]

    def update(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        y_unique = y.unique()
        for target in y_unique:
            mbatch = (x[y == target], y[y == target], t[y == target])
            trg = target.item()
            if trg in self.__buffer_x_class:
                buff_x, buff_y, buff_t = self.__buffer_x_class[trg]
                if buff_y.shape[0] >= self.size_x_class:
                    rand_extraction = torch.randint(0, self.iteration + mbatch[1].shape[0], mbatch[1].shape)
                    valid = rand_extraction < self.size_x_class
                    mbatch = [v[valid] for v in mbatch]
                    rand_extraction = rand_extraction[valid]
                    for buffer_v, mbatch_v in zip(self.__buffer_x_class[trg], mbatch):
                        buffer_v[rand_extraction] = mbatch_v
                else:
                    self.__buffer_x_class[trg] = [
                        torch.cat((buffer_v, mbatch_v))[:self.size_x_class] for buffer_v, mbatch_v in
                        zip(self.__buffer_x_class[trg], mbatch)
                    ]
            else:
                self.__buffer_x_class[trg] = mbatch
                self.__balance()
        self.iteration += y.shape[0]

    @property
    def buffer(self):
        if len(self.__buffer_x_class) == 0:
            return __empty__()
        buff = [v for v in self.__buffer_x_class.values()]
        buff = [torch.cat(v) for v in zip(*buff)]
        perm = torch.randperm(len(buff[0]))
        return tuple([b[perm] for b in buff])

    def __len__(self):
        if len(self.__buffer_x_class) == 0:
            return 0
        return self.buffer[0].shape[0]

    def sample(self, size):
        buff = self.buffer
        if buff is None:
            return __empty__()
        perm = torch.randperm(buff[0].shape[0])[:size]
        return tuple([v[perm] for v in buff])
