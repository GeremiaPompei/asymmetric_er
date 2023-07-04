import torch


def __empty__() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function able to return state of buffer.
    @return: Empty state of buffer composed by three empty tensors..
    """
    return torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0,))


class BalancedReservoirSampling:
    """
    Balanced reservoir sampling class.
    """
    def __init__(self, mem_size: int):
        """
        Balanced reservoir sampling constructor.
        @param mem_size: Maximum memory size splitted in equal part for each class.
        """
        self.mem_size = mem_size
        self.__buffer_x_class = {}
        self.iteration = 0

    @property
    def n_classes(self) -> int:
        """
        Property able to return the number of classes inside the buffer.
        @return: Number of classes inside the buffer.
        """
        return len(self.__buffer_x_class)

    @property
    def size_x_class(self) -> int:
        """
        Maximum memory size for each class.
        @return: Maximum size of each class.
        """
        return self.mem_size // self.n_classes

    def __balance(self):
        """
        Method able to balance examples keeping with equal portion items of different classes.
        """
        for k, (x, y, t) in self.__buffer_x_class.items():
            self.__buffer_x_class[k] = x[:self.size_x_class], y[:self.size_x_class], t[:self.size_x_class]

    def update(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """
        Method able to update the buffer inserting new data.
        @param x: New inputs dats.
        @param y: New outputs data.
        @param t: New task data.
        """
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
    def buffer(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Property able to return the buffer items.
        @return: Tuple of inputs, outputs and task data.
        """
        if len(self.__buffer_x_class) == 0:
            return __empty__()
        buff = [v for v in self.__buffer_x_class.values()]
        buff = [torch.cat(v) for v in zip(*buff)]
        perm = torch.randperm(len(buff[0]))
        return tuple([b[perm] for b in buff])

    def __len__(self) -> int:
        """
        Method able to return the length of the total buffer.
        @return: Length of the buffer.
        """
        if len(self.__buffer_x_class) == 0:
            return 0
        return self.buffer[0].shape[0]

    def sample(self, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method able to sample randomly items from buffer.
        @param size: Size of items to sample.
        @return: Tuple of inputs, outputs and tasks items.
        """
        buff = self.buffer
        if buff is None:
            return __empty__()
        perm = torch.randperm(buff[0].shape[0])[:size]
        return tuple([v[perm] for v in buff])
