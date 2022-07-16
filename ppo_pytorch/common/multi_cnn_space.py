from gym import Space


class MultiCNNSpace(Space):
    def __init__(self, space: Space):
        super().__init__(space.shape, space.dtype)
        self.space = space

    def sample(self):
        return self.space.sample()

    def contains(self, x):
        return self.space.contains(x)