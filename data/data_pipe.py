
class Pipe:
    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, x):
        for func in self.funcs:
            x = func(self, x)
        return x


import numpy as np


class ProcessData(Pipe):
    def np_load(self, x):
        return np.load(x)


class ProcessClassLabel(Pipe):
    def __init__(self, funcs):
        super(ProcessClassLabel, self).__init__(funcs)
        self.one_hot_enc = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.float16)

    def to_vector(self, x):
        return self.one_hot_enc[x]
