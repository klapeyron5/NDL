import numpy as np
from data.get_dataset import get_dataset, PROBLEM_CLASSIFICATION, PROBLEM_DENOISING
from data.data_pipe import Pipe


class Data_manager:
    def __init__(self, data_path, problem=PROBLEM_CLASSIFICATION,
                 preproc_trn_X=None, preproc_trn_Y=None, preproc_val_X=None, preproc_val_Y=None,):

        self.problem = problem
        self.val_X, self.val_Y, self.trn_X, self.trn_Y = get_dataset(data_path, problem)

        self.set_preprocessings(preproc_trn_X, preproc_trn_Y, preproc_val_X, preproc_val_Y,)

    def set_preprocessings(self, preproc_trn_X=None, preproc_trn_Y=None, preproc_val_X=None, preproc_val_Y=None,):
        preprocesses = [preproc_trn_X, preproc_trn_Y, preproc_val_X, preproc_val_Y]
        preprocesses = [Pipe([]) if x is None else x for x in preprocesses]
        assert all([isinstance(x, Pipe) for x in preprocesses])
        self.preproc_trn_X, self.preproc_trn_Y, self.preproc_val_X, self.preproc_val_Y = preprocesses

    def get_next_trn_batch(self, batch_size=1):
        """
        yield batch_X, batch_Y, batch_number
        """
        self.p = 0
        self.b = 0
        L = len(self.trn_X)
        ids = np.random.permutation(L)
        last_p = self.p + batch_size
        while last_p <= L:
            batch_X_ = self.trn_X[ids[self.p:last_p]]
            batch_Y_ = self.trn_Y[ids[self.p:last_p]]
            batch_X = []
            batch_Y = []
            for x, y in zip(batch_X_, batch_Y_):
                x = self.preproc_trn_X(x)
                y = self.preproc_trn_Y(y)
                batch_X.append(x)
                batch_Y.append(y)
            batch_X = np.array(batch_X)
            batch_Y = np.array(batch_Y)
            self.b += 1
            self.p = last_p
            last_p = self.p + batch_size

            assert batch_X.shape[0] == batch_Y.shape[0]
            yield batch_X, batch_Y, self.b

    def get_next_trn_pair_classification(self):
        """
        yield batch_X, batch_Y, batch_number
        batch_size = 2
        """
        assert self.problem == PROBLEM_CLASSIFICATION
        self.b = 0
        L = len(self.trn_X)
        ids = list(range(0, L, 2))
        ids = np.random.permutation(ids)
        for id in ids:
            batch_X_ = self.trn_X[[id, id+1]]
            batch_Y_ = self.trn_Y[[id, id+1]]
            batch_X = []
            batch_Y = []
            for x, y in zip(batch_X_, batch_Y_):
                x = self.preproc_trn_X(x)
                y = self.preproc_trn_Y(y)
                batch_X.append(x)
                batch_Y.append(y)
            batch_X = np.array(batch_X)
            batch_Y = np.array(batch_Y)
            self.b += 1

            assert batch_X.shape[0] == batch_Y.shape[0]
            yield batch_X, batch_Y, self.b

    def get_next_val_pair_classification(self, part=1.0):
        """
        yield batch_X, batch_Y
        batch_size = 2
        """
        assert self.problem == PROBLEM_CLASSIFICATION
        L = len(self.val_X)
        ids = list(range(0, L, 2))
        if 0 < part < 1:
            part = int(round(len(ids)*part))
            ids = np.random.permutation(ids)[:part]
        elif part == 1:
            pass
        else:
            errmsg = "part is {}, but should be 0 < part <= 1".format(part)
            raise Exception(errmsg)

        for id in ids:
            batch_X_ = self.val_X[[id, id + 1]]
            batch_Y_ = self.val_Y[[id, id + 1]]
            batch_X = []
            batch_Y = []
            for x, y in zip(batch_X_, batch_Y_):
                x = self.preproc_val_X(x)
                y = self.preproc_val_Y(y)
                batch_X.append(x)
                batch_Y.append(y)
            batch_X = np.array(batch_X)
            batch_Y = np.array(batch_Y)

            assert batch_X.shape[0] == batch_Y.shape[0]
            yield batch_X, batch_Y

    def get_next_val_batch(self, batch_size=1, part=1.0):
        """
        yield batch_X, batch_Y
        """
        L = len(self.val_X)

        ids = np.random.permutation(L)
        L = int(round(L*part))
        ids = ids[:L]
        if batch_size == -1:
            batch_size = L

        def get_data():
            batch_X_ = self.val_X[ids[p:last_p]]
            batch_Y_ = self.val_Y[ids[p:last_p]]
            batch_X = []
            batch_Y = []
            for x, y in zip(batch_X_, batch_Y_):
                x = self.preproc_val_X(x)
                y = self.preproc_val_Y(y)
                batch_X.append(x)
                batch_Y.append(y)
            batch_X = np.array(batch_X)
            batch_Y = np.array(batch_Y)
            return batch_X, batch_Y

        p = 0
        last_p = p + batch_size
        while last_p < L:
            yield get_data()
            p = last_p
            last_p += batch_size

        for i in range(1):
            last_p = None
            yield get_data()

    def get_next_trn_pair_denoising(self):
        """
        yield batch_X, batch_Y, batch_number
        batch_size = 2
        """
        bs = 1
        self.p = 0
        self.b = 0
        L = len(self.trn_X)
        ids = np.random.permutation(L)
        last_p = self.p+bs
        while last_p <= L:
            batch_X_ = self.trn_X[ids[self.p:last_p]]
            batch_Y_ = self.trn_Y[ids[self.p:last_p]]
            batch_X = []
            batch_Y = []
            for x, y in zip(batch_X_, batch_Y_):
                x = self.preproc_trn_X(x)
                y = self.preproc_trn_Y(y)
                batch_X.append(x)
                batch_Y.append(y)
            batch_X = np.array(batch_X + batch_Y)
            batch_Y = np.array(batch_Y + batch_Y)
            self.b += 1
            self.p = last_p
            last_p = self.p+bs

            assert batch_X.shape[0] == batch_Y.shape[0]
            yield batch_X, batch_Y, self.b
