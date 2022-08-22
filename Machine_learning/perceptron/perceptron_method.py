import numpy as np


class Perceptron:
    """eta: float
    n_iter: int
    random_state: int
    w_ : 1d-array
    b_ : Scalar
    errors : list
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter=n_iter
        self.random_state=random_state

    def fit(self, x, y):
        """
        :param x: array
        :param y: array
        :return: self object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=x.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors=0
            for xi, target in zip(x,y):
                update = self.eta*(target-self.predict(xi))
                self.w_ += update*xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        return np.dot(x, self.w_) + self.b_

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)



