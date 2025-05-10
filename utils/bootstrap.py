import numpy as np
class Bootstrap:

    def __init__(self, X):
        self.X = X
        self.bootstraps = []

    def calculate_bootstrap(self, bootstraps=9000, estimator=np.mean):
        for i in range(bootstraps):
            X_sample = np.random.choice(self.X, size=len(self.X), replace=True)
            o = estimator(X_sample)
            self.bootstraps.append(o)

    def mean(self):
        mean = np.sum(self.bootstraps) / len(self.bootstraps)
        return mean

    def std(self):
        mean = self.mean()
        sum_of_diff = 0
        for xi in self.bootstraps:
            sum_of_diff += (xi-mean) * (xi-mean)
        stand = sum_of_diff/ (len(self.bootstraps) - 1)
        stand = np.sqrt(stand)
        return stand
    