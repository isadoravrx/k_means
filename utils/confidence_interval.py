import numpy as np
from scipy.stats import norm
from utils.bootstrap import Bootstrap

class ConfidenceInterval:
  def __init__(self, data, alpha) -> None:
    bootstrap = Bootstrap(data)
    bootstrap.calculate_bootstrap(bootstraps=9000, estimator=np.mean)
    self.mean = bootstrap.mean()
    self.stand = bootstrap.std()
    self.z = norm.ppf(1 - (alpha/2))

  def calculate_lower_bound(self):
    return self.mean - (self.z * self.stand)

  def calculate_upper_bound(self):
    return self.mean + (self.z * self.stand)