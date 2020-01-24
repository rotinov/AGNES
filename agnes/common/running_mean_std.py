import numpy as np
import abc


class _BaseMeanStd(object):
    @abc.abstractmethod
    def update(self, x: np.ndarray):
        pass


class RunningMeanStd(_BaseMeanStd):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon: float = 1e-4, shape=()):
        self.mean: np.ndarray = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self._shape = shape

    def update(self, x: np.ndarray):
        assert len(x.shape) == 1 + len(self._shape), "Number of dimensions should be {}, provided {}".format(
            1 + len(self._shape), len(x.shape))
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class EMeanStd(_BaseMeanStd):
    # https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation
    def __init__(self, alpha=1e-5, epsilon: float = 1e-8, shape=()):
        self.mean: np.ndarray = np.ones(shape, 'float64')
        self.var: np.ndarray = np.ones(shape, 'float64')
        self.std: np.ndarray = np.ones(shape, 'float64')
        self.alpha: float = alpha
        self.epsilon = epsilon
        self._shape = shape if isinstance(shape, tuple) else (1,)
        self.first = True

    def update(self, x: np.ndarray):
        assert len(x.shape) == 1 + len(self._shape), "Number of dimensions should be {}, provided {}".format(
            1 + len(self._shape), len(x.shape))
        batch_mean: np.ndarray = np.mean(x, axis=0, dtype=np.float64)
        if self.first:
            self.mean: np.ndarray = batch_mean
            self.var: np.ndarray = np.mean(np.square(x - self.mean), axis=0, dtype=np.float64)
            if x.shape[0] == 1:
                self.var: np.ndarray = self.mean
            else:
                self.var: np.ndarray = np.mean(np.square(x - self.mean), axis=0, dtype=np.float64)
            self.first = False
        else:
            self.mean: np.ndarray = self.alpha * batch_mean + (1 - self.alpha) * self.mean

            self.var: np.ndarray = (1 - self.alpha) * self.var + self.alpha * np.mean(np.square(x - self.mean, dtype=np.float64), axis=0, dtype=np.float64)
        self.std: np.ndarray = np.sqrt(self.var.clip(min=self.epsilon))
