from torch.optim.lr_scheduler import _LRScheduler
from agnes.algos.base import _BaseAlgo


class LinearAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, eta_min=0.0, to_epoch=1000):
        self.eta_min = eta_min
        self.to_epoch = to_epoch
        last_epoch = -1
        super(LinearAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + max(0, (base_lr - self.eta_min) * (1 - self._step_count / self.to_epoch))
                for base_lr in self.base_lrs]

    def get_count(self):
        return self._step_count


class LinearSchedule:
    _step_count = 0

    def __init__(self, val_fun, eta_min=0.0, to_epoch=1000):
        self.eta_min = eta_min
        self.to_epoch = to_epoch
        self.val_fun = val_fun

    def step(self):
        self._step_count += 1

    def get_v(self):
        return self.val_fun(self._get_k())

    def _get_k(self):
        return self.eta_min + max(0.,
                                  (1. - self.eta_min) * (1. - self._step_count / self.to_epoch)
                                  )


class Saver:
    filename: str = None
    frames_period: int = None
    _counter: int = 0
    _active: bool = False

    def __init__(self, filename: str = None, frames_period: int = None):
        if filename is not None:
            self.filename = filename
            self.frames_period = frames_period
            self._active = True

    def save(self, algo: _BaseAlgo, frames_now: int):
        if not self._active:
            return

        if self.frames_period * self._counter < frames_now:
            if self._counter != 0:
                algo.save(self.filename)
            self._counter += 1
