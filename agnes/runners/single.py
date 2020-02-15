import time
import typing
from collections import deque

import numpy
from torch import cuda

from agnes.algos.base import _BaseAlgo
from agnes.nns.initializer import _BaseChooser
from agnes.runners.base_runner import BaseRunner


class Single(BaseRunner):
    """"Single" runner releases learning with a single worker and a trainer.
    "Single" runner is compatible with vector environments(config or env_type can be specified manually).
    """

    def __init__(self, env, algo, nn: _BaseChooser, config: typing.Dict = None, all_cuda: bool = False, **network_args):
        super().__init__(env, algo, nn, config)

        print('Env type: ', self.env_type, 'Envs num:', self.vec_num)

        self.trainer: _BaseAlgo = algo(nn, self.env.observation_space, self.env.action_space,
                                       self.cnfg, workers=self.vec_num, **network_args)

        self.worker: _BaseAlgo = algo(nn, self.env.observation_space, self.env.action_space,
                                      self.cnfg, workers=self.vec_num, trainer=False, **network_args)
        if cuda.is_available():
            try:
                self.trainer = self.trainer.to('cuda:0')
            except RuntimeError:
                self.trainer = self.trainer.to('cpu')
            if all_cuda:
                try:
                    self.worker = self.worker.to('cuda:0')
                except RuntimeError:
                    self.worker = self.worker.to('cpu')

    def run(self, log_interval: int = 1):
        print(self.trainer.device_info(), 'will be used.')
        nbatch = self.nsteps * self.env.num_envs

        self.worker.update(self.trainer)

        self.state = self.env.reset()
        self.done = numpy.zeros(self.env.num_envs, dtype=numpy.bool)

        run_times = int(numpy.ceil(self.timesteps / self.nsteps))
        epinfobuf = deque(maxlen=100)
        tfirststart = time.perf_counter()

        for nupdates in range(1, run_times+1):
            lr_things = []

            if nupdates % log_interval == 0:
                self.logger.stepping_environment()
            tstart = time.perf_counter()

            data, epinfos = self._one_run()

            if nupdates % log_interval == 0:
                self.logger.done()
            lr_thing = self.trainer.train(data)
            lr_things.extend(lr_thing)
            epinfobuf.extend(epinfos)
            
            self.worker.update(self.trainer)

            tnow = time.perf_counter()

            self._one_log(lr_things, epinfobuf, nbatch, tfirststart, tstart, tnow, nupdates,
                          print_out=nupdates % log_interval == 0)

        self.env.close()

    def __del__(self):
        self.logger.close()

        if hasattr(self, 'env'):
            del self.env
        if hasattr(self, 'trainer'):
            del self.trainer
        if hasattr(self, 'worker'):
            del self.worker
