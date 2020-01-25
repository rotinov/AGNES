import time
import typing
from collections import deque

import numpy

from agnes.algos.base import _BaseAlgo
from agnes.common.envs_prep.vec_env import VecEnv


class Visualize:
    def __init__(self, algo: _BaseAlgo, env: typing.Dict[str, VecEnv or str or int]):
        self.worker: _BaseAlgo = algo
        self.env = env["env"]

    def prerun(self, nsteps: int = 1000):
        state = self.env.reset()
        done = numpy.zeros(1)

        for step in range(nsteps):
            action, pred_action, out = self.worker(state, done)
            nstate, reward, done, infos = self.env.step(action)
            state = nstate

        return self

    def run(self) -> None:
        state = self.env.reset()
        # self.worker.reset()
        self.env.render()
        done = numpy.zeros(1)

        fps = 30

        while True:
            action, pred_action, out = self.worker(state, done)
            nstate, reward, done, infos = self.env.step(action)
            self.env.render()
            time.sleep(1. / float(fps))
            state = nstate
