import abc
from typing import Dict

import numpy
from gym.core import Env

from agnes.algos.base import _BaseAlgo
from agnes.common import logger
from agnes.common.schedules import Saver
from agnes.nns.initializer import _BaseChooser


class BaseRunner(abc.ABC):
    logger = logger.ListLogger()
    saver: Saver = Saver()
    workers_num = 1
    trainer: _BaseAlgo
    worker: _BaseAlgo
    env: Env

    state: numpy.ndarray
    done: numpy.ndarray

    def __init__(self, env, algo, nn: _BaseChooser, config: Dict):
        self.env = env["env"]
        self.nn_name = nn.meta

        self.cnfg, self.env_type = algo.get_config(env["env_type"])
        if config is not None:
            self.cnfg = config

        self.timesteps = self.cnfg['timesteps']
        self.nsteps = self.cnfg['nsteps']

        self.vec_num = env["env_num"]
        self.env_id = env["env_name"]

    def is_trainer(self) -> bool:
        return True

    def load(self, filename) -> None:
        if self.is_trainer():
            self.trainer.load(filename)

        if hasattr(self, "worker"):
            self.worker.load(filename)

    def log(self, *args) -> None:
        if self.is_trainer():
            self.logger = logger.ListLogger(*args)
            self.logger.info({
                "envs_num": self.vec_num * self.workers_num,
                "device": self.trainer.device_info(),
                "env_type": self.env_type,
                "NN type": self.nn_name,
                "algo": self.trainer.meta,
                "env_name": self.env_id
            })

    def run(self, log_interval: int = 1):
        pass

    def save_every(self, filename: str, frames_period: int) -> None:
        if self.is_trainer():
            self.saver = Saver(filename, frames_period)

    def save(self, filename: str) -> None:
        if self.is_trainer():
            self.trainer.save(filename)

    def _one_log(self, lr_things, epinfobuf, nbatch, tfirststart, tstart, tnow, nupdates, stepping_to_learning=None,
                 print_out=True):

        train_dict = {k: logger.safemean([dic[k] for dic in lr_things]) for k in lr_things[0]}

        kvpairs = {
            "eplenmean": logger.safemean(numpy.asarray([epinfo['l'] for epinfo in epinfobuf]).reshape(-1)),
            "eprewmean": logger.safemean(numpy.asarray([epinfo['r'] for epinfo in epinfobuf]).reshape(-1)),
            "fps": int(nbatch / (tnow - tstart)),
            "misc/nupdates": nupdates,
            "misc/serial_timesteps": self.nsteps * nupdates,
            "misc/time_elapsed": (tnow - tfirststart),
            "misc/total_timesteps": self.nsteps * nupdates * self.workers_num * self.vec_num
        }

        kvpairs.update(train_dict)

        if stepping_to_learning is not None:
            kvpairs['misc/stepping_to_learning'] = stepping_to_learning

        self.logger(kvpairs, nupdates, print_out=print_out)

    def _one_run(self):
        data = None
        epinfos = []
        for step in range(self.nsteps):
            action, pred_action, out = self.worker(self.state, self.done)
            nstate, reward, done, infos = self.env.step(action)
            self.done = done
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            transition = {
                "state": self.state,
                "action": pred_action,
                "new_state": nstate,
                "reward": reward,
                "done": done
            }

            transition.update(out)

            data = self.worker.experience(transition)

            self.state = nstate

        return data, epinfos

    def __del__(self):
        self.env.close()
        del self.env
