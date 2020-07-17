import random
from collections import deque
from typing import *

from agnes.runners.single import Single
from agnes.algos.base import _BaseAlgo


class CompetitiveRunner(Single):
    def __init__(self, env, algo, nn, config=None, all_cuda=False, breed_size=50, **network_args):
        super().__init__(env, algo, nn, config, all_cuda, **network_args)

        self.opponent: _BaseAlgo = algo(nn, self.env.observation_space, self.env.action_space,
                                        self.cnfg, trainer=False, workers=self.vec_num, **network_args)

        self.breed: Deque[dict] = deque(maxlen=breed_size)

    def _one_run(self):
        self.breed.append(self.trainer.get_state_dict())

        data = None
        epinfos = []
        for step in range(self.nsteps):
            # AI's turn
            action, pred_action, out = self.worker(self.state, self.done)
            n_state_other, reward, done, infos = self.env.step(action)

            if infos[0].get("change_turn") is None or infos[0].get("change_turn"):
                change_turn = done_second = False
                while not change_turn and not done_second:
                    # other's turn
                    action_other, _, _ = self.opponent(n_state_other, done)

                    nstate, _, done_second, infos_others = self.env.step(action_other)
                    change_turn = infos_others[0].get("change_turn") is None or infos_others[0].get("change_turn")
                self.done = done * done_second
            else:
                self.done = done
                nstate = n_state_other

            maybeepinfo = infos[0].get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)

            transition = {
                "state": self.state,
                "action": pred_action,
                "new_state": nstate,
                "reward": reward,
                "done": self.done
            }

            transition.update(out)

            data = self.worker.experience(transition)

            self.state = nstate

        # variety
        self.opponent.load_state_dict(random.choice(self.breed))

        return data, epinfos
