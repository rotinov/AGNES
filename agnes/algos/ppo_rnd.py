from typing import *

import numpy
import torch
from gym import Space

from agnes.algos.ppo import PpoClass
from agnes.common.running_mean_std import EMeanStd
from agnes.nns.initializer import _BaseChooser


class PpoRndClass(PpoClass):
    meta = "PPO-RND"

    rnd_nnets: torch.nn.Module
    rnd_nnets_cuda: bool = False

    def __init__(self, nn: _BaseChooser,
                 observation_space: Space,
                 action_space: Space,
                 rnd_nnet,
                 cnfg=None,
                 workers=1,
                 trainer=True,
                 betas: Tuple[float, float] = (0.99, 0.999),
                 eps=1e-5,
                 intr_coef=0.1, **network_args):
        super().__init__(nn, observation_space, action_space, cnfg, workers, trainer, betas, eps, **network_args)

        self.rnd_nnets = rnd_nnet(observation_space, **network_args)
        self.running_mean_std = None
        self.running_mean_std_returns = None

        if self._trainer:
            self.rnd_optimizer = torch.optim.Adam(self.rnd_nnets.parameters(), lr=self._lr_scheduler.get_lr()[0],
                                                  betas=betas,
                                                  eps=eps,
                                                  weight_decay=1e-6)
        else:
            self.rnd_nnets = self.rnd_nnets.eval()

        self.intr_coef = intr_coef

    def to(self, device: str):
        super().to(device)
        self.rnd_nnets = self.rnd_nnets.to(self._device)
        self.rnd_nnets_cuda = self._device.type == 'cuda'
        return self

    def _calculate_advantages(self, train_dict: Dict[str, numpy.ndarray]):
        # transition = {
        #     "state": self.state,
        #     "action": pred_action,
        #     "new_state": nstate,
        #     "reward": reward,
        #     "done": done,
        #     "old_log_probs": old_log_probs,
        #     "old_vals": old_vals
        # }

        n_shape = train_dict["done"].shape

        with torch.no_grad():
            t_nstates = torch.from_numpy(numpy.asarray(train_dict["new_state"][-1], dtype=numpy.float32))
            t_states = torch.from_numpy(numpy.asarray(train_dict["state"], dtype=numpy.float32))
            if self._device != torch.device('cpu'):
                t_nstates = t_nstates.to(self._device)
            if self.rnd_nnets_cuda:
                t_states = t_states.to(self._device)

            last_values = self._nnet.get_val(t_nstates).detach().squeeze(-1).cpu().numpy()

            prediction_proj, real_proj = self.rnd_nnets(t_states.view((-1,) + self.observation_space.shape))

            intr_rewards: numpy.ndarray = torch.mean(
                (prediction_proj - real_proj).pow(2),
                dim=-1
            ).detach().cpu().numpy().reshape(train_dict['reward'].shape)

            if self.running_mean_std is None:
                self.running_mean_std = EMeanStd(shape=(intr_rewards.shape[-1] if intr_rewards.ndim == 3 else 1),
                                                 epsilon=1e-8)

            self.running_mean_std.update(intr_rewards.reshape(-1,
                                                              intr_rewards.shape[-1] if intr_rewards.ndim == 3 else 1
                                                              ))
            intr_rewards_mean = self.running_mean_std.mean
            intr_rewards_std = self.running_mean_std.std
            intr_rewards = (intr_rewards - intr_rewards_mean) / (intr_rewards_std + 1e-8)

            train_dict["old_vals"] = train_dict["old_vals"].reshape(n_shape)

        train_dict['reward'] = train_dict['reward'] + self.intr_coef * intr_rewards

        # Making GAE from td residual
        n_returns = numpy.zeros_like(train_dict["old_vals"])
        lastgaelam = 0.
        nextvalues = last_values
        for t in reversed(range(n_returns.shape[0])):
            nextnonterminal = 1. - train_dict["done"][t]
            delta = train_dict["reward"][t] + self.GAMMA * nextnonterminal * nextvalues - train_dict["old_vals"][t]
            n_returns[t] = lastgaelam = delta + self.LAM * self.GAMMA * nextnonterminal * lastgaelam
            nextvalues = train_dict["old_vals"][t]

        n_returns += train_dict["old_vals"]

        transitions = {
            "state": train_dict["state"],
            "action": train_dict["action"],
            "returns": n_returns,
            "old_log_probs": train_dict["old_log_probs"],
            "old_vals": train_dict["old_vals"]
        }

        if train_dict.get("additions") is not None:
            transitions["additions"] = train_dict.get("additions")

        if train_dict["reward"].ndim == 1 or self.is_recurrent:
            if self.is_recurrent:
                transitions["done"] = train_dict["done"]
        else:
            for key in transitions.keys():
                transitions[key] = transitions[key].reshape((-1,) + transitions[key].shape[2:])

        return [dict(zip(transitions, t)) for t in zip(*transitions.values())]

    def _one_train(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        info = super()._one_train(data)

        for key in data.keys():
            if self._device != torch.device('cpu'):
                data[key] = data[key].to(self._device)

        # Random Network Distillation
        prediction_proj, real_proj = self.rnd_nnets(data["state"].view((-1,) + self.observation_space.shape))
        rnd_loss = (prediction_proj - real_proj.detach()).pow(2).mean()
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

        return info

    def _one_train_seq(self, data: Dict[str, torch.Tensor or List[torch.Tensor]]) -> Dict[str, float]:
        info = super()._one_train_seq(data)

        STATES = torch.cat(data["state"], dim=0).to(self._device)

        # Random Network Distillation
        prediction_proj, real_proj = self.rnd_nnets(STATES.view((-1,) + self.observation_space.shape))
        rnd_loss = (prediction_proj - real_proj.detach()).pow(2).mean()
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

        return info

    def get_state_dict(self) -> Dict[str, dict]:
        assert self._trainer
        return {
            "nnet": self._nnet.state_dict(),
            "rnd": self.rnd_nnets.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "rnd_optimizer": self.rnd_optimizer.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, dict]) -> tuple:
        info = []
        if state_dict.get("optimizer") and hasattr(self, "_optimizer"):
            info.append(self._optimizer.load_state_dict(state_dict["optimizer"]))
        if state_dict.get("rnd_optimizer") and hasattr(self, "rnd_optimizer"):
            info.append(self.rnd_optimizer.load_state_dict(state_dict["rnd_optimizer"]))
        return (self._nnet.load_state_dict(state_dict["nnet"]),
                self.rnd_nnets.load_state_dict(state_dict["rnd"]),
                *info)


class PpoRndInitializer:
    rnd_nnet = None
    intr_coef = 0.01
    output_shape = 10

    betas = (0.99, 0.999)
    eps = 1e-5

    def config(self, rnd_nnet=None, intr_coef: float = 0.01, output_shape: int = 10,
               betas: Tuple[float, float] = (0.99, 0.999), eps: float = 1e-5):
        self.rnd_nnet = rnd_nnet
        self.intr_coef = intr_coef
        self.betas = betas
        self.eps = eps
        return self

    def __call__(self, nn: _BaseChooser,
                 observation_space: Space,
                 action_space: Space,
                 cnfg: Dict = None,
                 workers: int = 1,
                 trainer: bool = True, **network_args):
        from agnes.common import rnd_networks
        if self.rnd_nnet is not None:
            pass
        elif nn.meta == "MLP" or nn.meta == "RNN":
            self.rnd_nnet = rnd_networks.RndMlp
        elif nn.meta.find("CNN") != -1:
            self.rnd_nnet = rnd_networks.RndCnn
        else:
            raise UserWarning("No network for RND provided")

        self.rnd_nnet.output_shape = self.output_shape

        return PpoRndClass(nn,
                           observation_space,
                           action_space,
                           self.rnd_nnet,
                           cnfg,
                           workers,
                           trainer,
                           betas=self.betas,
                           eps=self.eps,
                           intr_coef=self.intr_coef, **network_args)

    @staticmethod
    def get_config(env_type: str):
        return PpoRndClass.get_config(env_type)

    @property
    def meta(self) -> str:
        return PpoRndClass.meta


PPORND = PpoRndInitializer()
