from abc import ABC

import torch
from torch.distributions import Categorical, Distribution
import numpy


class MultiDiscrete(Distribution):
    def __init__(self, probs: torch.Tensor, actions_shapes: numpy.ndarray, validate_args=None):
        super().__init__(validate_args)
        l_probs = torch.split(probs, tuple(actions_shapes), dim=-1)
        self.l_dists = [Categorical(logits=item) for item in l_probs]

    def sample(self, sample_shape=torch.Size()):
        l_actions = [dist.sample() for dist in self.l_dists]

        t_actions = torch.stack(l_actions, dim=-1)
        return t_actions

    def log_prob(self, value):
        shape = value.shape
        l_probs = [dist.log_prob(action.view(dist.logits.shape[:-1]))
                   for (dist, action)
                   in zip(self.l_dists, value.view(-1, shape[-1]).transpose(1, 0))]
        t_probs = torch.stack(l_probs, dim=-1)
        return t_probs

    def entropy(self):
        l_entropies = [dist.entropy() for dist in self.l_dists]
        t_entropies = torch.stack(l_entropies, dim=-1)
        return t_entropies
