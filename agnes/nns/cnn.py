import abc
import torch
from torch.distributions import Categorical, Normal
from gym import spaces
import numpy

from agnes.nns.base import _BasePolicy
from agnes.common import make_nn
from agnes.common.init_weights import get_weights_init


class _CnnFamily(_BasePolicy, abc.ABC):
    pass


class CNNDiscreteCopy(_CnnFamily):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, policy_cnn=None, value_cnn=None):
        super(CNNDiscreteCopy, self).__init__(observation_space, action_space)

        input_shape = observation_space.shape

        # actor's layer
        if policy_cnn is None:
            self.actor_head = make_nn.Cnn(input_shape, self.actions_n, body=make_nn.CnnImpalaShallowBody)
        else:
            self.actor_head = policy_cnn(input_shape, self.actions_n)

        # critic's layer
        if value_cnn is None:
            self.critic_head = make_nn.Cnn(input_shape, 1, body=make_nn.CnnImpalaShallowBody)
        else:
            self.critic_head = value_cnn(input_shape, 1)

        self.actor_head.conv.apply(get_weights_init('relu'))

        self.critic_head.conv.apply(get_weights_init('relu'))

        if policy_cnn is None:
            self.actor_head[-1].apply(get_weights_init(0.01))
        if value_cnn is None:
            self.critic_head[-1].apply(get_weights_init(0.01))

    def wrap_dist(self, vec) -> torch.distributions.Categorical:
        dist = Categorical(logits=vec)
        return dist

    def forward(self, x):
        state_value = self.critic_head(x)

        policy = self.actor_head(x)

        return policy, state_value


class CNNDiscreteShared(_CnnFamily):
    hidden_size = 256

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super(CNNDiscreteShared, self).__init__(observation_space, action_space)
        input_shape = observation_space.shape

        self.conv = make_nn.Cnn(input_shape, self.hidden_size, num_layers=2, activate_last=True,
                                body=make_nn.CnnImpalaShallowBody)

        # actor's layer
        self.actor_head = make_nn.make_fc(self.hidden_size, self.actions_n)

        # critic's layer
        self.critic_head = make_nn.make_fc(self.hidden_size, 1)
        self.apply(get_weights_init('relu'))

        self.actor_head[-1].apply(get_weights_init(0.01))
        self.critic_head[-1].apply(get_weights_init(0.01))

    def wrap_dist(self, vec) -> torch.distributions.Categorical:
        dist = Categorical(logits=vec)
        return dist

    def forward(self, x):
        both = self.conv(x)

        state_value = self.critic_head(both)
        policy = self.actor_head(both)

        return policy, state_value
