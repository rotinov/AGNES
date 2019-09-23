import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from common.init_weights import weights_init
from gym import spaces
import numpy


def mlp2l(x, y):
    return nn.Sequential(nn.Linear(x, 64),
                         nn.Tanh(),
                         nn.Linear(64, 64),
                         nn.Tanh(),
                         nn.Linear(64, y))


def mlp1l(x, y):
    return nn.Sequential(nn.Linear(x, 128),
                         nn.Tanh(),
                         nn.Linear(128, y))


class MLPDiscrete(nn.Module):
    np_type = numpy.int16

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Discrete(5), mlp_fun=mlp1l):
        super(MLPDiscrete, self).__init__()
        self.action_space = action_space

        # actor's layer
        self.actor_head = nn.Sequential(mlp_fun(observation_space.shape[0], action_space.n),
                                        nn.Softmax(-1))

        # critic's layer
        self.critic_head = mlp_fun(observation_space.shape[0], 1)

        # self.apply(weights_init)

    def forward(self, x):
        state_value = self.critic_head(x)

        policy = self.actor_head(x)
        # print(policy)
        dist = Categorical(policy)

        return dist, state_value

    def get_action(self, x):
        dist, _ = self.forward(x)
        action = dist.sample()

        state_value = self.critic_head(x)
        return action.detach().cpu().numpy(), action.detach().cpu().numpy(), (dist.log_prob(action).detach().cpu().numpy(), state_value.detach().cpu().item())


class MLPContinuous(nn.Module):
    np_type = numpy.float32

    def __init__(self,
                 observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 action_space=spaces.Box(low=-10, high=10, shape=(1,)),
                 logstd=0.0, mlp_fun=mlp2l):
        super(MLPContinuous, self).__init__()
        self.action_space = action_space

        # actor's layer
        self.actor_head = mlp_fun(observation_space.shape[0], action_space.shape[0])

        # critic's layer
        self.critic_head = mlp_fun(observation_space.shape[0], 1)

        self.log_std = nn.Parameter(torch.ones(action_space.shape[0]) * logstd)

        self.apply(weights_init)

    def forward(self, x):
        state_value = self.critic_head(x)

        mu = self.actor_head(x)
        std = self.log_std.expand_as(mu).exp()
        dist = Normal(mu, std)

        return dist, state_value

    def get_action(self, x):
        dist, _ = self.forward(x)
        smpled = dist.sample()
        action = torch.clamp(smpled, self.action_space.low[0], self.action_space.high[0])

        state_value = self.critic_head(x)
        return action.detach().cpu().numpy(), smpled.detach().cpu().numpy(), (dist.log_prob(smpled).detach().cpu().numpy(), state_value.detach().cpu().item())


def MLP(observation_space=spaces.Box(low=-10, high=10, shape=(1,)),
        action_space=spaces.Discrete(5),
        logstd=0.0):
    if isinstance(action_space, spaces.Box):
        return MLPContinuous(observation_space, action_space, logstd)
    else:
        return MLPDiscrete(observation_space, action_space)
