import typing

import torch
from gym import spaces
from torch.distributions import Categorical, Normal

from agnes.common import make_nn
from agnes.common.init_weights import get_weights_init
from agnes.nns.base import _BasePolicy


class _RecurrentFamily(_BasePolicy):
    _hs: torch.Tensor or typing.Tuple[torch.Tensor, torch.Tensor] or None = None

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super().__init__(observation_space, action_space)
        torch.backends.cudnn.benchmark = False
    
    def forward(self, *args) -> typing.Tuple[torch.Tensor,
                                             torch.Tensor or typing.Tuple[torch.Tensor, torch.Tensor],
                                             torch.Tensor]:
        pass

    def wrap_dist(self, vec) -> torch.distributions.Categorical:
        dist = Categorical(logits=vec)
        return dist

    def get_action(self, x, dones: torch.Tensor):
        assert x.ndimension() == 1 + self.obs_space_n, "Only batches are supported"
        hs = self._hs
        if self._hs is not None:
            self._hs = self._hs.masked_fill(dones.unsqueeze(0).unsqueeze(-1).bool(), 0.0)
            hs = self._hs

        probs, self._hs, state_value = self.forward(x, self._hs)

        dist = self.wrap_dist(probs)

        if hs is None:
            hs = torch.zeros_like(self._hs)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return (action.detach().squeeze(-1).cpu().numpy(),
                action.detach().cpu().numpy(),
                {
                    "additions": hs.detach().cpu().unsqueeze(0).numpy(),
                    "old_log_probs": log_prob.detach().cpu().numpy(),
                    "old_vals": state_value.detach().squeeze(-1).cpu().numpy()
                })

    def reset(self):
        self._hs = None


class RNNDiscrete(_RecurrentFamily):
    hidden_size = 64
    layers_num = 1

    def __init__(self,
                 observation_space: spaces.Space, action_space: spaces.Space):
        super(RNNDiscrete, self).__init__(observation_space, action_space)

        activation = torch.nn.Tanh

        # actor's layer
        self.actor_body_fc = make_nn.make_fc(self.obs_space, self.hidden_size, hidden_size=self.hidden_size,
                                             num_layers=2, activate_last=True, layer_norm=True, activation=activation)
        self.actor_body_rnn = torch.nn.RNN(self.hidden_size,
                                           self.hidden_size,
                                           self.layers_num)
        self.actor_head = make_nn.make_fc(self.hidden_size, self.actions_n,
                                          num_layers=1, layer_norm=True, activation=activation)

        # critic's layer
        self.critic_head = make_nn.make_fc(self.hidden_size, 1, num_layers=1, layer_norm=True, activation=activation)

        self.apply(get_weights_init('tanh'))
        self.actor_head[-1].apply(get_weights_init(0.01))
        self.critic_head[-1].apply(get_weights_init(0.01))

    def get_val(self, x) -> torch.Tensor:
        rnn_out, _, _ = self.forward_body(x, self._hs)

        state_value = self.critic_head(rnn_out)
        return state_value

    def forward(self, x: torch.Tensor, hs: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rnn_out, hs, shapes = self.forward_body(x, self._hs)

        state_value = self.critic_head(rnn_out)

        probs = self.actor_head(rnn_out)

        if shapes is not None:
            probs = probs.view(shapes + (-1,))
            state_value = state_value.view(shapes + (-1,))

        return probs, hs, state_value

    def forward_body(self, x: torch.Tensor, hs: torch.Tensor):
        shapes = None
        if len(x.shape) > 2:
            shapes = x.shape[:-1]
            x = x.contiguous().view(-1, x.shape[-1])

        fc_out = self.actor_body_fc(x)

        if shapes is not None:
            fc_out = fc_out.view(shapes + (-1,))
        else:
            fc_out = fc_out.unsqueeze(0)

        rnn_out, hs = self.actor_body_rnn(fc_out, hs)
        rnn_out = rnn_out.contiguous().view(-1, self.hidden_size)

        return rnn_out, hs, shapes


class RNNContinuous(RNNDiscrete):
    hidden_size = 64
    layers_num = 1

    def __init__(self,
                 observation_space: spaces.Space, action_space: spaces.Space, logstd=0.0):
        super(RNNContinuous, self).__init__(observation_space, action_space)

        # actor's layer
        self.actor_body_fc = make_nn.make_fc(self.obs_space, self.hidden_size, activate_last=True)
        self.actor_body_rnn = torch.nn.RNN(self.hidden_size,
                                           self.hidden_size,
                                           self.layers_num)
        self.actor_head = make_nn.make_fc(self.hidden_size, self.actions_n, num_layers=1)

        # critic's layer
        self.critic_head = make_nn.make_fc(self.hidden_size, 1, num_layers=1)

        self.log_std = torch.nn.Parameter(torch.ones(self.actions_n) * logstd, requires_grad=True)

        self.apply(get_weights_init('tanh'))
        self.actor_head.apply(get_weights_init(0.01))
        self.critic_head.apply(get_weights_init(0.01))

    def wrap_dist(self, vec) -> torch.distributions.Normal:
        std = self.log_std.expand_as(vec).exp()
        dist = Normal(vec, std)
        return dist

    def forward(self, x: torch.Tensor, hs: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rnn_out, hs, shapes = self.forward_body(x, self._hs)

        state_value = self.critic_head(rnn_out)

        mu = self.actor_head(rnn_out)

        if shapes is not None:
            mu = mu.view(shapes + (-1,))
            state_value = state_value.view(shapes + (-1,))

        return mu, hs, state_value


class _RecurrentCnnFamily(_RecurrentFamily):

    def get_val(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.actor_body_conv(x)
        conv_out_prep = conv_out.unsqueeze(1)
        conv_out_prep = conv_out_prep.transpose(0, 1)

        hs = self._hs

        rnn_out, hs = self.actor_body_rnn(conv_out_prep, hs)
        rnn_out = rnn_out.contiguous().view(-1, self.hidden_size)

        state_value = self.critic_head(rnn_out)
        return state_value

    def forward(self, x: torch.Tensor,
                hs: torch.Tensor or typing.Tuple[torch.Tensor, torch.Tensor]
                ) -> typing.Tuple[torch.Tensor, torch.Tensor or
                                  typing.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        shapes = None
        if len(x.shape) > 4:
            shapes = x.shape[:-3]
            x = x.contiguous().view((-1,) + x.shape[-3:])

        conv_out = self.actor_body_conv(x)

        if shapes is not None:
            fc_out = conv_out.view(shapes + (-1,))
        else:
            fc_out = conv_out.unsqueeze(0)

        if isinstance(hs, torch.Tensor) and isinstance(self.actor_body_rnn, torch.nn.LSTM):
            hs = tuple(hs)

        rnn_out, hs = self.actor_body_rnn(fc_out, hs)

        rnn_out = rnn_out.contiguous().view(-1, self.hidden_size)

        state_value = self.critic_head(rnn_out)
        probs = self.actor_head(rnn_out)

        if shapes is not None:
            probs = probs.view(shapes + (-1,))
            state_value = state_value.view(shapes + (-1,))

        return probs, hs, state_value


class RNNCNNDiscrete(_RecurrentCnnFamily):
    hidden_size = 256
    layers_num = 1

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, gru: bool = False):
        super(RNNCNNDiscrete, self).__init__(observation_space, action_space)

        self.actor_body_conv = make_nn.Cnn(observation_space.shape, self.hidden_size, num_layers=1,
                                           hidden_size=256, body=make_nn.CnnImpalaShallowBody, activate_last=True)

        if gru:
            self.actor_body_rnn = torch.nn.GRU(self.hidden_size,
                                               self.hidden_size,
                                               self.layers_num)
        else:
            self.actor_body_rnn = torch.nn.RNN(self.hidden_size,
                                               self.hidden_size,
                                               self.layers_num)

        # actor's layer
        self.actor_head = make_nn.make_fc(self.hidden_size, self.actions_n, num_layers=1)

        # critic's layer
        self.critic_head = make_nn.make_fc(self.hidden_size, 1, num_layers=1)

        self.apply(get_weights_init('relu'))
        self.actor_head[-1].apply(get_weights_init(0.01))
        self.critic_head[-1].apply(get_weights_init(0.01))


class LSTMCNNDiscrete(_RecurrentCnnFamily):
    hidden_size = 256
    layers_num = 1

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super(LSTMCNNDiscrete, self).__init__(observation_space, action_space)

        self.actor_body_conv = make_nn.Cnn(observation_space.shape, self.hidden_size, num_layers=1,
                                           hidden_size=256, body=make_nn.CnnImpalaShallowBody, activate_last=True)

        self.actor_body_rnn = torch.nn.LSTM(self.hidden_size,
                                            self.hidden_size,
                                            self.layers_num)

        # actor's layer
        self.actor_head = make_nn.make_fc(self.hidden_size, self.actions_n, num_layers=1)

        # critic's layer
        self.critic_head = make_nn.make_fc(self.hidden_size, 1, num_layers=1)

        self.apply(get_weights_init('relu'))
        self.actor_head[-1].apply(get_weights_init(0.01))
        self.critic_head[-1].apply(get_weights_init(0.01))

    def get_action(self, x: torch.Tensor, dones: torch.Tensor):
        assert x.ndimension() == 1 + self.obs_space_n, "Only batches are supported"
        hs = self._hs
        if self._hs is not None:
            self._hs: typing.Tuple[torch.Tensor, torch.Tensor] = (self._hs[0].
                                                                  masked_fill(dones.unsqueeze(0).unsqueeze(-1).bool(),
                                                                              0.0),
                                                                  self._hs[1].
                                                                  masked_fill(dones.unsqueeze(0).unsqueeze(-1).bool(),
                                                                              0.0)
                                                                  )
            hs = self._hs

        probs, self._hs, state_value = self.forward(x, self._hs)

        dist = self.wrap_dist(probs)

        if hs is None:
            hs = (torch.zeros_like(self._hs[0]), torch.zeros_like(self._hs[1]))

        action = dist.sample()

        log_prob = dist.log_prob(action)

        hs_numpy = (hs[0].detach().squeeze(-1).cpu().numpy(),
                    hs[1].detach().squeeze(-1).cpu().numpy())

        return (action.detach().squeeze(-1).cpu().numpy(),
                action.detach().cpu().numpy(),
                {
                    "additions": hs_numpy,
                    "old_log_probs": log_prob.detach().cpu().numpy(),
                    "old_vals": state_value.detach().squeeze(-1).cpu().numpy()
                 })
