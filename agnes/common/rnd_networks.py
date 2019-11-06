import typing

import torch
from gym.spaces.space import Space

from agnes.common.make_nn import make_fc, Cnn, CnnImpalaShallowBody


class RndMlp(torch.nn.Module):
    output_shape = 5

    def __init__(self, observation_space: Space):
        super().__init__()
        self.predictor_network = make_fc(observation_space.shape[0], self.output_shape,
                                         num_layers=3, hidden_size=32)

        self.target_network = make_fc(observation_space.shape[0], self.output_shape,
                                      num_layers=3, hidden_size=32).eval()

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self.predictor_network(x), self.target_network(x)


class RndCnn(torch.nn.Module):
    output_shape = 5

    def __init__(self, observation_space: Space):
        super().__init__()
        self.predictor_network = Cnn(observation_space.shape, self.output_shape,
                                     num_layers=2, hidden_size=128, body=CnnImpalaShallowBody)

        self.target_network = Cnn(observation_space.shape, self.output_shape,
                                  num_layers=2, hidden_size=128, body=CnnImpalaShallowBody).eval()

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self.predictor_network(x), self.target_network(x)
