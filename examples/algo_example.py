import agnes
from agnes.algos.base import _BaseAlgo
from gym.spaces import Space


class RandomAlgo(_BaseAlgo):

    get_config = agnes.PPO.get_config

    def __init__(self, nn,
                 observation_space: Space,
                 action_space: Space,
                 *args, **kwargs):
        super().__init__()

        self.action_space = action_space

    def __call__(self, state, done):
        return self.action_space.sample(), None, None


env_name = "Ant-v2"  # "InvertedDoublePendulum-v2"  # "Swimmer-v2"  #

if __name__ == '__main__':
    env = agnes.make_env(env_name)

    runner = agnes.Single(env, RandomAlgo, agnes.MLP)

    agnes.common.Visualize(runner.worker, env).run()