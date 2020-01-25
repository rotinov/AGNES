import time

import agnes

env_name = "InvertedDoublePendulum-v2"  # "Swimmer-v2"  # "BreakoutNoFrameskip-v4"  #

if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name)

    runner = agnes.Single(envs, agnes.PPORND, agnes.MLP)
    runner.log(agnes.TensorboardLogger('.logs/'), agnes.log)
    runner.run()

    env = agnes.make_env(env_name)

    agnes.common.Visualize(runner.worker, env).run()
