import agnes
import time


env_name = "Ant-v2"  # "InvertedDoublePendulum-v2"  # "CartPole-v1"  # "Swimmer-v2"  #

if __name__ == '__main__':
    envs = agnes.make_vec_env(env_name, envs_num=16)

    runner = agnes.Single(envs, agnes.PPO, agnes.MLP)

    runner.log(agnes.CsvLogger(".logs/"), agnes.log, agnes.TensorboardLogger(".logs/"))
    runner.run()

    env = agnes.make_env(env_name)

    agnes.common.Visualize(runner.worker, env).run()
