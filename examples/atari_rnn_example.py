import time

import agnes


env_name = "BreakoutNoFrameskip-v4"

if __name__ == '__main__':
    env = agnes.make_vec_env(env_name, envs_num=4, config={"frame_stack": True})
    config, _ = agnes.PPO.get_config(env[1])

    runner = agnes.Single(env, agnes.PPO, agnes.LSTMCNN, config=config)

    # runner.worker.load("examples/distributed_rnn/Breakout.pth")
    # runner.trainer.load("examples/distributed_rnn/Breakout.pth")

    runner.log(agnes.TensorboardLogger(".logs/"), agnes.log)  #
    runner.run()

    agnes.common.Visualize(runner.worker, env).run()
