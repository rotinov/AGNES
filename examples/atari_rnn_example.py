import time

import agnes


env_name = "BreakoutNoFrameskip-v4"

if __name__ == '__main__':
    env = agnes.make_vec_env(env_name, envs_num=4, config={"frame_stack": True})

    runner = agnes.Single(env, agnes.PPO, agnes.LSTMCNN)

    # runner.worker.load("examples/distributed_rnn/Breakout.pth")
    # runner.trainer.load("examples/distributed_rnn/Breakout.pth")

    runner.log(agnes.TensorboardLogger(".logs/"), agnes.CsvLogger(".logs/"))
    runner.save_every("temp.pth", int(1e6))
    runner.run()
    runner.save("final.pth")
