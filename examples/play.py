import agnes

env_name = "Ant-v2"

if __name__ == '__main__':
    env = agnes.make_env(env_name)

    runner = agnes.Single(env, agnes.PPO, agnes.MLP)

    runner.load("results/MuJoCo/Ant-v2_MLP/PPO/weights.pth")

    agnes.common.Visualize(runner.worker, env).prerun(1000).run()
