import agnes


def test_config():
    return dict(
        timesteps=128*4,
        nsteps=128,
        nminibatches=4,
        gamma=0.99,
        lam=0.95,
        noptepochs=4,
        max_grad_norm=0.5,
        learning_rate=2.5e-4,
        cliprange=0.1,
        vf_coef=0.5,
        ent_coef=.01,
        bptt=16
    )


def test_single_cnn():
    env_name = "PongNoFrameskip-v4"

    envs = agnes.make_env(env_name)

    runner = agnes.runners.Single(envs, agnes.PPO, agnes.CNN, config=test_config())
    runner.log(agnes.TensorboardLogger(), agnes.log)
    runner.run()


def test_single_cnn_rnn():
    env_name = "PongNoFrameskip-v4"

    envs = agnes.make_env(env_name)

    runner = agnes.runners.Single(envs, agnes.PPO, agnes.LSTMCNN, config=test_config())
    runner.run()


def test_single_cnn_a2c():
    env_name = "PongNoFrameskip-v4"

    envs = agnes.make_env(env_name)

    runner = agnes.runners.Single(envs, agnes.A2C, agnes.CNN, config=test_config())
    runner.run()


def test_single_cnn_ppo_rnd():
    env_name = "PongNoFrameskip-v4"

    envs = agnes.make_env(env_name)

    runner = agnes.runners.Single(envs, agnes.PPORND, agnes.CNN, config=test_config())
    runner.run()
