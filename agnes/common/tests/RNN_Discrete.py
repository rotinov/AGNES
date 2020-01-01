import agnes


def test_config():
    return dict(
        timesteps=30000,
        nsteps=128,
        nminibatches=4,
        gamma=0.98,
        lam=0.95,
        noptepochs=4,
        max_grad_norm=1.0,
        learning_rate=lambda x: 2.5e-4*x,
        cliprange=lambda x: 0.1*x,
        vf_coef=0.5,
        ent_coef=.01,
        bptt=8
    )


def test_single():
    env = agnes.make_env('CartPole-v0')

    runner = agnes.Single(env, agnes.PPO, agnes.RNN, config=test_config())
    runner.log(agnes.log)
    runner.run()
