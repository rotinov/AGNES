import agnes


def test_config():
    return dict(
        timesteps=30000,
        nsteps=128,
        nminibatches=4,
        gamma=1.0,
        lam=0.95,
        noptepochs=4,
        max_grad_norm=2.0,
        learning_rate=lambda x: 2.5e-4*x,
        cliprange=lambda x: 0.1*x,
        vf_coef=0.5,
        ent_coef=.01
    )


if __name__ == '__main__':
    env = agnes.make_vec_env('CartPole-v0', envs_num=2)

    runner = agnes.DistributedMPI(env, agnes.PPO, agnes.MLP, config=test_config())
    runner.log(agnes.log)
    runner.run()
