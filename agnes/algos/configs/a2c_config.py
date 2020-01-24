from typing import Dict, Tuple


def atari_config() -> Dict:
    return dict(
        timesteps=10e6,
        nsteps=32,
        nminibatches=1,
        gamma=0.99,
        lam=0.95,
        noptepochs=1,
        max_grad_norm=0.5,
        learning_rate=lambda x: 7e-4*x,
        vf_coef=0.5,
        ent_coef=0.01,
        bptt=16
    )


def mujoco_config() -> Dict:
    return dict(
        timesteps=1e6,
        nsteps=64,
        nminibatches=1,
        gamma=0.99,
        lam=0.95,
        noptepochs=1,
        max_grad_norm=0.5,
        learning_rate=lambda x: 7e-4*x,
        vf_coef=0.5,
        ent_coef=0.0,
        bptt=8
    )


def get_config(env_type: str) -> Tuple[Dict, str]:
    if env_type == 'mujoco':
        cnfg = mujoco_config()
    elif env_type == 'atari':
        cnfg = atari_config()
    else:
        cnfg = atari_config()

    return cnfg, env_type
