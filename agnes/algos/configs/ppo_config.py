import re
from typing import Dict, Tuple


def atari_config() -> Dict:
    return dict(
        timesteps=10e6,
        nsteps=128,
        nminibatches=4,
        gamma=0.99,
        lam=0.95,
        noptepochs=4,
        max_grad_norm=0.5,
        learning_rate=2.5e-4,
        cliprange=lambda x: x*0.1,
        vf_coef=0.5,
        ent_coef=0.01
    )


def mujoco_config() -> Dict:
    return dict(
        timesteps=1e6,
        nsteps=2048,
        nminibatches=32,
        gamma=0.99,
        lam=0.95,
        noptepochs=10,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        cliprange=0.2,
        vf_coef=0.5,
        ent_coef=0.0
    )


def get_config(env_type: str) -> Tuple[Dict, str]:
    if env_type == 'mujoco':
        cnfg = mujoco_config()
    elif env_type == 'atari':
        cnfg = atari_config()
    else:
        cnfg = atari_config()

    return cnfg, env_type
