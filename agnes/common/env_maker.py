import multiprocessing
import re
import typing
from collections import defaultdict

import gym

from agnes.common.envs_prep import Monitor
from agnes.common.envs_prep import wrap_deepmind, make_atari, SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from agnes.common.envs_prep.vec_env import VecEnv

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)


def make_vec_env(env_id: str or typing.Callable,
                 envs_num: int = multiprocessing.cpu_count(),
                 config: typing.Dict = None) -> typing.Dict[str, VecEnv or str or int]:
    r"""
    Function for initializing environments in a suitable for runners way.
    Args:
        env_id (str or Callable): environment id from gym(what should be provided in gym.make(env_id)
                                    or if the environment is custom, callable which will return initialized environment
                                    (every call it should be initialized newly).
        envs_num (int): how many environments should be initialized in VecEnv object.
        config (dict): config parameters for environment.

    Returns:
        envs (VecEnv): VecEnv object with initialized environments.
        env_type (str): type of environment identified by gym.
        num_envs (int): how many environments initialized in VecEnv object.
    """
    if config is not None and "path" in config:
        if config["path"][-1] != '/':
            config["path"] = config["path"] + '/'

    if isinstance(env_id, str):
        env_type, env_id = get_env_type(env_id)

        if env_type == 'atari':
            envs, num_envs = wrap_vec_atari(env_id, envs_num=envs_num, config=config)
        else:
            envs, num_envs = wrap_vec_gym(env_id, envs_num=envs_num, config=config)
    else:
        envs, num_envs = wrap_vec_custom(env_id, envs_num=envs_num, config=config)
        env_type = 'custom'
        env_id = str(env_id).split(" ")[0].replace("<", "")

    if env_type == 'mujoco':
        envs = VecNormalize(envs)

    return {
        "env": envs,
        "env_type": env_type,
        "env_num": num_envs,
        "env_name": str(env_id)
    }


def make_env(env: str or typing.Callable, config: typing.Dict = None) -> typing.Dict[str, VecEnv or str or int]:
    return make_vec_env(env, envs_num=1, config=config)


def get_env_type(env: str) -> typing.Tuple[str, str]:
    env_id = env

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env_id types {}'.format(env_id, _game_envs.keys())

    return env_type, env_id


def wrap_vec_atari(env_name, envs_num=multiprocessing.cpu_count(), config=None) -> typing.Tuple[VecEnv, int]:
    config_default = {"frame_stack": True,
                      "path": None
                      }
    if config is None:
        config = config_default
    safe_keys = set(config_default).difference(set(config))

    for key in safe_keys:
        config[key] = config_default[key]

    def make_env(i):
        def _thunk():
            env = wrap_deepmind(
                Monitor(make_atari(env_name, max_episode_steps=100000),
                        filename=config["path"], rank=i, allow_early_resets=False)
            )
            return env

        return _thunk

    envs = [make_env(i) for i in range(envs_num)]

    if envs_num == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    if config["frame_stack"]:
        envs = VecFrameStack(envs, nstack=4)

    return envs, envs_num


def wrap_vec_gym(env_name, envs_num=multiprocessing.cpu_count(), config=None) -> typing.Tuple[VecEnv, int]:
    if config is None:
        config = {"path": None}

    def make_env(i):
        def _thunk():
            return Monitor(gym.make(env_name), filename=config["path"], rank=i, allow_early_resets=True)

        return _thunk

    envs = [make_env(i) for i in range(envs_num)]

    if envs_num == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    return envs, envs_num


def wrap_vec_custom(env_init_fun, envs_num=multiprocessing.cpu_count(), config=None) -> typing.Tuple[VecEnv, int]:
    if config is None:
        config = {"path": None}

    def make_env(i):
        return Monitor(env_init_fun, filename=config["path"], rank=i, allow_early_resets=True)

    envs = [make_env(i) for i in range(envs_num)]

    if envs_num == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    return envs, envs_num
