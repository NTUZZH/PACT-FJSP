"""
Compatibility stand-ins that let the upstream HGNN files run unchanged on a
CPU-only machine with a current Python stack.

install() does three things, each only when needed:
  * gym: if the package is missing, registers a minimal module that provides
    gym.Env, gym.make and gym.envs.registration.register (the only parts the
    upstream code uses). gym.make builds the registered class directly.
  * pynvml: if the package is missing, registers a stub whose memory query
    reports zero usage (upstream test.py uses it to decide when to clear its
    environment cache on a GPU).
  * pandas.ExcelWriter.save: removed in pandas 2.0; aliased to a no-op so the
    upstream scripts, which call save() and then close(), still write their
    spreadsheets through close().
"""
import importlib
import sys
import types


def _install_gym():
    try:
        import gym  # noqa: F401
        return
    except ImportError:
        pass
    registry = {}

    class Env(object):
        pass

    def register(id, entry_point, **kwargs):
        registry[id] = entry_point

    def make(id, **kwargs):
        mod_name, cls_name = registry[id].split(':')
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls(**kwargs)

    gym = types.ModuleType('gym')
    gym.Env = Env
    gym.make = make
    envs = types.ModuleType('gym.envs')
    registration = types.ModuleType('gym.envs.registration')
    registration.register = register
    envs.registration = registration
    gym.envs = envs
    sys.modules['gym'] = gym
    sys.modules['gym.envs'] = envs
    sys.modules['gym.envs.registration'] = registration


def _install_pynvml():
    try:
        import pynvml  # noqa: F401
        return
    except ImportError:
        pass

    class _MemInfo(object):
        used = 0
        total = 1

    nv = types.ModuleType('pynvml')
    nv.nvmlInit = lambda: None
    nv.nvmlDeviceGetHandleByIndex = lambda idx: None
    nv.nvmlDeviceGetMemoryInfo = lambda handle: _MemInfo()
    sys.modules['pynvml'] = nv


def _patch_pandas():
    try:
        import pandas as pd
    except ImportError:
        return
    if not hasattr(pd.ExcelWriter, 'save'):
        pd.ExcelWriter.save = lambda self: None


def install():
    _install_gym()
    _install_pynvml()
    _patch_pandas()
