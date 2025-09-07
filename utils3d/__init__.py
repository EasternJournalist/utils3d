"""
A package for common utility functions in 3D computer graphics and vision. 
Providing NumPy utilities in `utils3d.numpy`, PyTorch utilities in `utils3d.torch`, and IO utilities in `utils3d.io`.
"""
import importlib
from typing import TYPE_CHECKING

try:
    from .interface import *
except ImportError:
    pass

__all__ = ['numpy', 'torch', 'io', 'np', 'pt']

def __getattr__(name: str):
    if (module := globals().get(name, None)):
        return module
    if name == 'numpy' or name == 'np':
        module = importlib.import_module(f'.numpy', __package__)
        globals()['numpy'] = globals()['np'] = module
    if name == 'torch' or name == 'pt':
        module = importlib.import_module(f'.torch', __package__)
        globals()['torch'] = globals()['pt'] = module
    if name == 'io':
        module = importlib.import_module(f'.io', __package__)
        globals()['io'] = module
    return module


if TYPE_CHECKING:
    from . import numpy
    from . import numpy as np   # short alias
    from . import torch
    from . import torch as pt   # short alias
    from . import io