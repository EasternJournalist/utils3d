"""
A collection of useful functions for 3D computer vision and graphics researchers in Python.
- Use `utils3d.{function}` to call the function automatically selecting the backend based on the input type (Numpy ndarray or Pytorch tensor).
- Use `utils3d.{np/pt}.{function}` to specifically call the Numpy or Pytorch version.
"""
import importlib
from typing import TYPE_CHECKING
from .helpers import lazy_import, lazy_import_from

try:
    from .interface import *
except ImportError:
    pass

__all__ = ['numpy', 'torch', 'np', 'pt']


lazy_import(globals(), '.numpy', 'numpy')
lazy_import(globals(), '.numpy', 'np')
lazy_import(globals(), '.torch', 'torch')
lazy_import(globals(), '.torch', 'pt')

# def __getattr__(name: str):
#     if (module := globals().get(name, None)):
#         return module
#     if name == 'numpy' or name == 'np':
#         module = importlib.import_module(f'.numpy', __package__)
#         globals()['numpy'] = globals()['np'] = module
#     if name == 'torch' or name == 'pt':
#         module = importlib.import_module(f'.torch', __package__)
#         globals()['torch'] = globals()['pt'] = module
#     return module


if TYPE_CHECKING:
    from . import numpy
    from . import numpy as np   # short alias
    from . import torch
    from . import torch as pt   # short alias
    from . import io