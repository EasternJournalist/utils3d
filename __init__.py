import importlib
if importlib.find_loader('torch'):
    from . import torch_
if importlib.find_loader('numpy'):
    from . import numpy_

from .wavefront_obj import read_obj