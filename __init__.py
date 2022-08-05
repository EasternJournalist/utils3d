import importlib
if importlib.find_loader('torch'):
    from . import torch_ as torch
if importlib.find_loader('numpy'):
    from . import numpy_ as numpy
    from . import gl
else:
    UserWarning("Module numpy is not found.")

from .wavefront_obj import read_obj, write_obj
