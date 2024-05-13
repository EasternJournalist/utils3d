"""
A package for common utility functions in 3D computer graphics and vision.
"""

import importlib

def __getattr__(module_name: str):
    return importlib.import_module(f'.{module_name}', __package__)

if __name__ == '__main__':
    from . import torch
    from . import numpy
    from . import io