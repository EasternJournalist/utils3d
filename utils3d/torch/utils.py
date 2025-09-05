from typing import *

import torch
import torch.nn.functional as F

from ._helpers import batched
from .._helpers import no_warnings


__all__ = [
    'sliding_window',
    'masked_min',
    'masked_max',
    'lookup',
]


def sliding_window(x: torch.Tensor, window_size: Tuple[int, ...], stride: Tuple[int, ...], dim: Tuple[int, ...]) -> torch.Tensor:
    "Create a sliding window view of the input tensor on specified dimensions."
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    assert len(window_size) == len(stride) == len(dim)
    for i in range(len(window_size)):
        x = x.unfold(dim[i], window_size[i], stride[i])
    return x


def masked_min(input: torch.Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Similar to torch.min, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min()
    else:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min(dim=dim, keepdim=keepdim)


def masked_max(input: torch.Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Similar to torch.max, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max()
    else:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max(dim=dim, keepdim=keepdim)
    

def lookup(key: torch.Tensor, query: torch.Tensor) -> torch.LongTensor:
    """
    Find the indices of `query` in `key`.

    ### Parameters
        key (torch.Tensor): shape (K, ...), the array to search in
        query (torch.Tensor): shape (Q, ...), the array to search for

    ### Returns
        torch.Tensor: shape (Q,), indices of `query` in `key`, or -1. If a query is not found in key, the corresponding index will be -1.
    """
    unique, inverse = torch.unique(
        torch.cat([key, query], dim=0),
        dim=0,
        return_inverse=True
    )
    index = torch.full((unique.shape[0],), -1, dtype=torch.long, device=key.device)
    index.scatter_(0, inverse[:key.shape[0]], torch.arange(key.shape[0], device=key.device))
    return index[inverse[key.shape[0]:]]
