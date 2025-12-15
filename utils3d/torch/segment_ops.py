from typing import *
from numbers import Number
from itertools import chain
from numbers import Integral

import torch
from torch import Tensor
import torch.nn.functional as F

from .utils import lexsort


__all__ = [
    'segment_roll',
    'segment_take',
    'segment_argmax',
    'segment_argmin',
    'group_as_segments',
    'segment_sort',
    'segment_argsort',
    'segment_topk',
    'stack_segments',
    'segment_multinomial',
]



def segment_roll(data: torch.Tensor, offsets: torch.Tensor, shift: int) -> Tensor:
    """Roll the data within each segment.
    """
    lengths = torch.diff(offsets)
    start = offsets[:-1].repeat_interleave(lengths)
    elem_indices = start + (torch.arange(data.shape[0], dtype=offsets.dtype) - start - shift) % lengths.repeat_interleave(lengths)
    data = data.gather(0, elem_indices)
    return data


def segment_take(data: Tensor, offsets: Tensor, taking: Tensor) -> Tuple[Tensor, Tensor]:
    """Take some segments from a segmented array
    """
    lengths = torch.diff(offsets)
    new_lengths = lengths[taking]
    new_offsets = torch.cat([torch.tensor([0], dtype=lengths.dtype, device=lengths.device), torch.cumsum(new_lengths, dim=0)])
    indices = torch.arange(new_offsets[-1]) + torch.repeat_interleave(offsets[taking] - new_offsets[:-1], new_lengths)
    new_data = data.index_select(0, indices)
    return new_data, new_offsets


def group_as_segments(labels: Tensor, data: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Group as segments by labels

    ## Parameters
    
    - `labels` (Tensor): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data` (Tensor, optional): shape `(N, *data_dims)` array.
        If None, return the indices in each group instead.

    ## Returns

    Assuming there are `M` difference labels:

    - `segment_labels`: `(Tensor)` shape `(M, *label_dims)` labels of of each segment
    - `data`: `(Tensor)` shape `(N,)` or `(N, *data_dims)` the rearranged data (or indices) where the same labels are grouped as a continous segment.
    - `offsets`: `(Tensor)` shape `(M + 1,)`
    
    `data[offsets[i]:offsets[i + 1]]` corresponding to the i-th segment whose label is `segment_labels[i]`
    """
    group_labels, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True, dim=0)
    if data is None:
        data = torch.arange(labels.shape[0], device=labels.device)
    offsets = torch.cat([torch.tensor([0], dtype=counts.dtype, device=counts.device), torch.cumsum(counts, dim=0)])
    data = data[torch.argsort(inv)]
    return group_labels, data, offsets


def segment_argmax(data: Tensor, offsets: Tensor) -> Tensor:
    """Compute the argmax of each segment in the segmented data.

    ## Parameters
    - `data`: (Tensor) shape `(N, ...)` the data to compute argmax from. If `data` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of the segmented data

    ## Returns
    - `argmax_indices`: (Tensor) shape `(M, ...)` the argmax indices of each segment along the first dimension.
    NOTE: If there are multiple maximum values in a segment, the index of the first one is returned.
    """
    seg_maxs = torch.segment_reduce(data, 'max', offsets=offsets, axis=0)
    lengths = torch.diff(offsets)
    is_max_mask = data == torch.repeat_interleave(seg_maxs, lengths, dim=0)
    candidate_indices = torch.where(is_max_mask, torch.arange(data.shape[0], device=data.device)[(..., *((None,) * (data.ndim - 1)))], data.shape[0])
    argmax_indices = torch.segment_reduce(candidate_indices, 'min', offsets=offsets, axis=0)
    return argmax_indices


def segment_argmin(data: Tensor, offsets: Tensor) -> Tensor:
    """Compute the argmin of each segment in the segmented data.

    ## Parameters
    - `data`: (Tensor) shape `(N, ...)` the data to compute argmin from. If `data` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of the segmented data

    ## Returns
    - `argmin_indices`: (Tensor) shape `(M, ...)` the argmin indices of each segment along the first dimension.
    NOTE: If there are multiple minimum values in a segment, the index of the first one is returned.
    """
    seg_mins = torch.segment_reduce(data, 'min', offsets=offsets, axis=0)
    lengths = torch.diff(offsets)
    is_min_mask = data == torch.repeat_interleave(seg_mins, lengths, dim=0)
    candidate_indices = torch.where(is_min_mask, torch.arange(data.shape[0], device=data.device)[(..., *((None,) * (data.ndim - 1)))], data.shape[0])
    argmin_indices = torch.segment_reduce(candidate_indices, 'min', offsets=offsets, axis=0)
    return argmin_indices


@torch.no_grad()
def large_multinomial(weights: torch.Tensor, num_samples: int, replacement: bool = False) -> torch.LongTensor:
    weights = weights.double()
    weights = weights / weights.sum()

    if replacement:
        cum_weights = torch.cumsum(weights, dim=0)
        rand = torch.rand(num_samples, dtype=torch.float64, device=weights.device)
        indices = torch.searchsorted(cum_weights, rand)
    else:
        scores = weights.log() - torch.empty_like(weights).exponential_().log()
        indices = torch.topk(scores, num_samples).indices
    return indices


def segment_argsort(input: torch.Tensor, offsets: torch.Tensor, descending: bool = False) -> torch.Tensor:
    """Compute the argsort indices within each segment.
    
    Parameters
    ----
    - `input`: (Tensor) shape `(N, ...)` the data to sort. The first dimension is treated as the segment dimension. Extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of each segment.
    - `descending`: (bool) whether to sort in descending order.

    Returns
    ----
    - `sorted_indices`: (Tensor) shape `(N, ...)` the indices that would

    """
    lengths = torch.diff(offsets)
    dim = dim % input.ndim

    segment_ids = torch.repeat_interleave(torch.arange(len(lengths), device=input.device), lengths)
    segment_ids = segment_ids.reshape(segment_ids.shape + (1,) * (input.ndim - 1))
    sorted_indices = lexsort([-input if descending else input, segment_ids], dim=0)
    return sorted_indices


def segment_sort(input: torch.Tensor, offsets: torch.Tensor = None, descending: bool = False) -> torch.return_types.sort:
    """Sort the data within each segment.
    
    Parameters
    ----
    - `input`: (Tensor) shape `(N, ...)` the data to sort.
    - `lengths`: (Tensor) shape `(M,)` the lengths of each segment, alternatively to `offsets`.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets
    - `descending`: (bool) whether to sort in descending order.
    Returns
    ----
    - `sorted`: (Tensor) shape `(N, ...)` the sorted data.
    - `indices`: (Tensor) shape `(N, ...)` the indices that would sort the data within each segment.
    """
    sorted_indices = segment_argsort(input, offsets, descending=descending)
    return torch.return_types.sort((input.index_select(0, sorted_indices), sorted_indices))


def segment_topk(input: torch.Tensor, offsets: torch.Tensor, k: Union[int, torch.Tensor], largest: bool = True) -> torch.return_types.topk:
    """Compute the top-k values and indices within each segment.
    
    Parameters
    ----
    - `input`: (Tensor) shape `(N, ...)` the data to compute top
    - `k`: (int or Tensor) the number of top elements to retrieve from each segment. If a Tensor, it should have shape `(M,)` where `M` is the number of segments.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of each segment.
    - `largest`: (bool) whether to return the largest or smallest elements.
    Returns
    ----
    - `values`: (Tensor) shape `(sum_k, ...)` the top-k values
    - `indices`: (Tensor) shape `(sum_k, ...)` the indices of the top-k values in the original input.
    where `sum_k` is the sum of all k's across segments.
    """
    lengths = torch.diff(offsets)

    sorted_indices = segment_argsort(input, offsets, descending=largest)
    local_index = torch.arange(len(input), device=input.device) - torch.repeat_interleave(offsets[:-1], lengths)
    
    if isinstance(k, int) or isinstance(k, torch.Tensor) and k.dim() == 0:
        topk_indices = sorted_indices[local_index < k]
    else:
        topk_indices = sorted_indices[local_index < torch.repeat_interleave(k, lengths)]
    
    topk_values = input.index_select(0, topk_indices)
    return torch.return_types.topk((topk_values, topk_indices))


def stack_segments(input: torch.Tensor, offsets: torch.Tensor, max_length: int = None, padding_value: Number = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack segments into a padded tensor.
    
    Parameters
    ----
    - `input`: (Tensor) shape `(N, ...)` the data to stack.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of each segment.
    - `max_length`: (int, optional) the maximum length to pad/truncate each segment to. If None, use the maximum segment length.
    - `padding_value`: (Number) the value to use for padding.

    Returns
    ----
    - `stacked`: (Tensor) shape `(M, max_length, ...)` the stacked segments, where `max_length` is the maximum segment length.
    - `mask`: (Tensor) shape `(M, max_length)` boolean mask indicating valid entries in `stacked`.
    - `indices`: (Tensor) shape `(M, max_length)` the original indices of
    """
    lengths = torch.diff(offsets)

    if max_length is None:
        max_length = lengths.max().item()
    indices = torch.arange(0, max_length, device=input.device)[None, :] + offsets[:-1, None]
    mask = indices < offsets[1:, None]
    stacked = torch.where(
        mask[(...,) + (None,) * (input.ndim - 1)], 
        input.index_select(0, indices.clamp(0, input.shape[0] - 1).flatten()).reshape(indices.shape + input.shape[1:]),
        padding_value
    )

    return stacked, mask, indices


@torch.no_grad()
def segment_multinomial(weights: torch.Tensor, offsets: torch.Tensor, num_samples: torch.Tensor, eps: float = 1e-12, replacement: bool = False) -> torch.LongTensor:
    """
    Perform multinomial sampling within each segment.
    Parameters
    ----
    - `weights`: (Tensor) shape `(N,)` the weights for sampling.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of each segment.
    - `n`: (int) the number of samples to draw from each segment.
    - `eps`: (float) a small value to avoid division by zero.
    - `replacement`: (bool) whether to sample with replacement.
    Returns
    ----
    - `sampled_indices`: (LongTensor) shape `(M * n,)` the sampled indices from each segment.
    """
    
    dtype, device = weights.dtype, weights.device
    lengths = torch.diff(offsets)

    M = len(lengths)
    weights_segment_sum = torch.segment_reduce(weights, 'sum', offsets=offsets)
    weights = weights / torch.repeat_interleave(weights_segment_sum.clamp_min(eps), lengths)

    if replacement:
        cdf = torch.cumsum(weights, dim=0)
        sampled_segment_ids = torch.repeat_interleave(torch.arange(M, device=device), n)
        u = sampled_segment_ids.to(weights.dtype) + torch.rand_like(sampled_segment_ids, dtype=weights.dtype)
        sampled_indices = torch.searchsorted(cdf, u, right=False)
        sampled_indices.clamp_(
            min=torch.repeat_interleave(offsets[:-1], num_samples),
            max=torch.repeat_interleave(offsets[1:], num_samples) - 1
        )
    else:
        scores = weights.clamp_min_(eps).log() - torch.empty_like(weights).exponential_().clamp_min(eps).log()
        sampled_indices = segment_topk(scores, offsets, k=num_samples).indices

    return sampled_indices
