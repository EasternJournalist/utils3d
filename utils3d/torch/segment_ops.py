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
    'segment_median',
    'segment_sum',
    'group_as_segments',
    'segment_sort',
    'segment_argsort',
    'segment_topk',
    'stack_segments',
    'segment_multinomial',
    'segment_combinations',
]


def _lengths_to_offsets(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros(1, dtype=lengths.dtype, device=lengths.device), torch.cumsum(lengths, dim=0)])


def segment_roll(data: torch.Tensor, offsets: torch.Tensor, shift: int, dim: int = 0) -> Tensor:
    """Roll the data within each segment.

    Parameters
    ------
    - `data`: (Tensor).
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of the segmented data. `M` is the number of segments. Starts with 0 and end with `data.shape[dim]`.
    - `shift`: (int) the number of places by which elements are shifted. If negative, shift to left.
    - `dim`: (int) the segment dimension to roll along. Default is 0.

    Returns
    -------
    - `data`: (Tensor) the rolled data, same shape as input.
    """
    lengths = torch.diff(offsets)
    start = offsets[:-1].repeat_interleave(lengths)
    elem_indices = start + (torch.arange(data.shape[dim], dtype=offsets.dtype) - start - shift) % lengths.repeat_interleave(lengths)
    data = data.index_select(dim, elem_indices)
    return data


def segment_take(data: Tensor, offsets: Tensor, taking: Tensor, dim: int = 0) -> Tuple[Tensor, Tensor]:
    """Take some segments from a segmented array
    
    Parameters
    ------
    - `data`: (Tensor) the segmented data.
    - `offsets`: (Tensor) 1-D tensor of shape `(M + 1,)` the offsets of the segmented data. `M` is the number of segments. Starts with 0 and end with `data.shape[dim]`.
    - `taking`: (Tensor) 1-D tensor of the indices of segments to take of shape `(K,)`, or boolean mask of shape `(M,)`
    - `dim`: (int) the segment dimension to take along. Default is 0. Other dimensions are treated as batch dimensions.

    Returns
    -------
    - `new_data`: (Tensor) the new segmented data.
    - `new_offsets`: (Tensor) shape `(K + 1,)` the offsets of the new segmented data. `K` is the number of taken segments.
    """
    lengths = torch.diff(offsets)
    new_lengths = lengths[taking]
    new_offsets = _lengths_to_offsets(new_lengths)
    indices = torch.arange(data.shape[dim], device=data.device) + torch.repeat_interleave(offsets[taking] - new_offsets[:-1], new_lengths)
    new_data = data.index_select(dim, indices)
    return new_data, new_offsets


def group_as_segments(labels: Tensor, data: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Group as segments by labels

    Parameters
    ----
    - `labels` (Tensor): shape `(N, *label_dims)` array of labels for each data point. Labels can be multi-dimensional.
    - `data` (Tensor, optional): shape `(N, *data_dims)` array.
        If None, return the indices in each group instead.

    Returns
    -------
    Assuming there are `M` difference labels:

    - `segment_labels`: `(Tensor)` shape `(M, *label_dims)` labels of of each segment
    - `data`: `(Tensor)` shape `(N,)` or `(N, *data_dims)` the rearranged data (or indices) where the same labels are grouped as a continous segment.
    - `offsets`: `(Tensor)` shape `(M + 1,)`
    
    `data[offsets[i]:offsets[i + 1]]` corresponding to the i-th segment whose label is `segment_labels[i]`
    """
    group_labels, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True, dim=0)
    offsets = _lengths_to_offsets(counts)
    if data is None:
        data = torch.argsort(inv)
    else:
        data = data.index_select(0, torch.argsort(inv))
    return group_labels, data, offsets


def segment_argmax(data: Tensor, offsets: Tensor, dim: int = 0) -> Tensor:
    """Compute the argmax of each segment in the segmented data.

    Parameters
    ----------
    - `data`: (Tensor) shape `(..., N, ...)` the data to compute argmax from. If `data` may have multiple dimensions, extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of the segmented data
    - `dim`: (int) the segment axis to compute along. Default is 0.

    Returns
    -------
    - `argmax_indices`: (Tensor) shape `(..., M, ...)` the argmax indices of each segment along the first dimension.

    NOTE: If there are multiple maximum values in a segment, the index of the first one is returned. If a segment is empty, -1 is returned.
    """
    lengths = torch.diff(offsets)
    seg_maxs = torch.segment_reduce(data, 'max', offsets=offsets, axis=dim)
    seg_ids = torch.repeat_interleave(torch.arange(len(offsets) - 1, device=data.device), lengths)
    where_in_data = torch.where(data == seg_maxs.index_select(dim, seg_ids))
    where_in_argmax = where_in_data[:dim] + (seg_ids[where_in_data[dim]],) + where_in_data[dim + 1:]
    value_in_argmax = where_in_data[dim]
    sentinel_value = torch.iinfo(torch.int64).max
    argmax = torch.full(data.shape[:dim] + (len(offsets) - 1,) + data.shape[dim + 1:], fill_value=sentinel_value, dtype=torch.int64)
    flat_where_in_argmax = (torch.stack(where_in_argmax, dim=1) * torch.tensor(argmax.stride(), device=argmax.device)).sum(dim=1)
    argmax.view(-1).scatter_reduce_(0, flat_where_in_argmax, value_in_argmax, reduce='amin')
    argmax[argmax == sentinel_value] = -1
    return argmax


def segment_argmin(data: Tensor, offsets: Tensor, dim: int = 0) -> Tensor:
    """Compute the argmin of each segment in the segmented data.

    Parameters
    ----------
    - `data`: (Tensor) shape `(..., N, ...)` the data to compute argmin from. If `data` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of the segmented data
    - `dim`: (int) the segment axis to compute along. Default is 0.
    Returns
    -------
    - `argmin_indices`: (Tensor) shape `(..., M, ...)` the argmin indices of each segment along the first dimension.

    NOTE: If there are multiple minimum values in a segment, the index of the first one is returned. If a segment is empty, -1 is returned.
    """
    lengths = torch.diff(offsets)
    seg_mins = torch.segment_reduce(data, 'min', offsets=offsets, axis=dim)
    seg_ids = torch.repeat_interleave(torch.arange(len(offsets) - 1, device=data.device), lengths)
    where_in_data = torch.where(data == seg_mins.index_select(dim, seg_ids))
    where_in_argmin = where_in_data[:dim] + (seg_ids[where_in_data[dim]],) + where_in_data[dim + 1:]
    value_in_argmin = where_in_data[dim]
    sentinel_value = torch.iinfo(torch.int64).max
    argmin = torch.full(data.shape[:dim] + (len(offsets) - 1,) + data.shape[dim + 1:], fill_value=sentinel_value, dtype=torch.int64)
    flat_where_in_argmin = (torch.stack(where_in_argmin, dim=1) * torch.tensor(argmin.stride(), device=argmin.device)).sum(dim=1)
    argmin.view(-1).scatter_reduce_(0, flat_where_in_argmin, value_in_argmin, reduce='amin')
    argmin[argmin == sentinel_value] = -1
    return argmin


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


def segment_argsort(input: torch.Tensor, offsets: torch.Tensor, descending: bool = False, dim: int = 0) -> torch.Tensor:
    """Compute the argsort indices within each segment.
    
    Parameters
    ----
    - `input`: (Tensor) shape `(..., N, ...)` the data to sort. The first dimension is treated as the segment dimension. Extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of each segment.
    - `descending`: (bool) whether to sort in descending order.
    - `dim`: (int) the segment axis to sort along. Default is 0.

    Returns
    ----
    - `sorted_indices`: (Tensor) shape `(..., N, ...)` the indices that would sort the data within each segment.

    """
    lengths = torch.diff(offsets)
    segment_ids = torch.repeat_interleave(torch.arange(len(lengths), device=input.device), lengths)
    segment_ids = segment_ids.reshape(segment_ids.shape + (1,) * (input.ndim - 1))
    sorted_indices = lexsort([-input if descending else input, segment_ids], dim=dim)
    return sorted_indices


def segment_sort(input: torch.Tensor, offsets: torch.Tensor = None, descending: bool = False, dim: int = 0) -> torch.return_types.sort:
    """Sort the data within each segment.
    
    Parameters
    ----
    - `input`: (Tensor) shape `(..., N, ...)` the data to sort.
    - `lengths`: (Tensor) shape `(M,)` the lengths of each segment, alternatively to `offsets`.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets
    - `descending`: (bool) whether to sort in descending order.
    - `dim`: (int) the segment axis to sort along. Default is 0.

    Returns
    ----
    - `sorted`: (Tensor) shape `(..., N, ...)` the sorted data.
    - `indices`: (Tensor) shape `(..., N, ...)` the indices that would sort the data within each segment.
    """
    sorted_indices = segment_argsort(input, offsets, descending=descending, dim=dim)
    return torch.return_types.sort((input.index_select(dim, sorted_indices), sorted_indices))


def segment_topk(input: torch.Tensor, offsets: torch.Tensor, k: Union[int, torch.Tensor], largest: bool = True, dim: int = 0) -> torch.return_types.topk:
    """Compute the top-k values and indices within each segment.
    NOTE: if the length of a segment is less than k, the returns will contain all elements in that segment but fewer than k elements.

    Parameters
    ----
    - `input`: (Tensor) shape `(N, ...)` the data to compute top
    - `k`: (int or Tensor) the number of top elements to retrieve from each segment. If a Tensor, it should have shape `(M,)` where `M` is the number of segments.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of each segment.
    - `largest`: (bool) whether to return the largest or smallest elements. Otherwise, return the smallest elements.
    - `dim`: (int) the segment axis to compute along. Default is 0.

    Returns
    ----
    - `values`: (Tensor) shape `(sum_k, ...)` the top-k values
    - `indices`: (Tensor) shape `(sum_k, ...)` the indices of the top-k values in the original input.
    where `sum_k` is the sum of all k's across segments.
    """
    lengths = torch.diff(offsets)
    
    sorted_indices = segment_argsort(input, offsets, descending=largest, dim=dim)
    local_index = torch.arange(len(input), device=input.device) - torch.repeat_interleave(offsets[:-1], lengths)
    
    if isinstance(k, int) or isinstance(k, torch.Tensor) and k.dim() == 0:
        topk_indices = sorted_indices.index_select(dim, local_index < k)
    else:
        topk_indices = sorted_indices.index_select(dim, local_index < torch.repeat_interleave(k, lengths))
    
    topk_values = input.index_select(dim, topk_indices)
    return torch.return_types.topk((topk_values, topk_indices))


def segment_median(input: torch.Tensor, offsets: torch.Tensor, dim: int = 0) -> torch.return_types.median:
    """Compute the median of each segment.
    
    Parameters
    ----
    - `input`: (Tensor) shape `(..., N, ...)` the data to compute median from. The first dimension is treated as the segment dimension. Extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of each segment.
    - `dim`: (int) the segment axis to compute along. Default is 0.

    Returns
    ----
    - `medians`: (Tensor) shape `(..., M, ...)` the median of each segment.
    - `indices`: (Tensor) shape `(..., M, ...)` the indices of the median values in the original input.
    """
    lengths = torch.diff(offsets)
    sorted_indices = segment_argsort(input, offsets, descending=False, dim=dim)
    median_pos = offsets[:-1] + (lengths - 1) // 2
    median_indices = torch.take_along_dim(sorted_indices, median_pos[(None,) * dim + (slice(None),) + (None,) * (input.ndim - dim - 1)], dim=dim)
    median_values = torch.take_along_dim(input, median_indices, dim=dim)
    return torch.return_types.median((median_values, median_indices))


def stack_segments(input: torch.Tensor, offsets: torch.Tensor, max_length: int = None, padding_value: Number = 0, dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack segments into a padded tensor.
    
    Parameters
    ----
    - `input`: (Tensor) shape `(..., N, ...)` the data to stack.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of each segment.
    - `max_length`: (int, optional) the maximum length to pad/truncate each segment to. If None, use the maximum segment length.
    - `padding_value`: (Number) the value to use for padding.
    - `dim`: (int) the segment axis to stack along. Default is 0.

    Returns
    ----
    - `stacked`: (Tensor) shape `(..., M, max_length, ...)` the stacked segments, where `max_length` is the maximum segment length.
    - `mask`: (Tensor) shape `(..., M, max_length)` boolean mask indicating valid entries in `stacked`.
    - `indices`: (Tensor) shape `(..., M, max_length)` the indices of the stacked entries in the original input.
    """
    lengths = torch.diff(offsets)

    if max_length is None:
        max_length = lengths.max().item()
    indices = torch.arange(0, max_length, device=input.device)[None, :] + offsets[:-1, None]
    mask = indices < offsets[1:, None]
    stacked = torch.where(
        mask[(None,) * (dim) + (...,) + (None,) * (input.ndim - dim - 1)], 
        input.index_select(dim, indices.clamp(0, input.shape[dim] - 1).flatten()).reshape(input.shape[:dim] + indices.shape + input.shape[dim + 1:]),
        padding_value
    )

    return stacked, mask, indices


@torch.no_grad()
def segment_multinomial(weights: torch.Tensor, offsets: torch.Tensor, num_samples: torch.Tensor, eps: float = 1e-12, replacement: bool = False) -> torch.LongTensor:
    """
    Perform multinomial sampling within each segment.
    
    Parameters
    ----
    - `weights`: (Tensor) 1-D tensor of shape `(N,)` the weights for sampling.
    - `offsets`: (Tensor) 1-D tensor of shape `(M + 1,)` the offsets of each segment.
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
        sampled_segment_ids = torch.repeat_interleave(torch.arange(M, device=device), num_samples)
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


def _segment_local_indices(lengths: torch.Tensor = None, offsets: torch.Tensor = None):
    if offsets is None:
        offsets = _lengths_to_offsets(lengths)
    if lengths is None:
        lengths = torch.diff(offsets)
    local_indices = torch.arange(offsets[-1].item(), device=lengths.device) - torch.repeat_interleave(offsets[:-1], lengths)
    return local_indices


def segment_sum(input: torch.Tensor, offsets: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Compute the sum of each segment in the segmented data. Workaround supports for dtypes other than float32.

    NOTE: Silently assumes that the input does not contain negative lengths.

    Parameters
    ----
    - `input`: (Tensor) shape `(..., N, ...)` the data to compute sum from. If `input` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of the segmented data
    - `dim`: (int) the segment axis to compute sum along. Default is 0.

    Returns
    ----
    - `segment_sums`: (Tensor) shape `(..., M, ...)` the sum of each segment along the specified dimension.
    """
    cumsum = torch.cat([
        torch.zeros((*input.shape[:dim], 1, *input.shape[dim + 1:]), dtype=input.dtype, device=input.device), 
        torch.cumsum(input, dim=dim)
    ], dim=dim)
    segsum = torch.diff(torch.take_along_dim(cumsum, offsets, dim=dim), dim=dim)
    return segsum


def segment_cumsum(input: torch.Tensor, offsets: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute the sum of each segment in the segmented data. Workaround supports for dtypes other than float32.

    NOTE: Silently assumes that the input does not contain negative lengths.

    Parameters
    ----
    - `input`: (Tensor) shape `(..., N, ...)` the data to compute sum from. If `input` may have multiple dimensionsm, extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of the segmented data

    Returns
    ----
    - `segment_sums`: (Tensor) shape `(..., N, ...)` the cumulative sum of each segment along the specified dimension.
    """
    cumsum = torch.cat([
        torch.zeros((*input.shape[:dim], 1, *input.shape[dim + 1:]), dtype=input.dtype, device=input.device), 
        torch.cumsum(input, dim=dim)
    ], dim=dim)
    lengths = torch.diff(offsets)
    segcumsum = cumsum[(slice(None),) * dim + (slice(1, None),) + (slice(None),) * (input.ndim - dim - 1)]\
        - torch.take_along_dim(cumsum, offsets[(slice(None),) * dim + (slice(None, -1),) + (slice(None),) * (input.ndim - dim - 1)], dim=dim).repeat_interleave(lengths, dim=dim)
    return segcumsum


def segment_combinations(input: torch.Tensor, offsets: torch.Tensor, r: int = 2, with_replacement: bool = False) -> torch.Tensor:
    """Generate all combinations of elements within each segment. Vectorized implementation.
    
    Parameters
    ----
    - `input`: (Tensor) shape `(N,)` the data to generate combinations from. The first dimension is treated as the segment dimension. Extra dimensions are treated as batch dimensions.
    - `offsets`: (Tensor) shape `(M + 1,)` the offsets of each segment.
    - `r`: (int) the number of elements in each combination.
    - `with_replacement`: (bool) whether to allow repeated elements in a combination.

    Returns
    ----
    - `combinations`: (Tensor) shape `(K, r,)` the combinations from all segments, where `K` is the total number of combinations across all segments.
    - `combination_offsets`: (Tensor) shape `(M + 1,)` the offsets of combinations for each segment. NOTE: may contain zero-length segments if a segment has less than `r` elements.
    """
    lengths = torch.diff(offsets)
    
    if with_replacement is False:
        # Initialize from r=1
        comb_lengths = (lengths - r + 1).clamp(min=0)
        comb_offsets = _lengths_to_offsets(comb_lengths)
        last_indices = _segment_local_indices(lengths=comb_lengths)
        comb_indices = last_indices[:, None]
        # Iteratively build up combinations (a_0, a_1, ..., a_{r-1}),
        for r_ in range(2, r + 1):
            # Extend a_k in [a_{r_-1} + 1, n - 1 - (r - r_)], of size - (r - r_ + 1) - a_{r_-1}
            ext_lengths = torch.repeat_interleave(lengths - (r - r_ + 1), comb_lengths) - last_indices
            ext_local_indices = _segment_local_indices(lengths=ext_lengths)
            last_indices = torch.repeat_interleave(last_indices + 1, ext_lengths) + ext_local_indices           
            comb_indices = torch.cat([
                comb_indices.repeat_interleave(ext_lengths, dim=0),
                last_indices[:, None]
            ], dim=1)
            # Update combination lengths
            comb_lengths = segment_sum(ext_lengths, comb_offsets)
            comb_offsets = _lengths_to_offsets(comb_lengths)

        comb_indices = comb_indices + torch.repeat_interleave(offsets[:-1], comb_lengths).reshape(-1, 1)
        combinations = input.index_select(0, comb_indices.flatten()).reshape(comb_indices.shape)
    else:
        # Expand to lengths^r
        comb_lengths = lengths.pow(r)
        comb_offsets = _lengths_to_offsets(comb_lengths)
        base = lengths[:, None] ** torch.arange(r - 1, -1, -1, device=lengths.device)[None, :]
        comb_indices = _segment_local_indices(lengths=comb_lengths)[:, None] // base.repeat_interleave(comb_lengths, dim=0) % lengths[:, None].repeat_interleave(comb_lengths, dim=0)
        comb_indices = comb_indices + torch.repeat_interleave(offsets[:-1], comb_lengths).reshape(-1, 1)
        combinations = input.index_select(0, comb_indices.flatten()).reshape(comb_indices.shape)
    
    return combinations, comb_offsets

