from typing import Tuple
import torch
from torch import Tensor

import triton
import triton.language as tl



@triton.jit
def _hash_multidim_32bit(ptr, dim, mask):
    hash_val = tl.load(ptr, mask=mask, other=0)
    for i in range(1, dim):
        k = tl.load(ptr + i, mask=mask, other=0)
        hash_val = (hash_val * 31) ^ k
    hash_val ^= hash_val >> 17
    hash_val *= 0x1b873593
    return hash_val


@triton.jit
def _hashmap_build_kernel_32bit(
    keys_ptr,  
    hashmap_ptr, 
    hashmap_size,
    n_elements,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements   

    slot_bit_mask = tl.cast(hashmap_size - 1, tl.int32)
    tag_bit_mask = (~slot_bit_mask) & 0x7FFF_FFFF

    # Compute hash value
    hash_val = _hash_multidim_32bit(keys_ptr + idx * dim, dim, mask)
    # Upper tag bits, lower index bits. (index must be smaller than hashmap_size)
    store_val = (hash_val & tag_bit_mask) | idx

    # Probing loop
    to_be_inserted = mask
    target_slot = hash_val & slot_bit_mask
    attempt = 0
    while (tl.sum(to_be_inserted) > 0) and (attempt < n_elements):
        # Try to insert the key index into the hash table
        prev = tl.atomic_cas(hashmap_ptr + target_slot, tl.where(to_be_inserted, -1, -2), store_val)
        # Update mask: keep only those that failed to insert
        to_be_inserted = to_be_inserted & (prev >= 0)
        attempt += 1
        # Update target_slot for next attempt
        target_slot += tl.where(to_be_inserted, 1, 0)
        target_slot &= slot_bit_mask


@triton.jit
def _hashmap_lookup_kernel_32bit(
    queries_ptr,   
    keys_ptr,       
    hashmap_ptr,    
    results_ptr,       
    hashmap_size,
    n_queries,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_queries

    slot_bit_mask = tl.cast(hashmap_size - 1, tl.int32)
    tag_bit_mask = (~slot_bit_mask) & 0x7FFF_FFFF

    # Compute hash value for queries
    query_base_ptr = queries_ptr + offs * dim
    hash_val = _hash_multidim_32bit(query_base_ptr, dim, mask)
    query_tag = hash_val & tag_bit_mask

    is_active = mask
    found_idx = tl.full((BLOCK_SIZE,), -1, tl.int32)

    # Probing loop
    curr_slot = hash_val & slot_bit_mask
    while tl.sum(is_active) > 0:
        # Compute current slot to probe
        stored_val = tl.load(hashmap_ptr + curr_slot, mask=is_active, other=-1)

        # Drop queries that hit empty slots
        is_active = is_active & (stored_val >= 0)
        
        # Extract stored index & tag
        stored_idx = stored_val & slot_bit_mask
        stored_tag = stored_val & tag_bit_mask
        # First compare tags
        is_match = is_active & (stored_tag == query_tag)
        key_base_ptr = keys_ptr + stored_idx * dim
        # Then compare full keys
        for i in range(dim):
            q = tl.load(query_base_ptr + i, mask=is_match, other=0)
            k = tl.load(key_base_ptr + i, mask=is_match, other=0)
            is_match = is_match & (q == k)
    
        # Update found indices
        success = is_match & is_active
        found_idx = tl.where(success, stored_idx, found_idx)
        is_active = is_active & (~success)
        
        # Update current slot
        curr_slot += 1
        curr_slot &= slot_bit_mask

    # Store results
    tl.store(results_ptr + offs, found_idx, mask=mask)



@triton.jit
def _hash_multidim_64bit(ptr, dim, mask):
    hash_val = tl.load(ptr, mask=mask, other=0)
    for i in range(1, dim):
        k = tl.load(ptr + i, mask=mask, other=0)
        hash_val = (hash_val * 31) ^ k
    hash_val ^= hash_val >> 33
    hash_val *= 0x100000001B3 
    return hash_val


@triton.jit
def _hashmap_build_kernel_64bit(
    keys_ptr,    
    hashmap_ptr, 
    hashmap_size,
    n_elements,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = idx < n_elements   

    slot_bit_mask = tl.cast(hashmap_size - 1, tl.int64)
    tag_bit_mask = (~slot_bit_mask) & 0x7FFF_FFFF_FFFF_FFFF

    # Compute hash value
    hash_val = _hash_multidim_64bit(keys_ptr + idx * dim, dim, mask)
    # Upper tag bits, lower index bits. (index must be smaller than hashmap_size)
    store_val = (hash_val & tag_bit_mask) | idx

    # Probing loop
    to_be_inserted = mask
    target_slot = hash_val & slot_bit_mask
    while tl.sum(to_be_inserted) > 0:
        # Try to insert the key index into the hash table
        prev = tl.atomic_cas(hashmap_ptr + target_slot, tl.where(to_be_inserted, -1, -2), store_val)
        # Update mask: keep only those that failed to insert
        to_be_inserted = to_be_inserted & (prev >= 0)

        # Update target_slot for next attempt
        target_slot += tl.where(to_be_inserted, 1, 0)
        target_slot &= slot_bit_mask


@triton.jit
def _hashmap_lookup_kernel_64bit(
    queries_ptr,   
    keys_ptr,       
    hashmap_ptr,    
    results_ptr,       
    hashmap_size,
    n_queries,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_queries

    slot_bit_mask = tl.cast(hashmap_size - 1, tl.int64)
    tag_bit_mask = (~slot_bit_mask) & 0x7FFF_FFFF_FFFF_FFFF

    # Compute hash value for queries
    query_base_ptr = queries_ptr + offs * dim
    hash_val = _hash_multidim_64bit(query_base_ptr, dim, mask)
    query_tag = hash_val & tag_bit_mask

    is_active = mask
    found_idx = tl.full((BLOCK_SIZE,), -1, tl.int64)

    # Probing loop
    curr_slot = hash_val & slot_bit_mask
    while tl.sum(is_active) > 0:
        # Compute current slot to probe
        stored_val = tl.load(hashmap_ptr + curr_slot, mask=is_active, other=-1)

        # Drop queries that hit empty slots
        is_active = is_active & (stored_val >= 0)
        
        # Extract stored index & tag
        stored_idx = stored_val & slot_bit_mask
        stored_tag = stored_val & tag_bit_mask
        # First compare tags
        is_match = is_active & (stored_tag == query_tag)
        key_base_ptr = keys_ptr + stored_idx * dim
        # Then compare full keys
        for i in range(dim):
            q = tl.load(query_base_ptr + i, mask=is_match, other=0)
            k = tl.load(key_base_ptr + i, mask=is_match, other=0)
            is_match = is_match & (q == k)
    
        # Update found indices
        success = is_match & is_active
        found_idx = tl.where(success, stored_idx, found_idx)
        is_active = is_active & (~success)
        
        # Update current slot
        curr_slot += 1
        curr_slot &= slot_bit_mask

    # Store results
    tl.store(results_ptr + offs, found_idx, mask=mask)


def hashmap_build_triton(keys: Tensor) -> Tuple[Tensor, Tensor]:
    # Convert to byte view
    keys = keys.flatten(1).contiguous().view(torch.uint8)

    # Determine hash map size (next power of two greater than 2x number of elements)
    n_keys = keys.shape[0]
    hashmap_size = 1 << ((n_keys - 1).bit_length() + 1)

    # Select 32-bit or 64-bit hash map based on size
    if hashmap_size < (1 << 28):
        dtype = torch.int32
        hashmap_build_kernel = _hashmap_build_kernel_32bit
    else:
        dtype = torch.int64
        hashmap_build_kernel = _hashmap_build_kernel_64bit

    # Convert keys and queries to appropriate dtype. Pad if necessary.
    bytes_alignment = 4 if dtype == torch.int32 else 8
    pad_size = (bytes_alignment - (keys.shape[1] % bytes_alignment)) % bytes_alignment
    if pad_size > 0:
        keys = torch.nn.functional.pad(keys, (0, pad_size), value=0)
    keys = keys.view(dtype)

    dim = keys.shape[1]
    hashmap = torch.full((hashmap_size,), -1, dtype=dtype, device=keys.device)
    
    BLOCK_SIZE = 64
    grid = ((n_keys + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    hashmap_build_kernel[grid](
        keys_ptr=keys,
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        n_elements=n_keys,
        dim=dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return hashmap


def hashmap_lookup_triton(hashmap: Tensor, keys: Tensor, queries: Tensor) -> Tensor:
    if keys.dtype != queries.dtype:
        raise ValueError(f"Keys and queries must have the same dtype. Got {keys.dtype} and {queries.dtype}.")
    if keys.shape[1:] != queries.shape[1:]:
        raise ValueError(f"Keys and queries must have matching key dimensions. Got {keys.shape[1:]} and {queries.shape[1:]}.")
    
    # Convert to byte view
    keys = keys.flatten(1).contiguous().view(torch.uint8)
    queries = queries.flatten(1).contiguous().view(torch.uint8)

    n_queries = queries.shape[0]

    hashmap_size = hashmap.shape[0]
    dtype = hashmap.dtype

    # Select 32-bit or 64-bit kernel based on dtype
    if dtype == torch.int32:
        hashmap_lookup_kernel = _hashmap_lookup_kernel_32bit
    elif dtype == torch.int64:
        hashmap_lookup_kernel = _hashmap_lookup_kernel_64bit

    # Pad and convert keys and queries to appropriate dtype.
    bytes_alignment = 4 if dtype == torch.int32 else 8
    pad_size = (bytes_alignment - (keys.shape[1] % bytes_alignment)) % bytes_alignment
    if pad_size > 0:
        keys = torch.nn.functional.pad(keys, (0, pad_size), value=0)
        queries = torch.nn.functional.pad(queries, (0, pad_size), value=0)
    keys = keys.view(dtype)
    queries = queries.view(dtype)

    dim = keys.shape[1]
    results = torch.full((n_queries,), -1, dtype=dtype, device=keys.device)
    
    BLOCK_SIZE = 64
    grid = ((n_queries + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    hashmap_lookup_kernel[grid](
        queries_ptr=queries,
        keys_ptr=keys,
        hashmap_ptr=hashmap,
        results_ptr=results,
        hashmap_size=hashmap_size,
        n_queries=n_queries,
        dim=dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return results.to(torch.int64)


def hashmap_build_lookup_triton(keys: Tensor, queries: Tensor) -> Tensor:
    if keys.dtype != queries.dtype:
        raise ValueError(f"Keys and queries must have the same dtype. Got {keys.dtype} and {queries.dtype}.")
    if keys.shape[1:] != queries.shape[1:]:
        raise ValueError(f"Keys and queries must have matching key dimensions. Got {keys.shape[1:]} and {queries.shape[1:]}.")
    
    # Convert to byte view
    keys = keys.flatten(1).contiguous().view(torch.uint8)
    queries = queries.flatten(1).contiguous().view(torch.uint8)

    n_keys = keys.shape[0]
    n_queries = queries.shape[0]

    # Determine hash map size (next power of two greater than 2x number of elements)
    hashmap_size = 1 << ((n_keys - 1).bit_length() + 1)

    # Select 32-bit or 64-bit hash map based on size
    if hashmap_size < (1 << 28):
        dtype = torch.int32
        hashmap_build_kernel = _hashmap_build_kernel_32bit
        hashmap_lookup_kernel = _hashmap_lookup_kernel_32bit
    else:
        dtype = torch.int64
        hashmap_build_kernel = _hashmap_build_kernel_64bit
        hashmap_lookup_kernel = _hashmap_lookup_kernel_64bit

    # Convert keys and queries to appropriate dtype. Pad if necessary.
    bytes_alignment = 4 if dtype == torch.int32 else 8
    pad_size = (bytes_alignment - (keys.shape[1] % bytes_alignment)) % bytes_alignment
    if pad_size > 0:
        keys = torch.nn.functional.pad(keys, (0, pad_size), value=0)
        queries = torch.nn.functional.pad(queries, (0, pad_size), value=0)
    keys = keys.view(dtype)
    queries = queries.view(dtype)

    dim = keys.shape[1]
    hashmap = torch.full((hashmap_size,), -1, dtype=dtype, device=keys.device)
    results = torch.full((n_queries,), -1, dtype=dtype, device=keys.device)
    
    BLOCK_SIZE = 64
    grid = ((n_keys + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    hashmap_build_kernel[grid](
        keys_ptr=keys,
        hashmap_ptr=hashmap,
        hashmap_size=hashmap_size,
        n_elements=n_keys,
        dim=dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    grid = ((n_queries + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    hashmap_lookup_kernel[grid](
        queries_ptr=queries,
        keys_ptr=keys,
        hashmap_ptr=hashmap,
        results_ptr=results,
        hashmap_size=hashmap_size,
        n_queries=n_queries,
        dim=dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return results.to(torch.int64)

