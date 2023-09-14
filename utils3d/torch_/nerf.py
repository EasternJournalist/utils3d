from typing import *
from numbers import Number

import torch
from .utils import image_uv


__all__ = [
    'get_rays',
    'get_image_rays',
    'get_mipnerf_cones',
    'volume_rendering',
    'uniform_bin_sample',
    'importance_sample',
]


def get_rays(extrinsics: torch.Tensor, intrinsics: torch.Tensor, uv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        extrinsics: (..., 4, 4) extrinsic matrices.
        intrinsics: (..., 3, 3) intrinsic matrices.
        uv: (..., n_rays, 2) uv coordinates of the rays. 

    Returns:
        rays_o: (..., 1, 3) ray origins
        rays_d: (..., n_rays, 3) ray directions. NOTE: ray directions are NOT normalized. They actuallys makes rays_o + rays_d * z = world coordinates, where z is the depth.
    """
    uvz = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1).to(extrinsics)                                                          # (n_batch, n_views, n_rays, 3)

    with torch.cuda.amp.autocast(enabled=False):
        inv_transformation = (intrinsics @ extrinsics[:, :, :3, :3]).inverse()
        inv_extrinsics = extrinsics.inverse()
    rays_d = uvz @ inv_transformation.transpose(-1, -2)                                                  
    rays_o = inv_extrinsics[:, :, None, :3, 3]                                                                                          # (n_batch, n_views, 1, 3)
    return rays_o, rays_d


def get_image_rays(extrinsics: torch.Tensor, intrinsics: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    uv = image_uv(height, width).to(extrinsics).flatten(0, 1)
    return get_rays(extrinsics, intrinsics, uv)


def get_mipnerf_cones(rays_o: torch.Tensor, rays_d: torch.Tensor, z_vals: torch.Tensor, pixel_width: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        rays_o: (..., n_rays, 3) ray origins
        rays_d: (..., n_rays, 3) ray directions.
        z_vals: (..., n_rays, n_samples) z values.
        pixel_width: (...) pixel width. = 1 / (normalized focal length * width)
    
    Returns:
        mu: (..., n_rays, n_samples, 3) cone mu.
        sigma: (..., n_rays, n_samples, 3, 3) cone sigma.
    """
    t_mu = (z_vals[..., 1:] + z_vals[..., :-1]).mul_(0.5)
    t_delta = (z_vals[..., 1:] - z_vals[..., :-1]).mul_(0.5)
    t_mu_square = t_mu.square()
    t_delta_square = t_delta.square()
    t_delta_quad = t_delta_square.square()
    mu_t = t_mu + 2.0 * t_mu * t_delta_square / (3.0 * t_mu_square + t_delta_square)
    sigma_t = t_delta_square / 3.0 - (4.0 / 15.0) * t_delta_quad / (3.0 * t_mu_square + t_delta_square).square() * (12.0 * t_mu_square - t_delta_square)
    sigma_r = (pixel_width[..., None, None].square() / 3.0) * (t_mu_square / 4.0 + (5.0 / 12.0) * t_delta_square - (4.0 / 15.0) * t_delta_quad / (3.0 * t_mu_square + t_delta_square))
    points_mu = rays_o[:, :, :, None, :] + rays_d[:, :, :, None, :] * mu_t[..., None]
    d_dt = rays_d[..., :, None] * rays_d[..., None, :]      # (..., n_rays, 3, 3)
    points_sigma = sigma_t[..., None, None] * d_dt[..., None, :, :] + sigma_r[..., None, None] * (torch.eye(3).to(rays_o) - d_dt[..., None, :, :])
    return points_mu, points_sigma


def volume_rendering(color: torch.Tensor, sigma: torch.Tensor, z_vals: torch.Tensor, ray_length: torch.Tensor, rgb: bool = True, depth: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given color, sigma and z_vals (linear depth of the sampling points), render the volume.

    NOTE: By default, color and sigma should have one less sample than z_vals, in correspondence with the average value in intervals.
    If queried color are aligned with z_vals, we use trapezoidal rule to calculate the average values in intervals.

    Args:
        color: (..., n_samples or n_samples - 1, 3) color values.
        sigma: (..., n_samples or n_samples - 1) density values.
        z_vals: (..., n_samples) z values.
        ray_length: (...) length of the ray

    Returns:
        rgb: (..., 3) rendered color values.
        depth: (...) rendered depth values.
        weights (..., n_samples) weights.
    """
    dists = (z_vals[..., 1:] - z_vals[..., :-1]) * ray_length[..., None]
    if color.shape[-2] == z_vals.shape[-1]:
        color = (color[..., 1:, :] + color[..., :-1, :]).mul_(0.5)
        sigma = (sigma[..., 1:] + sigma[..., :-1]).mul_(0.5)                                        
    sigma_delta = sigma * dists                                                      
    transparancy = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=-1).cumsum(dim=-1)).exp_()     # First cumsum then exp for numerical stability
    alpha = 1.0 - (-sigma_delta).exp_()                                               
    weights = alpha * transparancy
    if rgb:
        rgb = torch.sum(weights[:, None] * color, dim=-2) if rgb else None        
    if depth:
        z_vals = (z_vals[..., 1:] + z_vals[..., :-1]).mul_(0.5)
        depth = torch.sum(weights * z_vals, dim=-2) if depth else None            
    return rgb, depth, weights


def bin_sample(size: Union[torch.Size, Tuple[int, ...]], n_samples: int, min_value: Number, max_value: Number, spacing: Literal['linear', 'inverse_linear'], dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    """
    Uniformly (or uniformly in inverse space) sample z values in `n_samples` bins in range [min_value, max_value].
    Args:
        size: size of the rays
        n_samples: number of samples to be sampled, also the number of bins
        min_value: minimum value of the range
        max_value: maximum value of the range
        space: 'linear' or 'inverse_linear'. If 'inverse_linear', the sampling is uniform in inverse space.
    
    Returns:
        z_rand: (*size, n_samples) sampled z values, sorted in ascending order.
    """
    if spacing == 'linear':
        pass
    elif spacing == 'inverse_linear':
        min_value = 1.0 / min_value
        max_value = 1.0 / max_value
    bin_length = (max_value - min_value) / n_samples
    z_rand = (torch.rand(*size, n_samples, device=device, dtype=dtype) - 0.5) * bin_length + torch.linspace(min_value + bin_length * 0.5, max_value - bin_length * 0.5, n_samples, device=device, dtype=dtype)   
    if spacing == 'inverse_linear':
        z_rand = 1.0 / z_rand
    return z_rand


def importance_sample(z_vals: torch.Tensor, weights: torch.Tensor, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Importance sample z values.

    NOTE: By default, weights should have one less sample than z_vals, in correspondence with the intervals.
    If weights has the same number of samples as z_vals, we use trapezoidal rule to calculate the average weights in intervals.

    Args:
        z_vals: (..., n_rays, n_input_samples) z values, sorted in ascending order.
        weights: (..., n_rays, n_input_samples or n_input_samples - 1) weights.
        n_samples: number of output samples for importance sampling.
    
    Returns:
        z_importance: (..., n_rays, n_samples) importance sampled z values, unsorted.
    """
    if weights.shape[-1] == z_vals.shape[-1]:
        weights = (weights[..., 1:] + weights[..., :-1]).mul_(0.5)
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)                      # (..., n_rays, n_input_samples - 1)
    bins_a, bins_b = z_vals[..., :-1], z_vals[..., 1:]

    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)                          # (..., n_rays, n_input_samples - 1)
    cdf = torch.cumsum(pdf, dim=-1)
    u = torch.rand(z_vals.shape[:-1], n_samples, device=z_vals.device, dtype=z_vals.dtype)
    
    inds = torch.searchsorted(cdf, u, right=True).clamp(0, cdf.shape[-1] - 1)         # (..., n_rays, n_samples)
    
    bins_a = torch.gather(bins_a, dim=-1, index=inds)
    bins_b = torch.gather(bins_b, dim=-1, index=inds)
    z_importance = bins_a + (bins_b - bins_a) * torch.rand_like(u)
    return z_importance


def mipnerf_render_rays(
    mipnerf: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    rays_o: torch.Tensor, rays_d: torch.Tensor, pixel_width: torch.Tensor, 
    *, 
    return_dict: bool = False,
    n_coarse: int = 64, n_fine: int = 64, uniform_ratio: float = 0.4,
    near: float = 0.1, far: float = 100.0,
    z_spacing: Literal['linear', 'inverse_linear'] = 'linear',
) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    MipNeRF rendering.

    Args:
        mipnerf: mipnerf model, which takes (points_mu, points_sigma) as input and returns (color, density) as output.
            The shape of points_mu and points_sigma should be (..., n_rays, n_samples, 3) and (..., n_rays, n_samples, 3, 3) respectively.
            The shape of color and density should be (..., n_rays, n_samples, 3) and (..., n_rays, n_samples) respectively.
        rays_o: (..., n_rays, 3) ray origins
        rays_d: (..., n_rays, 3) ray directions.
        pixel_width: (..., n_rays) pixel width. How to compute? pixel_width = 1 / (normalized focal length * width)
    
    Returns 
        if return_dict is False, return fine results only:
            rgb_fine: (..., n_rays, 3) rendered color values.
            depth_fine: (..., n_rays, 1) rendered depth values.
        else, return a dict like:
        {
            "coarse": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..},
            "fine": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..}
        }
    """
    # 1. Coarse: bin sampling
    z_coarse = bin_sample(rays_d.shape[:-1], n_coarse, near, far, spacing=z_spacing, device=rays_o.device, dtype=rays_o.dtype)
    points_mu_coarse, points_sigma_coarse = get_mipnerf_cones(rays_o, rays_d, z_coarse, pixel_width)
    ray_length = rays_d.norm(dim=-1)

    #    Query color and density
    color_coarse, density_coarse = mipnerf(points_mu_coarse, points_sigma_coarse)

    #    Volume rendering
    rgb_coarse, depth_coarse, weights_coarse = volume_rendering(color_coarse, density_coarse, z_coarse, ray_length)                                # (n_batch, n_views, n_rays, 3), (n_batch, n_views, n_rays, 1), (n_batch, n_views, n_rays, n_samples)

    if n_fine == 0:
        if return_dict:
            return {
                'coarse': {'rgb': rgb_coarse, 'depth': depth_coarse, 'weights': weights_coarse, 'z_vals': z_coarse, 'color': color_coarse, 'density': density_coarse},
                'fine': None,
            }
        else:
            return rgb_coarse, depth_coarse

    # 2. Fine: Importance sampling
    with torch.no_grad():
        weights_coarse = (1.0 - uniform_ratio) * weights_coarse + uniform_ratio / weights_coarse.shape[-1]
    z_fine = importance_sample(z_coarse, weights_coarse, n_fine)
    z_fine, _ = torch.sort(z_fine, dim=-2)
    points_mu_fine, points_sigma_fine = get_mipnerf_cones(rays_o, rays_d, z_fine, pixel_width)                                                           
    color_fine, density_fine = mipnerf(points_mu_fine, points_sigma_fine)  

    #   Volume rendering                    
    rgb_fine, depth_fine, weights_fine = volume_rendering(color_fine, density_fine, z_fine, ray_length)

    if return_dict:
        return {
            'coarse': {'rgb': rgb_coarse, 'depth': depth_coarse, 'weights': weights_coarse, 'z_vals': z_coarse, 'color': color_coarse, 'density': density_coarse},
            'fine': {'rgb': rgb_fine, 'depth': depth_fine, 'weights': weights_fine, 'z_vals': z_fine, 'color': color_fine, 'density': density_fine}
        }
    else:
        return rgb_fine, depth_fine


def nerf_render_rays(
    nerf: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    rays_o: torch.Tensor, rays_d: torch.Tensor,
    *, 
    return_dict: bool = False,
    n_coarse: int = 64, n_fine: int = 64,
    near: float = 0.1, far: float = 100.0,
    z_spacing: Literal['linear', 'inverse_linear'] = 'linear',
):
    # TODO
    pass
