import torch
from torch import Tensor
from typing import *

from .transforms import transform_points
from .utils import matrix_trace, vector_outer


__all__ = ['procrustes', 'affine_procrustes']


def procrustes(cov_yx: Tensor, cov_xx: Optional[Tensor] = None, cov_yy: Optional[Tensor] = None, mean_x: Optional[Tensor] = None, mean_y: Optional[Tensor] = None, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
    """
    Procrustes analysis to solve for scale `s`, rotation `R` and translation `t` such that `y_i ~= s R x_i + t`.

    Parameters
    ----
    - `cov_yx`: (..., 3, 3) covariance matrix between y and x points.
    - `cov_xx`: (..., 3, 3) covariance matrix of x points. If None, no scaling is solved.
    - `cov_yy`: (..., 3, 3) covariance matrix of y points. If None, no scaling is solved.
    - `mean_x`: (..., 3) mean of x points. If None, no translation is solved.
    - `mean_y`: (..., 3) mean of y points. If None, no translation is solved.

    Specifically, based on provided inputs:
    
    - To solve the rotation `R`, `cov_yx` must be given.
    - To solve the scale `s`, at least one of `cov_xx` and `cov_yy` must be given.
        - (Recommended) If both `cov_xx` and `cov_yy` are given, the scale will be solved by minimizing a symmetric cost:
            `||s R X + t - Y||_F^2 / ||Y||_F^2 + ||s R^T (Y - t)  - X||_F^2 / ||X||_F^2`
        - If only `cov_xx` is given, the scale will be solved by minimizing forward cost
            `||s R X  + t - Y||_F^2`
        - If only `cov_yy` is given, the scale will be solved by minimizing inverse cost 
            `||s R^T (Y - t)  - X||_F^2`
    - To solve the translation `t`, provide `mean_x` and `mean_y`.

    Returns
    ----
    - `s`: (...) scale factor. None if both cov_xx and cov_yy are None. 
    - `R`: (..., 3, 3) rotation matrix.
    - `t`: (..., 3) translation vector. None if mean_x or mean_y is None.
    """
    dtype = mean_x.dtype
    U, _, Vh = torch.linalg.svd(cov_yx)
    R = U @ Vh
    Vh[..., 2, :] *= torch.sign(torch.linalg.det(R))[..., None]
    R = U @ Vh
    if cov_xx is not None and cov_yy is None:
        s = matrix_trace(cov_yx @ R.swapaxes(-2, -1), axis1=-2, axis2=-1) / matrix_trace(cov_xx, axis1=-2, axis2=-1).clamp_min(eps)
    if cov_xx is None and cov_yy is not None:
        s = matrix_trace(cov_yy, dim1=-2, dim2=-1) / matrix_trace(cov_yx @ R.swapaxes(-2, -1), axis1=-2, axis2=-1).clamp_min(eps)
    elif cov_xx is not None and cov_yy is not None:
        x_fnorm = matrix_trace(cov_xx, dim1=-2, dim2=-1).clamp_min(eps)
        y_fnorm = matrix_trace(cov_yy, dim1=-2, dim2=-1).clamp_min(eps)
        s = torch.sqrt(y_fnorm / x_fnorm)
    else:
        s = None
    if mean_x is not None and mean_y is not None:
        if s is not None:
            t = mean_y - transform_points(mean_x, s[..., None, None] * R)
        else:
            t = mean_y - transform_points(mean_x, R)
    else:
        t = None
    return s, R, t


def affine_procrustes(cov_yx: Tensor, cov_xx: Tensor, cov_yy: Tensor, mean_x: Tensor, mean_y: Tensor, lam: float = 1e-2, niter: int = 8, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
    """
    Extended Procrustes analysis to solve for affine transformation `A` and translation `t` such that `y_i ~= A x_i + t`.

    NOTE: This function may be not differentiable due to the iterative solving process. Use with `torch.no_grad()` if you don't need gradients.

    Parameters
    ----
    - `cov_yx`: (..., 3, 3) covariance matrix between y
    - `cov_xx`: (..., 3, 3) covariance matrix of x points.
    - `cov_yy`: (..., 3, 3) covariance matrix of y
    - `mean_x`: (..., 3) mean of x points.
    - `mean_y`: (..., 3) mean of y points.
    - `lam`: rigidity regularization weight.
    - `gamma`: symmetricity regularization annealing factor.
    - `niter`: number of iterations for solving.

    Returns
    ----
    - `A`: (..., 3, 3) affine transformation matrix.
    - `t`: (..., 3) translation vector.
    """
    dtype = mean_x.dtype
    U, _, Vh = torch.linalg.svd(cov_yx)
    R = U @ Vh
    Vh[..., 2, :] *= torch.sign(torch.linalg.det(R))[..., None]
    R = U @ Vh
    tr_xx = matrix_trace(cov_xx, axis1=-2, axis2=-1).clamp_min(eps)
    tr_yy = matrix_trace(cov_yy, axis1=-2, axis2=-1).clamp_min(eps)

    cov_yx, cov_xy = cov_yx / tr_xx[..., None, None], cov_yx.swapaxes(-2, -1) / tr_yy[..., None, None]
    cov_xx, cov_yy = cov_xx / tr_xx[..., None, None], cov_yy / tr_yy[..., None, None]
    
    A, B = torch.zeros_like(R), torch.zeros_like(R)
    I = torch.eye(cov_yx.shape[-1], dtype=dtype, device=cov_yx.device)
    
    def _step(A, B, R, cov_yx, cov_xy, cov_xx, cov_yy, lam, gamma):
        A = (cov_yx + lam * R + gamma * B.swapaxes(-2, -1)) @ torch.linalg.inv(cov_xx + lam * I + gamma * (B @ B.swapaxes(-2, -1)))
        B = (cov_xy + lam * R.swapaxes(-2, -1) + gamma * A.swapaxes(-2, -1)) @ torch.linalg.inv(cov_yy + lam * I + gamma * (A @ A.swapaxes(-2, -1)))
        err = torch.square(A @ B - I).mean(axis=(-2, -1))
        return A, B, err
    
    not_converged = torch.argwhere(torch.ones(R.shape[:-2], dtype=torch.bool, device=R.device))
    for i in range(niter):
        gamma_i = 1.2 ** i - 1
        non_converged_indices = tuple(not_converged.T)
        A[non_converged_indices], B[non_converged_indices], err = _step(*(x[non_converged_indices] for x in (A, B, R, cov_yx, cov_xy, cov_xx, cov_yy)), lam, gamma_i)
        not_converged = not_converged[err >= 1e-6]
        if len(not_converged) == 0:
            break
    t = mean_y - transform_points(mean_x, A)
    return A, t
