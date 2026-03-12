import torch
from torch import Tensor
from typing import *

from .transforms import transform_points, make_affine_matrix
from .utils import matrix_trace, vector_outer


__all__ = ['kabasch', 'umeyama', 'affine_umeyama', 'solve_pose', 'segment_solve_pose']


import torch


class Kabsch(torch.autograd.Function):
    "Customized backward function for Kabsch (SVD) for rotation matrix from covariance matrix."
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, cov: Tensor, eps: float = 1e-12):
        U, S, Vh = torch.linalg.svd(cov)        
        d = torch.where(torch.linalg.det(U) * torch.linalg.det(Vh) >= 0, 1., -1.)
        s = torch.ones(cov.shape[:-2] + (3,), dtype=cov.dtype, device=cov.device)
        s[..., -1] = d
        R = U @ (s[..., :, None] * Vh)
        ctx.save_for_backward(S, Vh, R, d)
        ctx.eps = eps
        return R

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_R: Tensor):
        S, Vh, R, d = ctx.saved_tensors
        eps = ctx.eps
        
        V = Vh.transpose(-1, -2)
        M = R.transpose(-1, -2) @ grad_R
        
        M_skew_times_2 = M - M.transpose(-1, -2)
        
        V_M_skew_V = Vh @ M_skew_times_2 @ V
        
        S_mod = S.clone()
        S_mod[..., -1] *= d
        D = S_mod[..., :, None] + S_mod[..., None, :]
        D = D + eps
        
        Omega_hat = V_M_skew_V / D
        Omega = V @ Omega_hat @ Vh
        
        grad_cov = R @ Omega
        
        return grad_cov, None


def _kabasch_classic(cov: Tensor, eps: float = 1e-12):
    """Reference implementation. Would encounter NaN gradients when singular values are too close.
    """
    U, _, Vh = torch.linalg.svd(cov)
    det = torch.sign(torch.linalg.det(U) * torch.linalg.det(Vh))
    ones = torch.ones_like(det)
    R = U @ (torch.stack([ones, ones, det], dim=-1)[..., :, None] * Vh)
    return R


def kabasch(cov: Tensor, eps: float = 1e-12):
    """Backward gradients friendly Kabasch method (compute rotation from input covarience matrix).
    """
    return Kabsch.apply(cov, eps)


def umeyama(cov_yx: Tensor, cov_xx: Optional[Tensor] = None, cov_yy: Optional[Tensor] = None, mean_x: Optional[Tensor] = None, mean_y: Optional[Tensor] = None, eps: float = 1e-12) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Umeyama method to solve for scale `s`, rotation `R` and translation `t` such that `y_i ~= s R x_i + t`.

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
    R = kabasch(cov_yx)
    if cov_xx is not None and cov_yy is None:
        s = matrix_trace(cov_yx @ R.swapaxes(-2, -1), dim1=-2, dim2=-1) / matrix_trace(cov_xx, dim1=-2, dim2=-1).clamp_min(eps)
    if cov_xx is None and cov_yy is not None:
        s = matrix_trace(cov_yy, dim1=-2, dim2=-1) / matrix_trace(cov_yx @ R.swapaxes(-2, -1), dim1=-2, dim2=-1).clamp_min(eps)
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


def affine_umeyama(cov_yx: Tensor, cov_xx: Tensor, cov_yy: Tensor, mean_x: Tensor, mean_y: Tensor, lam: float = 1e-2, niter: int = 8, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
    """
    Extended Procrustes analysis to solve for affine transformation `A` and translation `t` such that `y_i ~= A x_i + t`.

    NOTE: This function may be indifferentiable due to the iterative solving process. Use with `torch.no_grad()` if you don't need gradients.

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
    R = kabasch(cov_yx)
    tr_xx = matrix_trace(cov_xx, dim1=-2, dim2=-1).clamp_min(eps)
    tr_yy = matrix_trace(cov_yy, dim1=-2, dim2=-1).clamp_min(eps)

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


def solve_pose(
    p: Tensor, 
    q: Tensor, 
    w: Optional[Tensor] = None, 
    *,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2, 
    niter: int = 5,
    eps: float = 1e-12
) -> Tensor:
    """Solve for the pose (transformation from p to q) given weighted point correspondences.
    
    Parameters
    ----
    - `p`: (..., N, 3) source points
    - `q`: (..., N, 3) target points
    - `w`: optional (..., N) weights for each point correspondence. If None, uniform weights are used.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed.
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `lam`: regularization weight for 'affine' mode.
    - `niter`: number of iterations for 'affine' mode.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `pose`: (..., 4, 4) transformations matrix from p to q.
    """
    if w is None:
        w = torch.ones(p.shape[:-1], dtype=p.dtype, device=p.device)
    w_sum = torch.sum(w, dim=-1).clamp_min(eps)
    p_mean = torch.sum(p * w[..., None], dim=-2) / w_sum[..., None]
    q_mean = torch.sum(q * w[..., None], dim=-2) / w_sum[..., None]
    p = p - p_mean[..., None, :]
    q = q - q_mean[..., None, :]
    pw = p * w[..., None]
    qw = q * w[..., None]
    cov_qp = torch.sum(vector_outer(qw, p), dim=-3) / w_sum[..., None, None]
    if mode == 'similar' or mode == 'affine':
        cov_pp = torch.sum(vector_outer(pw, p), dim=-3) / w_sum[..., None, None]
        cov_qq = torch.sum(vector_outer(qw, q), dim=-3) / w_sum[..., None, None]
    
    if mode == 'rigid':
        _, R, t = umeyama(cov_qp, mean_x=p_mean, mean_y=q_mean, eps=eps)
        pose = make_affine_matrix(R, t)
    elif mode == 'similar':
        s, R, t = umeyama(cov_qp, cov_xx=cov_pp, cov_yy=cov_qq, mean_x=p_mean, mean_y=q_mean, eps=eps)
        pose = make_affine_matrix(s * R, t)
    elif mode == 'affine':
        A, t = affine_umeyama(cov_qp, cov_pp, cov_qq, p_mean, q_mean, lam=lam, niter=niter, eps=eps)
        pose = make_affine_matrix(A, t)
    
    return pose


def segment_solve_pose(
    p: Tensor, 
    q: Tensor, 
    w: Optional[Tensor] = None, 
    *,
    offsets: Tensor, 
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2, 
    niter: int = 5,
    eps: float = 1e-12
) -> Tensor:
    """Solve for the pose (transformation from p to q: q ≈ pose @ p) given weighted point correspondences.

    NOTE: Affine mode is solved by iterative method and may be indifferentiable. Use with `torch.no_grad()` if you don't need gradients.
    
    Parameters
    ----
    - `p`: (N, 3) source points
    - `q`: (N, 3) target points
    - `w`: (N,) weights for each point correspondence
    - `offsets`: (S + 1,) segment offsets. Points in each segment belong to the same rigid / affine body.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed.
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `lam`: regularization weight for 'affine' mode.
    - `niter`: number of iterations for 'affine' mode.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `pose`: (S, 4, 4) transformations matrix from p to q.
    """
    if w is None:
        w = torch.ones(p.shape[:-1], device=p.device, dtype=p.dtype)

    lengths = torch.diff(offsets)
    w_sum = torch.segment_reduce(w, 'sum', offsets=offsets, axis=0).clamp_min(eps)
    p_mean = torch.segment_reduce(p * w[..., None], 'sum', offsets=offsets, axis=0) / w_sum[:, None]
    q_mean = torch.segment_reduce(q * w[..., None], 'sum', offsets=offsets, axis=0) / w_sum[:, None]
    p = p - torch.repeat_interleave(p_mean, lengths, dim=0)
    q = q - torch.repeat_interleave(q_mean, lengths, dim=0)
    pw = p * w[..., None]
    qw = q * w[..., None]
    cov_qp = torch.segment_reduce(vector_outer(qw, p), 'sum', offsets=offsets, axis=0) / w_sum[:, None, None]
    if mode == 'similar' or mode == 'affine':
        cov_pp = torch.segment_reduce(vector_outer(pw, p), 'sum', offsets=offsets, axis=0) / w_sum[:, None, None]
        cov_qq = torch.segment_reduce(vector_outer(qw, q), 'sum', offsets=offsets, axis=0) / w_sum[:, None, None]

    if mode == 'rigid':
        _, R, t = umeyama(cov_qp, mean_x=p_mean, mean_y=q_mean, eps=eps)
        pose = make_affine_matrix(R, t)
    elif mode == 'similar':
        s, R, t = umeyama(cov_qp, cov_xx=cov_pp, cov_yy=cov_qq, mean_x=p_mean, mean_y=q_mean, eps=eps)
        pose = make_affine_matrix(s * R, t)
    elif mode == 'affine':
        A, t = affine_umeyama(cov_qp, cov_pp, cov_qq, p_mean, q_mean, lam=lam, niter=niter, eps=eps)
        pose = make_affine_matrix(A, t)
    
    return pose
