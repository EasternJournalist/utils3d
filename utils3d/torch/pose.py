import math

import torch
from torch import Tensor
from typing import *

from .transforms import transform_points, make_affine_matrix
from .utils import matrix_trace, vector_outer


__all__ = ['kabsch', 'umeyama', 'affine_umeyama', 'solve_pose', 'solve_pose_ransac', 'segment_solve_pose', 'solve_poses_sequential', 'segment_solve_poses_sequential', 'pose_graph_edge_moments', 'segment_pose_graph_edge_moments', 'pose_graph_optimization', 'pose_graph_optimization_gnc']


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


def _kabsch_classic(cov: Tensor, eps: float = 1e-12):
    """Reference implementation. Would encounter NaN gradients when singular values are too close.
    """
    U, _, Vh = torch.linalg.svd(cov)
    det = torch.sign(torch.linalg.det(U) * torch.linalg.det(Vh))
    ones = torch.ones_like(det)
    R = U @ (torch.stack([ones, ones, det], dim=-1)[..., :, None] * Vh)
    return R


def kabsch(cov: Tensor, eps: float = 1e-12):
    """Backward gradients friendly Kabsch method (compute rotation from input covarience matrix).
    """
    return Kabsch.apply(cov, eps)


def safe_inv(A: Tensor) -> Tensor:
    """Batched matrix inverse that returns NaN for singular inputs instead of raising.

    `torch.linalg.inv` raises `LinAlgError` whenever any batch element is singular, which forces
    callers to either pre-validate inputs or wrap in try/except. NumPy inherited this behavior
    and PyTorch followed; for our pipelines we'd rather let NaN propagate (consistent with the
    rest of floating-point arithmetic), so degenerate elements simply mark themselves as invalid
    downstream without taking out the whole batch. Uses `torch.linalg.inv_ex` under the hood.
    """
    inv, info = torch.linalg.inv_ex(A)
    if info.any():
        inv = torch.where((info > 0)[..., None, None], torch.full_like(inv, float('nan')), inv)
    return inv


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
    R = kabsch(cov_yx)
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


class _SymSqrtInvSqrt(torch.autograd.Function):
    """Customized backward for the symmetric square root and inverse square root of SPD matrices.

    The naive path (differentiating through `torch.linalg.eigh`) produces unstable gradients when
    eigenvalues are close, because the eigenvector backward contains bare `1 / (L_i - L_j)` terms.
    For a symmetric matrix function `h(A) = V h(L) V^T` the gradient only needs the divided
    differences `(h(L_i) - h(L_j)) / (L_i - L_j)`, which for sqrt / inverse-sqrt have closed forms
    with no `L_i - L_j` in the denominator (see the Loewner matrices below). They are bounded by the
    `eps` floor, so they stay finite even for coincident eigenvalues (the diagonal `i == j` case is
    the same formula).
    """
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, mat: Tensor, eps: float = 1e-12):
        L, V = torch.linalg.eigh(mat)
        sqrt_L = L.clamp_min(eps).sqrt()
        inv_sqrt_L = 1.0 / sqrt_L
        mat_sqrt = (V * sqrt_L[..., None, :]) @ V.mT
        mat_inv_sqrt = (V * inv_sqrt_L[..., None, :]) @ V.mT
        ctx.save_for_backward(V, sqrt_L)
        return mat_sqrt, mat_inv_sqrt

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_sqrt: Tensor, grad_inv_sqrt: Tensor):
        V, sqrt_L = ctx.saved_tensors

        # Loewner (divided-difference) matrices, expressed purely via the clamped sqrt eigenvalues.
        # f(L) = sqrt(L)     -> Lf_ij = 1 / (s_i + s_j)
        # g(L) = 1 / sqrt(L) -> Lg_ij = -1 / (s_i s_j (s_i + s_j))
        si = sqrt_L[..., :, None]
        sj = sqrt_L[..., None, :]
        s_sum = si + sj
        loewner_f = 1.0 / s_sum
        loewner_g = -1.0 / (si * sj * s_sum)

        inner = torch.zeros(V.shape, dtype=V.dtype, device=V.device)
        if grad_sqrt is not None:
            grad_sqrt = 0.5 * (grad_sqrt + grad_sqrt.mT)
            inner = inner + loewner_f * (V.mT @ grad_sqrt @ V)
        if grad_inv_sqrt is not None:
            grad_inv_sqrt = 0.5 * (grad_inv_sqrt + grad_inv_sqrt.mT)
            inner = inner + loewner_g * (V.mT @ grad_inv_sqrt @ V)

        grad_mat = V @ inner @ V.mT
        return grad_mat, None


def _sym_sqrt_and_inv_sqrt(mat: Tensor, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
    """Symmetric square root and inverse square root of a batch of SPD matrices, from a single
    eigendecomposition, with gradient-stable custom backward (see `_SymSqrtInvSqrt`)."""
    return _SymSqrtInvSqrt.apply(mat, eps)


@torch.no_grad()
def _affine_umeyama_iterative(cov_yx: Tensor, cov_xx: Tensor, cov_yy: Tensor, mean_x: Tensor, mean_y: Tensor, lam: float = 1e-2, niter: int = 8, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
    """Reference implementation. Solves the inverse-consistency constraint `A B = I` by an annealed
    quadratic penalty + alternating least squares. Kept for correctness verification of `affine_umeyama`.
    """
    dtype = mean_x.dtype
    R = kabsch(cov_yx)
    tr_xx = matrix_trace(cov_xx, dim1=-2, dim2=-1).clamp_min(eps)
    tr_yy = matrix_trace(cov_yy, dim1=-2, dim2=-1).clamp_min(eps)

    cov_yx, cov_xy = cov_yx / tr_xx[..., None, None], cov_yx.swapaxes(-2, -1) / tr_yy[..., None, None]
    cov_xx, cov_yy = cov_xx / tr_xx[..., None, None], cov_yy / tr_yy[..., None, None]
    
    A, B = torch.zeros_like(R), torch.zeros_like(R)
    I = torch.eye(cov_yx.shape[-1], dtype=dtype, device=cov_yx.device)
    
    def _step(A, B, R, cov_yx, cov_xy, cov_xx, cov_yy, lam, gamma):
        A = (cov_yx + lam * R + gamma * B.swapaxes(-2, -1)) @ safe_inv(cov_xx + lam * I + gamma * (B @ B.swapaxes(-2, -1)))
        B = (cov_xy + lam * R.swapaxes(-2, -1) + gamma * A.swapaxes(-2, -1)) @ safe_inv(cov_yy + lam * I + gamma * (A @ A.swapaxes(-2, -1)))
        err = torch.square(A @ B - I).mean(axis=(-2, -1))
        return A, B, err
    
    not_converged = torch.argwhere(torch.ones(R.shape[:-2], dtype=torch.bool, device=R.device))
    for i in range(niter):
        gamma_i = 1.2 ** i - 1
        non_converged_indices = tuple(not_converged.T)
        A[non_converged_indices], B[non_converged_indices], err = _step(*(x[non_converged_indices] for x in (A, B, R, cov_yx, cov_xy, cov_xx, cov_yy)), lam, gamma_i)
        # NaN err (from degenerate inputs that produced NaN via safe_inv) compares False with
        # `>= 1e-6`, so those segments naturally drop out of `not_converged`.
        not_converged = not_converged[err >= 1e-6]
        if len(not_converged) == 0:
            break
    t = mean_y - transform_points(mean_x, A)
    return A, t


def affine_umeyama(cov_yx: Tensor, cov_xx: Tensor, cov_yy: Tensor, mean_x: Tensor, mean_y: Tensor, lam: float = 1e-2, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
    """
    Extended Procrustes analysis to solve for affine transformation `A` and translation `t` such that `y_i ~= A x_i + t`.

    The inverse-consistency constraint (the inverse map `A^{-1}` should align `y` back onto `x`) is
    satisfied *exactly* in closed form by whitening both point clouds to unit covariance and solving
    an orthogonal Procrustes problem in the whitened space, where the optimal map is a rotation `Q`
    (so `(A^{-1})` is automatically the consistent inverse):

        `A = cov_yy^{1/2} @ Q @ cov_xx^{-1/2}`,   `Q = polar(cov_yy^{-1/2} @ cov_yx @ cov_xx^{-1/2})`

    No iteration, no penalty annealing, and the result is differentiable.

    Parameters
    ----
    - `cov_yx`: (..., 3, 3) covariance matrix between y and x points.
    - `cov_xx`: (..., 3, 3) covariance matrix of x points.
    - `cov_yy`: (..., 3, 3) covariance matrix of y points.
    - `mean_x`: (..., 3) mean of x points.
    - `mean_y`: (..., 3) mean of y points.
    - `lam`: rigidity regularization weight. Shrinks the whitening toward isotropic, biasing `A`
        toward a similarity (rotation + uniform scale) transform and stabilizing the inverse sqrt.
    - `eps`: small value to clamp eigenvalues / prevent division by zero.

    Returns
    ----
    - `A`: (..., 3, 3) affine transformation matrix.
    - `t`: (..., 3) translation vector.
    """
    n = cov_xx.shape[-1]
    I = torch.eye(n, dtype=cov_xx.dtype, device=cov_xx.device)
    tr_xx = matrix_trace(cov_xx, dim1=-2, dim2=-1).clamp_min(eps)
    tr_yy = matrix_trace(cov_yy, dim1=-2, dim2=-1).clamp_min(eps)

    # Mild rigidity / numerical ridge: shrink the whitening toward isotropic.
    reg_xx = cov_xx + lam * (tr_xx / n)[..., None, None] * I
    reg_yy = cov_yy + lam * (tr_yy / n)[..., None, None] * I

    _, cov_xx_inv_sqrt = _sym_sqrt_and_inv_sqrt(reg_xx, eps)
    cov_yy_sqrt, cov_yy_inv_sqrt = _sym_sqrt_and_inv_sqrt(reg_yy, eps)

    M = cov_yy_inv_sqrt @ cov_yx @ cov_xx_inv_sqrt
    U, _, Vh = torch.linalg.svd(M)
    Q = U @ Vh

    A = cov_yy_sqrt @ Q @ cov_xx_inv_sqrt
    t = mean_y - transform_points(mean_x, A)
    return A, t


def solve_pose(
    p: Tensor, 
    q: Tensor, 
    w: Optional[Tensor] = None, 
    sigma: Optional[Tensor] = None, 
    *,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2, 
    eps: float = 1e-12
) -> Tensor:
    """Solve for the pose (transformation from p to q) given weighted point correspondences.

    Minimizes `sum_i w_i (||pose @ p_i - q_i|| / sigma_i)^2`.

    Parameters
    ----
    - `p`: (..., N, 3) source points
    - `q`: (..., N, 3) target points
    - `w`: optional (..., N) per-point confidence weight. If None, uniform weights are used.
    - `sigma`: optional (..., N) per-point noise scale; contributes `1 / sigma_i^2` to the weight (only
        relative values matter). If None, treated as 1. E.g. for depth-proportional noise pass `sigma = ||p_i||`.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed.
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `lam`: regularization weight for 'affine' mode.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `pose`: (..., 4, 4) transformations matrix from p to q.
    """
    if w is None:
        w = torch.ones(p.shape[:-1], dtype=p.dtype, device=p.device)
    if sigma is not None:
        sigma = torch.as_tensor(sigma, dtype=p.dtype, device=p.device)
        w = w / sigma.square().clamp_min(eps)
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
        pose = make_affine_matrix(s[..., None, None] * R, t)
    elif mode == 'affine':
        A, t = affine_umeyama(cov_qp, cov_pp, cov_qq, p_mean, q_mean, lam=lam, eps=eps)
        pose = make_affine_matrix(A, t)
    
    return pose


def solve_pose_ransac(
    p: Tensor, 
    q: Tensor, 
    w: Optional[Tensor] = None, 
    sigma: Optional[Tensor] = None, 
    *,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    threshold: Union[float, Tensor] = 0.05,
    num_samples: int = 32,
    sample_size: Optional[int] = None,
    lam: float = 1e-2, 
    eps: float = 1e-12,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Robustly solve for the pose (transformation from p to q) given point correspondences using RANSAC.

    Hypotheses are sampled from minimal subsets, scored by a truncated soft-inlier cost, and the best
    one is refit on all of its inliers. Vectorized over hypotheses and leading batch dimensions.

    The solve minimizes `sum_i w_i (||pose @ p_i - q_i|| / sigma_i)^2` (as in `solve_pose`), while the
    inlier test is the purely geometric `||pose @ p_i - q_i|| < threshold_i`. 

    The best hypothesis is selected by minimizing `sum_i w_i * min(1, ||pose @ p_i - q_i|| / threshold_i)` (a truncated soft-inlier cost).

    Parameters
    ----
    - `p`: (..., N, 3) source points
    - `q`: (..., N, 3) target points
    - `w`: optional (..., N) per-point confidence weight. Biases the hypothesis sampling (drawn
        proportional to `w`), weights the solve and the consensus score; does not relax the
        threshold. If None, uniform weights are used.
    - `sigma`: optional (..., N) per-point noise scale used in the solve weighting `w_i / sigma_i^2`
        (same meaning as in `solve_pose`). If None, treated as 1.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed.
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `threshold`: inlier distance threshold (scalar or per-point tensor, broadcastable to (..., N)). A
        correspondence is an inlier when `||pose @ p_i - q_i|| < threshold_i`. For a relative tolerance
        pass `relative_threshold * ||p_i||`.
    - `num_samples`: number of RANSAC hypotheses per batch element. Compute/memory scale linearly with it.
    - `sample_size`: size of each minimal sample. If None, defaults to 3 for 'rigid'/'similar' and 4 for 'affine'.
    - `lam`: regularization weight for 'affine' mode.
    - `eps`: small value to prevent division by zero.
    - `generator`: optional random generator for reproducible sampling.

    Returns
    ----
    - `pose`: (..., 4, 4) transformations matrix from p to q.
    - `inliers`: (..., N) boolean mask of inliers w.r.t. the returned pose.
    """
    if sample_size is None:
        sample_size = 4 if mode == 'affine' else 3
    batch_shape = p.shape[:-2]
    N = p.shape[-2]
    B = math.prod(batch_shape)

    p_flat = p.reshape(B, N, 3)
    q_flat = q.reshape(B, N, 3)
    if w is None:
        w_flat = torch.ones((B, N), dtype=p.dtype, device=p.device)
    else:
        w_flat = w.reshape(B, N)

    # Optional per-point noise scale, passed through to every solve (identical meaning to solve_pose).
    if sigma is not None:
        sigma = torch.as_tensor(sigma, dtype=p.dtype, device=p.device)
        sigma_flat = sigma.expand(p.shape[:-1]).reshape(B, N)  # (B, N)
    else:
        sigma_flat = None

    # Inlier threshold: a purely geometric gate (scalar or per-point), never folded into the solve.
    threshold = torch.as_tensor(threshold, dtype=p.dtype, device=p.device).clamp_min(eps)
    threshold_b = threshold.expand(p.shape[:-1]).reshape(B, N)[:, None, :] if threshold.ndim > 0 else threshold  # (B, 1, N) or scalar

    # Draw `num_samples` minimal subsets without replacement per batch element, sampling each point
    # with probability proportional to its confidence `w`, so high-confidence correspondences are
    # more likely to seed a hypothesis. Weights are floored to `eps` so a draw is always possible
    # even when fewer than `sample_size` points have nonzero weight. Uniform `w` -> uniform sampling.
    probs = w_flat.clamp_min(eps)[:, None, :].expand(B, num_samples, N).reshape(B * num_samples, N)
    idx = torch.multinomial(probs, sample_size, replacement=False, generator=generator).reshape(B, num_samples, sample_size)  # (B, num_samples, sample_size)
    p_s = torch.take_along_dim(p_flat[:, None, :, :], idx[..., None], dim=2)  # (B, num_samples, sample_size, 3)
    q_s = torch.take_along_dim(q_flat[:, None, :, :], idx[..., None], dim=2)
    w_s = torch.take_along_dim(w_flat[:, None, :], idx, dim=2)
    sigma_s = torch.take_along_dim(sigma_flat[:, None, :], idx, dim=2) if sigma_flat is not None else None

    # Solve a candidate pose for every hypothesis.
    pose_h = solve_pose(p_s, q_s, w_s, sigma_s, mode=mode, lam=lam, eps=eps)  # (B, num_samples, 4, 4)

    # Score hypotheses by a robust, truncated soft-inlier cost. The threshold normalizes the raw
    # residual (purely geometric); the confidence `w` weights each point's contribution but does
    # NOT relax its threshold. Each correspondence contributes `w_i * min(1, residual_i / threshold_i)`.
    p_t = transform_points(p_flat[:, None, :, :], pose_h[:, :, None, :, :])     # (B, num_samples, N, 3)
    residual = torch.linalg.norm(p_t - q_flat[:, None, :, :], dim=-1)           # (B, num_samples, N)
    normalized_residual = residual / threshold_b                                # (B, num_samples, N)
    inliers = normalized_residual < 1.0  # (B, num_samples, N)
    inlier_cost = (w_flat[:, None, :] * normalized_residual.clamp_max(1.0)).mean(dim=-1)  # (B, num_samples)

    best = torch.argmin(inlier_cost, dim=-1)  # (B,)
    best_inliers = torch.take_along_dim(inliers, best[:, None, None], dim=1)[:, 0]  # (B, N)

    # Refit on all inliers of the best hypothesis (same w / sigma weighting as the hypotheses).
    refit_w = w_flat * best_inliers.to(w_flat.dtype)
    pose = solve_pose(p_flat, q_flat, refit_w, sigma_flat, mode=mode, lam=lam, eps=eps)  # (B, 4, 4)

    pose = pose.reshape(*batch_shape, 4, 4)
    best_inliers = best_inliers.reshape(*batch_shape, N)
    return pose, best_inliers


def segment_solve_pose(
    p: Tensor, 
    q: Tensor, 
    w: Optional[Tensor] = None, 
    sigma: Optional[Tensor] = None, 
    *,
    offsets: Tensor, 
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2, 
    eps: float = 1e-12
) -> Tensor:
    """Solve for the pose (transformation from p to q: q ≈ pose @ p) given weighted point correspondences.

    Minimizes `sum_i (w_i / sigma_i^2) ||pose @ p_i - q_i||^2` within each segment (see `solve_pose`).
    
    Parameters
    ----
    - `p`: (N, 3) source points
    - `q`: (N, 3) target points
    - `w`: (N,) weights for each point correspondence
    - `sigma`: optional (N,) per-point noise scale. Effective weight is `w_i / sigma_i^2`. If None, treated as 1.
    - `offsets`: (S + 1,) segment offsets. Points in each segment belong to the same rigid / affine body.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed.
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `lam`: regularization weight for 'affine' mode.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `pose`: (S, 4, 4) transformations matrix from p to q.
    """
    if w is None:
        w = torch.ones(p.shape[:-1], device=p.device, dtype=p.dtype)
    if sigma is not None:
        sigma = torch.as_tensor(sigma, dtype=p.dtype, device=p.device)
        w = w / sigma.square().clamp_min(eps)

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
        pose = make_affine_matrix(s[..., None, None] * R, t)
    elif mode == 'affine':
        A, t = affine_umeyama(cov_qp, cov_pp, cov_qq, p_mean, q_mean, lam=lam, eps=eps)
        pose = make_affine_matrix(A, t)
    
    return pose


def solve_poses_sequential(
    trajectories: Tensor,
    weights: Optional[Tensor] = None,
    noise_scales: Optional[Tensor] = None,
    *,
    accum: Optional[Tuple[Tensor, ...]] = None,
    min_valid_size: int = 3,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor, Tensor, Tuple[Tensor, ...]]:
    """
    Given trajectories of points over time, sequentially solve for the poses (transformations from canonical to each frame) of each body at each frame.

    Parameters
    ----
    - `trajectories`: (T, ..., N, 3) posed points. T is number of frames. `...` is optional batch dimensions. N is number of points per group.
    - `weights`: (T, ..., N) quardratic error term weights for each point at each frame
    - `noise_scales`: (T, ..., N) optional per-point noise scale per frame. The effective weight is
        `weights / noise_scales^2`. If None, treated as 1.
    - `accum`: accumulated statistics from previous calls. If None, start fresh.
    - `min_valid_size`: minimum number of valid points in each frame to consider the segment / group valid.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed.
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `lam`: rigidity regularization weight for 'affine' mode.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `poses`: (T, ..., 4, 4) transformations from canonical to each frame.
    - `valid`: (T, ...) boolean mask indicating valid segments
    - `stats`: canonical statistics of each group `(mu, cov, tot_w, nnz)`.
    - `canonical_points`: (..., N, 3) canonical points.
    - `err`: (..., N) per-point RMS error over all time.
    - `accum`: per-point accumulated statistics. Pass it to the next call for incremental solving.
    """
    dtype = trajectories.dtype
    device = trajectories.device
    num_frames = trajectories.shape[0]
    num_points = trajectories.shape[-2]
    batch_shape = trajectories.shape[1:-2]

    if weights is None:
        weights = torch.ones((num_frames, *batch_shape, num_points), dtype=dtype, device=device)
    if noise_scales is not None:
        noise_scales = torch.as_tensor(noise_scales, dtype=dtype, device=device)
        weights = weights / noise_scales.square().clamp_min(eps)

    poses = torch.zeros((num_frames, *batch_shape, 4, 4), dtype=dtype, device=device)

    if accum is not None:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = [a.clone() for a in accum]
    else:
        accum_sqrtw = torch.zeros((*batch_shape, num_points), dtype=dtype, device=device)
        accum_sqrtwx = torch.zeros((*batch_shape, num_points, 3), dtype=dtype, device=device)
        accum_sqrtwxx = torch.zeros((*batch_shape, num_points, 3, 3), dtype=dtype, device=device)
        accum_w = torch.zeros((*batch_shape, num_points), dtype=dtype, device=device)
        accum_wx = torch.zeros((*batch_shape, num_points, 3), dtype=dtype, device=device)
        accum_wxx = torch.zeros((*batch_shape, num_points, 3, 3), dtype=dtype, device=device)
        accum_nnz = torch.zeros((*batch_shape, num_points), dtype=dtype, device=device)

    for i in range(num_frames):
        mean_sqrtwx = accum_sqrtwx / accum_sqrtw.clamp_min(eps)[..., None]
        wi, yi = weights[i], trajectories[i]
        sqrtwi = torch.sqrt(wi)
        w = sqrtwi * accum_sqrtw
        sum_w = torch.sum(w, dim=-1).clamp_min(eps)
        center_x = torch.sum(sqrtwi[..., None] * accum_sqrtwx, dim=-2) / sum_w[..., None]
        center_y = torch.sum(w[..., None] * yi, dim=-2) / sum_w[..., None]
        xc = mean_sqrtwx - center_x[..., None, :]
        yc = yi - center_y[..., None, :]
        cov_yx = torch.einsum('...i,...ij,...ik->...jk', w, yc, xc) / sum_w[..., None, None]
        if mode == 'affine' or mode == 'similar':
            cov_xx = (torch.einsum('...i,...ij,...ik->...jk', w, xc, xc) + torch.einsum('...i,...ijk->...jk', sqrtwi, accum_sqrtwxx)) / sum_w[..., None, None]
            cov_yy = torch.einsum('...i,...ij,...ik->...jk', w, yc, yc) / sum_w[..., None, None]

        if mode == 'rigid':
            _, R, t = umeyama(cov_yx, mean_x=center_x, mean_y=center_y, eps=eps)
            poses[i] = make_affine_matrix(R, t)
        elif mode == 'similar':
            s, R, t = umeyama(cov_yx, cov_xx=cov_xx, mean_x=center_x, mean_y=center_y, eps=eps)
            poses[i] = make_affine_matrix(s[..., None, None] * R, t)
        elif mode == 'affine':
            A, t = affine_umeyama(cov_yx, cov_xx, cov_yy, center_x, center_y, lam=lam, eps=eps)
            poses[i] = make_affine_matrix(A, t)

        xi = transform_points(yi, safe_inv(poses[i])[..., None, :, :])

        # Update accum
        old_mean_sqrtwx, old_accum_sqrtw = mean_sqrtwx.clone(), accum_sqrtw.clone()
        accum_sqrtw = accum_sqrtw + sqrtwi
        accum_sqrtwx = accum_sqrtwx + sqrtwi[..., None] * xi
        mean_sqrtwx = accum_sqrtwx / accum_sqrtw.clamp_min(eps)[..., None]
        accum_sqrtwxx = accum_sqrtwxx + old_accum_sqrtw[..., None, None] * vector_outer(mean_sqrtwx - old_mean_sqrtwx) + sqrtwi[..., None, None] * vector_outer(xi - mean_sqrtwx)

        mean_wx = accum_wx / accum_w.clamp_min(eps)[..., None]
        old_mean_wx, old_accum_w = mean_wx.clone(), accum_w.clone()
        accum_w = accum_w + wi
        accum_wx = accum_wx + wi[..., None] * xi
        mean_wx = accum_wx / accum_w.clamp_min(eps)[..., None]
        accum_wxx = accum_wxx + old_accum_w[..., None, None] * vector_outer(mean_wx - old_mean_wx) + wi[..., None, None] * vector_outer(xi - mean_wx)
        accum_nnz = accum_nnz + (wi > 0).to(dtype)

    tot_w = torch.sum(accum_w, dim=-1)
    mu = torch.sum(accum_wx, dim=-2) / tot_w.clamp_min(eps)[..., None]
    mean_wx = accum_wx / accum_w.clamp_min(eps)[..., None]
    sigma = torch.sum(accum_wxx + accum_w[..., None, None] * vector_outer(mu[..., None, :] - mean_wx), dim=-3) / tot_w.clamp_min(eps)[..., None, None]
    nnz = torch.sum(accum_nnz, dim=-1)
    valid = torch.sum(weights > 0, dim=-1) >= min_valid_size
    err = torch.sqrt(matrix_trace(accum_wxx, dim1=-2, dim2=-1) / accum_nnz.clamp_min(eps))

    return poses, valid, (mu, sigma, tot_w, nnz), mean_wx, err, (accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz)


def segment_solve_poses_sequential(
    trajectories: Tensor,
    weights: Optional[Tensor] = None,
    offsets: Tensor = None,
    noise_scales: Optional[Tensor] = None,
    *,
    accum: Optional[Tuple[Tensor, ...]] = None,
    min_valid_size: int = 3,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor, Tensor, Tuple[Tensor, ...]]:
    """
    Segment array mode for `solve_poses_sequential`.

    Parameters
    ----
    - `trajectories`: (T, N, 3) posed points.
    - `weights`: (T, N) quardratic error term weights for each point at each frame
    - `offsets`: (S + 1,) segment offsets. Points in each segment belong to the same rigid / affine body.
    - `noise_scales`: (T, N) optional per-point noise scale per frame. The effective weight is
        `weights / noise_scales^2`. If None, treated as 1.
    - `accum`: accumulated statistics from previous calls. If None, start fresh.
    - `min_valid_size`: minimum number of valid points in each frame to consider the segment / group valid.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
    - `lam`: rigidity regularization weight for 'affine' mode.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `poses`: (T, S, 4, 4) transformations from canonical to each frame.
    - `valid`: (T, S) boolean mask indicating valid segments.
    - `stats`: canonical statistics `(mu, cov, tot_w, nnz)`.
    - `canonical_points`: (N, 3) canonical points.
    - `err`: (N,) per-point RMS error over all time.
    - `accum`: per-point accumulated statistics for incremental solving.
    """
    dtype = trajectories.dtype
    device = trajectories.device
    num_frames = trajectories.shape[0]
    num_points = trajectories.shape[1]

    if weights is None:
        weights = torch.ones((num_frames, num_points), dtype=dtype, device=device)
    if noise_scales is not None:
        noise_scales = torch.as_tensor(noise_scales, dtype=dtype, device=device)
        weights = weights / noise_scales.square().clamp_min(eps)

    num_segments = offsets.shape[0] - 1
    lengths = torch.diff(offsets)
    poses = torch.zeros((num_frames, num_segments, 4, 4), dtype=dtype, device=device)

    if accum is not None:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = [a.clone() for a in accum]
    else:
        accum_sqrtw = torch.zeros((num_points,), dtype=dtype, device=device)
        accum_sqrtwx = torch.zeros((num_points, 3), dtype=dtype, device=device)
        accum_sqrtwxx = torch.zeros((num_points, 3, 3), dtype=dtype, device=device)
        accum_w = torch.zeros((num_points,), dtype=dtype, device=device)
        accum_wx = torch.zeros((num_points, 3), dtype=dtype, device=device)
        accum_wxx = torch.zeros((num_points, 3, 3), dtype=dtype, device=device)
        accum_nnz = torch.zeros((num_points,), dtype=dtype, device=device)

    for i in range(num_frames):
        mean_sqrtwx = accum_sqrtwx / accum_sqrtw.clamp_min(eps)[:, None]
        wi, yi = weights[i], trajectories[i]
        sqrtwi = torch.sqrt(wi)
        w = sqrtwi * accum_sqrtw
        sum_w = torch.segment_reduce(w, 'sum', offsets=offsets, axis=0).clamp_min(eps)
        center_x = torch.segment_reduce(sqrtwi[:, None] * accum_sqrtwx, 'sum', offsets=offsets, axis=0) / sum_w[:, None]
        center_y = torch.segment_reduce(w[:, None] * yi, 'sum', offsets=offsets, axis=0) / sum_w[:, None]
        center_x_broadcast = torch.repeat_interleave(center_x, lengths, dim=0)
        center_y_broadcast = torch.repeat_interleave(center_y, lengths, dim=0)
        xc = mean_sqrtwx - center_x_broadcast
        yc = yi - center_y_broadcast
        cov_yx = torch.segment_reduce(w[:, None, None] * vector_outer(yc, xc), 'sum', offsets=offsets, axis=0) / sum_w[:, None, None]
        if mode == 'affine' or mode == 'similar':
            cov_xx = torch.segment_reduce(sqrtwi[:, None, None] * accum_sqrtwxx + w[:, None, None] * vector_outer(xc), 'sum', offsets=offsets, axis=0) / sum_w[:, None, None]
            cov_yy = torch.segment_reduce(w[:, None, None] * vector_outer(yc), 'sum', offsets=offsets, axis=0) / sum_w[:, None, None]

        if mode == 'rigid':
            _, R, t = umeyama(cov_yx, mean_x=center_x, mean_y=center_y, eps=eps)
            poses[i] = make_affine_matrix(R, t)
        elif mode == 'similar':
            s, R, t = umeyama(cov_yx, cov_xx=cov_xx, mean_x=center_x, mean_y=center_y, eps=eps)
            poses[i] = make_affine_matrix(s[..., None, None] * R, t)
        elif mode == 'affine':
            A, t = affine_umeyama(cov_yx, cov_xx, cov_yy, center_x, center_y, lam=lam, eps=eps)
            poses[i] = make_affine_matrix(A, t)

        xi = transform_points(yi, torch.repeat_interleave(safe_inv(poses[i]), lengths, dim=0))

        # Update accum
        old_mean_sqrtwx, old_accum_sqrtw = mean_sqrtwx.clone(), accum_sqrtw.clone()
        accum_sqrtw = accum_sqrtw + sqrtwi
        accum_sqrtwx = accum_sqrtwx + sqrtwi[:, None] * xi
        mean_sqrtwx = accum_sqrtwx / accum_sqrtw.clamp_min(eps)[:, None]
        accum_sqrtwxx = accum_sqrtwxx + old_accum_sqrtw[:, None, None] * vector_outer(mean_sqrtwx - old_mean_sqrtwx) + sqrtwi[:, None, None] * vector_outer(xi - mean_sqrtwx)

        mean_wx = accum_wx / accum_w.clamp_min(eps)[:, None]
        old_mean_wx, old_accum_w = mean_wx.clone(), accum_w.clone()
        accum_w = accum_w + wi
        accum_wx = accum_wx + wi[:, None] * xi
        mean_wx = accum_wx / accum_w.clamp_min(eps)[:, None]
        accum_wxx = accum_wxx + old_accum_w[:, None, None] * vector_outer(mean_wx - old_mean_wx) + wi[:, None, None] * vector_outer(xi - mean_wx)
        accum_nnz = accum_nnz + (wi > 0).to(dtype)

    tot_w = torch.segment_reduce(accum_w, 'sum', offsets=offsets, axis=0)
    mu = torch.segment_reduce(accum_wx, 'sum', offsets=offsets, axis=0) / tot_w.clamp_min(eps)[:, None]
    mean_wx = accum_wx / accum_w.clamp_min(eps)[:, None]
    mu_broadcast = torch.repeat_interleave(mu, lengths, dim=0)
    sigma = torch.segment_reduce(accum_wxx + accum_w[:, None, None] * vector_outer(mu_broadcast - mean_wx), 'sum', offsets=offsets, axis=0) / tot_w.clamp_min(eps)[:, None, None]
    nnz = torch.segment_reduce(accum_nnz, 'sum', offsets=offsets, axis=0)
    # `torch.segment_reduce` requires axis == offsets.ndim - 1, so reduce along dim 0 by transposing.
    valid = torch.segment_reduce((weights > 0).to(dtype).transpose(0, 1).contiguous(), 'sum', offsets=offsets, axis=0).transpose(0, 1) >= min_valid_size
    err = torch.sqrt(matrix_trace(accum_wxx, dim1=-2, dim2=-1) / accum_nnz.clamp_min(eps))

    return poses, valid, (mu, sigma, tot_w, nnz), mean_wx, err, (accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz)


def _pose_graph_optimization_construct_laplacian(edge: Tensor, num_nodes: int, R: Tensor, w: Tensor, s: Optional[Tensor] = None) -> Tensor:
    # Connection Laplacian for the per-edge block `M_ij = s_ij R_ij` (`s_ij = 1` in the orthonormal /
    # rigid case, `s = None`). For each edge i->j it contributes:
    # - `-w_ij * M_ij^T` to block (i, j)
    # - `-w_ij * M_ij` to block (j, i),
    # - `w_ij * s_ij^2 * I` to diagonal block (i, i) [i is the source: `M_ij^T M_ij = s_ij^2 I`],
    # - `w_ij * I` to diagonal block (j, j) [j is the target].
    if s is None:
        w_src, w_dst = w, w
        edge_elements = (-w[:, None, None] * R.mT).reshape(-1)   # -w_ij R_ij^T, (E * 3 * 3,)
    else:
        w_src, w_dst = w * s.square(), w
        edge_elements = (-(w * s)[:, None, None] * R.mT).reshape(-1)   # -w_ij s_ij R_ij^T
    w_agg = torch.zeros(num_nodes, device=R.device, dtype=R.dtype).index_add(0, edge[:, 0], w_src).index_add(0, edge[:, 1], w_dst)
    diag_elements = w_agg.repeat_interleave(3)   # (N * 3,)
    laplacian_data = torch.cat([diag_elements, edge_elements.reshape(-1), edge_elements.reshape(-1)], dim=0)

    local3 = torch.arange(3, device=R.device)
    local3x3 = torch.stack(torch.meshgrid(local3, local3, indexing='ij'), dim=-1)   # to get the local 3x3 block coordinates for each edge
    diag_coords = (torch.arange(num_nodes, device=R.device)[:, None].expand(num_nodes, 3) * 3 + local3[None, :]).reshape(-1, 1).expand(-1, 2)   # (N * 3, 2)
    edge_coords = (edge[:, None, None, :] * 3 + local3x3[None, :, :, :]).reshape(-1, 2)                       # (E * 3 * 3, 2)
    laplacian_coords = torch.cat([diag_coords, edge_coords, edge_coords.flip(-1)], dim=0)               # (2, N * 3 + 2 * E * 3 * 3)

    laplacian = torch.sparse_coo_tensor(laplacian_coords.T, laplacian_data, size=(num_nodes * 3, num_nodes * 3), check_invariants=False)
    return laplacian


def _pose_graph_optimization_scale_sync(edges: Tensor, num_nodes: int, s_relative: Tensor, w: Tensor, eps: float = 1e-12) -> Tensor:
    """Global per-node scales `s_i` from per-edge relative scales `s_ij ≈ s_j / s_i`.

    In log-space `log s_ij = log s_j - log s_i` is linear, so the scales follow from a weighted scalar
    graph-Laplacian least squares `min sum_ij w_ij (l_j - l_i - log s_ij)^2` (`l_i = log s_i`). The
    overall scale is a free gauge (the all-ones null space); the min-norm `lstsq` solution fixes it to
    `sum_i l_i = 0`, i.e. the geometric mean of the node scales is 1.
    """
    log_s = torch.log(s_relative.clamp_min(eps))
    src, dst = edges[:, 0], edges[:, 1]
    # Dense scalar Laplacian L = D - A (N x N; N is small, like the translation solve below).
    laplacian = torch.zeros((num_nodes, num_nodes), device=s_relative.device, dtype=s_relative.dtype)
    laplacian.index_put_((src, dst), -w, accumulate=True)
    laplacian.index_put_((dst, src), -w, accumulate=True)
    laplacian.index_put_((src, src), w, accumulate=True)
    laplacian.index_put_((dst, dst), w, accumulate=True)
    wl = w * log_s
    b = torch.zeros(num_nodes, device=s_relative.device, dtype=s_relative.dtype).index_add(0, dst, wl).index_add(0, src, -wl)
    log_s_global = torch.linalg.lstsq(laplacian, b).solution
    return torch.exp(log_s_global)



def _robust_smallest_eigenvectors(L: Tensor, k: int = 3) -> Tensor:
    """The `k` eigenvectors of the smallest eigenvalues of a symmetric PSD matrix, robust to solver
    non-convergence.

    `torch.linalg.eigh`'s divide-and-conquer driver (LAPACK `syevd` / cuSOLVER `syevd`) can fail to
    converge (error 97) on ill-conditioned matrices with many (near-)repeated eigenvalues — exactly the
    Laplacians produced by weakly-connected / degenerate pose graphs. We escalate through:
      1. `eigh` with a growing diagonal jitter (relative to the matrix magnitude). The shift leaves the
         eigenvectors essentially unchanged but breaks the exact degeneracy that trips the driver.
      2. The same on CPU (LAPACK is often more forgiving than cuSOLVER here).
      3. `svd`, which uses a different algorithm and yields the same eigenpairs for a symmetric PSD
         matrix (singular values = eigenvalues, descending), as a last resort.
    Returns the `k` eigenvectors in ascending-eigenvalue order, on `L`'s original device.
    """
    n = L.shape[-1]
    scale = torch.diagonal(L, dim1=-2, dim2=-1).abs().max().clamp_min(1.0)
    for device in (L.device, torch.device('cpu')):
        Ld = L.to(device)
        eye = torch.eye(n, device=device, dtype=L.dtype)
        for jitter in (0.0, 1e-9, 1e-7, 1e-5, 1e-3):
            try:
                _, eigenvectors = torch.linalg.eigh(Ld + (jitter * scale) * eye)
                return eigenvectors[:, :k].to(L.device)
            except torch._C._LinAlgError:
                continue
    # SVD fallback (different, more robust algorithm). For symmetric PSD L, U columns are eigenvectors
    # and singular values are the eigenvalues in *descending* order, so the smallest-eigenvalue vectors
    # are the last k columns; flip to return them in ascending order.
    U, _, _ = torch.linalg.svd(L.to('cpu'))
    return U[:, -k:].flip(-1).to(L.device)


def _pose_graph_optimization_eigen_decomposition(laplacian: Tensor) -> Tensor:
    # Dense `eigh` rather than `torch.lobpcg`: the connection Laplacian is only 3N x 3N (a few hundred
    # dims for typical graphs) and we need its 3 smallest eigenvectors. lobpcg's iteration count to a
    # fixed `tol` blows up when the spectral gap is tiny (weakly-connected / ill-conditioned graphs),
    # giving unbounded, unpredictable runtime (seconds). Dense `eigh` is exact, deterministic and
    # bounded (O((3N)^3)), and matches the dense solver already used for the translation step below.
    eigenvectors = _robust_smallest_eigenvectors(laplacian.to_dense(), k=3)
    R_global = eigenvectors[:, :3].reshape(-1, 3, 3)   # 3 eigenvectors of smallest eigenvalues (eigh returns ascending)
    R_global = torch.cat([
        R_global[:, :, :2],
        torch.sign(torch.linalg.det(R_global))[:, None, None] * R_global[:, :, 2:3]
    ], dim=-1)
    R_global = kabsch(R_global)    # Ensure SO(3)

    return R_global


def _pose_graph_optimization_procrustes_iteration(R_global: Tensor, edges: Tensor, R_rel: Tensor, w: Tensor, niter: int = 10) -> Tensor:
    DAMP = 0.4
    edges_flat, edges_flat_swap = edges.reshape(-1), edges.flip(1).reshape(-1)

    w_R = w[:, None, None] * R_rel
    w_R_dual = torch.stack([w_R, w_R.mT], dim=1).reshape(-1, 3, 3) 
    w_agg = torch.zeros(R_global.shape[0], device=R_global.device, dtype=R_global.dtype).index_add(0, edges_flat, w.repeat_interleave(2))
    for _ in range(niter):
        M = (DAMP * w_agg[..., None, None] * R_global).index_add(0, edges_flat_swap, w_R_dual @ R_global.index_select(0, edges_flat))
        R_global = kabsch(M)
    return R_global


def pose_graph_edge_moments(
    x: Tensor,
    y: Tensor,
    w: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reduce per-edge point correspondences to centered second-moment statistics for pose graph optimization.

    Each edge `i -> j` carries a fixed-size set of `M` 3D point correspondences: `x` points expressed in
    node `i`'s local frame and `y` points in node `j`'s local frame (the same world point observed in
    both nodes). For variable-length correspondences per edge use `segment_pose_graph_edge_moments`.

    The full per-edge point cost `sum_k w_k ||T_ij x_k - y_k||^2` is a quadratic in the global poses
    that depends on the points only through these statistics, so passing them is exactly equivalent to
    passing the raw points. They are *centered* (computed about each edge's weighted centroid) to avoid
    the catastrophic cancellation that would corrupt a polar decomposition when the centroid is far from
    the origin — the same reason `umeyama` takes separate covariance + mean rather than a raw
    (homogeneous) second-moment matrix.

    Parameters
    ----
    - `x`: (..., M, 3) source points (node `i` frame) per edge.
    - `y`: (..., M, 3) destination points (node `j` frame), aligned with `x`.
    - `w`: optional (..., M) per-correspondence weight. If None, uniform.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `cov_yx`: (..., 3, 3) centered cross-covariance `sum_k w_k (y_k - mean_y)(x_k - mean_x)^T / w_e`
    - `cov_xx`: (..., 3, 3) centered covariance of `src` (`x`) — the point spread that constrains rotation,
        and (with `cov_yy`) the scale information reserved for a future similar (`s, R, t`) extension.
    - `cov_yy`: (..., 3, 3) centered covariance of `dst` (`y`).
    - `mean_x`: (..., 3) weighted centroid of `src`.
    - `mean_y`: (..., 3) weighted centroid of `dst`.
    - `w`: (...,) total correspondence weight per edge (the per-edge sum of the input `w`; translation
        information).
    """
    if w is None:
        w = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
    weight = torch.sum(w, dim=-1)
    w_e = weight.clamp_min(eps)
    mean_x = torch.sum(w[..., None] * x, dim=-2) / w_e[..., None]
    mean_y = torch.sum(w[..., None] * y, dim=-2) / w_e[..., None]
    x_c = x - mean_x[..., None, :]
    y_c = y - mean_y[..., None, :]
    cov_yx = torch.sum(w[..., None, None] * vector_outer(y_c, x_c), dim=-3) / w_e[..., None, None]
    cov_xx = torch.sum(w[..., None, None] * vector_outer(x_c, x_c), dim=-3) / w_e[..., None, None]
    cov_yy = torch.sum(w[..., None, None] * vector_outer(y_c, y_c), dim=-3) / w_e[..., None, None]
    return cov_yx, cov_xx, cov_yy, mean_x, mean_y, weight


def segment_pose_graph_edge_moments(
    x: Tensor,
    y: Tensor,
    w: Optional[Tensor] = None,
    *,
    offsets: Tensor,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Segment array mode for `pose_graph_edge_moments`.

    Each edge `i -> j` carries a set of 3D point correspondences: `x` points expressed in node `i`'s
    local frame and `y` points in node `j`'s local frame (the same world point observed in both
    nodes). The number of correspondences may differ across edges; they are concatenated and delimited
    by `offsets` (segment layout, as in `segment_solve_pose`).

    Parameters
    ----
    - `x`: (M, 3) source points (node `i` frame) of all edges, concatenated.
    - `y`: (M, 3) destination points (node `j` frame), concatenated, aligned with `x`.
    - `w`: optional (M,) per-correspondence weight. If None, uniform.
    - `offsets`: (E + 1,) segment offsets delimiting each edge's correspondences.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `cov_yx`: (E, 3, 3) centered cross-covariance `sum_k w_k (y_k - mean_y)(x_k - mean_x)^T / w_e`
    - `cov_xx`: (E, 3, 3) centered covariance of `src` (`x`) — the point spread that constrains rotation,
        and (with `cov_yy`) the scale information reserved for a future similar (`s, R, t`) extension.
    - `cov_yy`: (E, 3, 3) centered covariance of `dst` (`y`).
    - `mean_x`: (E, 3) weighted centroid of `src`.
    - `mean_y`: (E, 3) weighted centroid of `dst`.
    - `w`: (E,) total correspondence weight per edge (the per-edge sum of the input `w`; translation
        information).
    """
    if w is None:
        w = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    lengths = torch.diff(offsets)
    weight = torch.segment_reduce(w, 'sum', offsets=offsets, axis=0)
    w_e = weight.clamp_min(eps)
    mean_x = torch.segment_reduce(w[:, None] * x, 'sum', offsets=offsets, axis=0) / w_e[:, None]
    mean_y = torch.segment_reduce(w[:, None] * y, 'sum', offsets=offsets, axis=0) / w_e[:, None]
    x_c = x - torch.repeat_interleave(mean_x, lengths, dim=0)
    y_c = y - torch.repeat_interleave(mean_y, lengths, dim=0)
    cov_yx = torch.segment_reduce(w[:, None, None] * vector_outer(y_c, x_c), 'sum', offsets=offsets, axis=0) / w_e[:, None, None]
    cov_xx = torch.segment_reduce(w[:, None, None] * vector_outer(x_c, x_c), 'sum', offsets=offsets, axis=0) / w_e[:, None, None]
    cov_yy = torch.segment_reduce(w[:, None, None] * vector_outer(y_c, y_c), 'sum', offsets=offsets, axis=0) / w_e[:, None, None]
    return cov_yx, cov_xx, cov_yy, mean_x, mean_y, weight


def pose_graph_optimization(
    num_nodes: int,
    edges: Tensor,
    cov_yx: Tensor,
    cov_xx: Tensor,
    cov_yy: Tensor,
    mean_x: Tensor,
    mean_y: Tensor,
    w: Tensor,
    edge_weights: Optional[Tensor] = None,
    *,
    mode: Literal['rigid', 'similar'] = 'rigid',
    niter: int = 10,
    eps: float = 1e-12,
) -> Tensor:
    """Pose graph optimization for global poses from per-edge centered point statistics.

    Build the per-edge statistics with `pose_graph_edge_moments` from point correspondences (or
    construct them yourself). Rotation and translation are weighted by their own information derived
    from the point geometry — rotation by the point spread (`tr(cov_xx) + tr(cov_yy)`), translation by
    the total correspondence weight (`w`) — so the rotation/translation trade-off is fixed by the
    data rather than by a hand-tuned scalar.

    With `mode='similar'` each node additionally carries its own scale relative to the world
    (`x_node_i = s_i R_i p + t_i`). The per-edge relative scale `s_ij = sqrt(tr(cov_yy) / tr(cov_xx))`
    satisfies `s_ij = s_j / s_i`, so the global node scales are recovered by a separate weighted
    least squares in log-scale (see `_pose_graph_optimization_scale_sync`); the overall scale is a
    free gauge, fixed so the geometric mean of the node scales is 1. Rotation is scale-invariant and
    solved identically to the rigid case; translation uses the scaled relative block `s_ij R_ij`.

    Parameters
    ----
    - `num_nodes`: number of nodes `N` in the pose graph.
    - `edges`: (E, 2) edge list. Each edge is a pair of node indices `i -> j`.
    - `cov_yx`: (E, 3, 3) centered cross-covariance per edge (see `pose_graph_edge_moments`); its polar
        factor is the relative rotation `R_ij` (node `i` -> node `j`).
    - `cov_xx`: (E, 3, 3) centered covariance of the source (node `i`) points per edge.
    - `cov_yy`: (E, 3, 3) centered covariance of the destination (node `j`) points per edge.
    - `mean_x`: (E, 3) source (node `i`) centroid per edge.
    - `mean_y`: (E, 3) destination (node `j`) centroid per edge.
    - `w`: (E,) total correspondence weight per edge (translation information).
    - `edge_weights`: optional (E,) extra per-edge confidence prior, multiplied into both information
        weights. If None, uniform.
    - `mode`: `'rigid'` for `(R, t)` poses (default), or `'similar'` to additionally solve a per-node
        scale `s` (poses become `s R | t`).
    - `niter`: number of Procrustes iterations to refine global rotations. If 0, only the Laplacian
        eigen-decomposition initialization is used.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `poses_global`: (N, 4, 4) global poses (world-to-node) for each node. The linear part is `R_i`
        (rigid) or `s_i R_i` (similar).

        `relative[i->j] ≈ poses_global[j] @ poses_global[i].inv()`, where `relative[i->j]` has rotation
        `polar(cov_yx)`, scale `sqrt(tr(cov_yy) / tr(cov_xx))` (similar mode) and translation
        `mean_y - s_ij R_ij @ mean_x`.
    """
    if edge_weights is None:
        edge_weights = torch.ones(edges.shape[0], device=cov_yx.device, dtype=cov_yx.dtype)
    # Rotation information is the total point spread (un-normalized scatter); translation information is
    # the total correspondence weight.
    tr_xx = matrix_trace(cov_xx, dim1=-2, dim2=-1)
    tr_yy = matrix_trace(cov_yy, dim1=-2, dim2=-1)
    scatter = w * (tr_xx + tr_yy)
    w_rotation = edge_weights * scatter
    w_translation = edge_weights * w

    R_relative = kabsch(cov_yx)

    # Global rotations: Laplacian eigen-decomposition initialization + Procrustes refinement, weighted
    # by per-edge rotation information. Scale-invariant, so identical for rigid and similar modes.
    laplacian_rot = _pose_graph_optimization_construct_laplacian(edges, num_nodes, R_relative, w_rotation)
    R_global = _pose_graph_optimization_eigen_decomposition(laplacian_rot)
    if niter > 0:
        R_global = _pose_graph_optimization_procrustes_iteration(R_global, edges, R_relative, w_rotation, niter=niter)

    # Global per-node scales (similar mode only): log-scale graph least squares from relative scales.
    if mode == 'similar':
        s_relative = torch.sqrt((tr_yy / tr_xx.clamp_min(eps)).clamp_min(eps))
        s_global = _pose_graph_optimization_scale_sync(edges, num_nodes, s_relative, w_translation, eps=eps)
    else:
        s_relative = None

    # Global translations: connection-Laplacian least squares, weighted by per-edge translation
    # information (a separate Laplacian from the rotation one). In similar mode the relative block is
    # the scaled rotation `s_ij R_ij`.
    sR_mean_x = (R_relative @ mean_x[..., None]).squeeze(-1)
    if mode == 'similar':
        sR_mean_x = s_relative[:, None] * sR_mean_x
    t_relative = mean_y - sR_mean_x
    laplacian_trans = _pose_graph_optimization_construct_laplacian(edges, num_nodes, R_relative, w_translation, s=s_relative)
    w_t = w_translation[:, None] * t_relative
    src_term = (R_relative.mT @ w_t[:, :, None]).squeeze(-1)
    if mode == 'similar':
        src_term = s_relative[:, None] * src_term
    b = torch.zeros((num_nodes, 3), device=cov_yx.device, dtype=cov_yx.dtype).index_add(0, edges[:, 1], w_t).index_add(0, edges[:, 0], -src_term).reshape(-1)
    # NOTE: currently we have to use dense solver for translations since PyTorch doesn't support sparse linear solver well.
    t_global = torch.linalg.lstsq(laplacian_trans.to_dense(), b).solution.reshape(num_nodes, 3)

    linear = R_global if mode == 'rigid' else s_global[:, None, None] * R_global
    return make_affine_matrix(linear, t_global)


def _pose_graph_optimization_moment_residual(
    poses_global: Tensor, edges: Tensor, cov_yx: Tensor, cov_xx: Tensor, cov_yy: Tensor, mean_x: Tensor, mean_y: Tensor,
    mode: Literal['rigid', 'similar'] = 'rigid',
) -> Tensor:
    """Per-edge mean squared point residual — the exact point-level cost, normalized by edge weight.

    For edge `i -> j` the predicted relative map is `M = A_j A_i^{-1}` (linear part `A_i = R_i` for
    rigid, `s_i R_i` for similar), with translation `t_pred = t_j - M t_i`, which maps node-`i` points
    onto node `j`. The full per-edge point cost `sum_k w_k ||M src_k + t_pred - dst_k||^2`, divided by
    the edge weight `w_e`, reduces in closed form (centered, no cancellation) to

        C_e / w_e = ||M mean_x + t_pred - mean_y||^2 + tr(M^T M cov_xx) + tr(cov_yy) - 2 <M, cov_yx>,

    i.e. the weighted-mean squared point error (`tr(M^T M cov_xx) = tr(cov_xx)` when `M` is a rotation),
    so `threshold` is a plain point distance, independent of point count / weight magnitude.
    """
    A_global, t_global = poses_global[:, :3, :3], poses_global[:, :3, 3]
    A_i = A_global.index_select(0, edges[:, 0])
    A_j = A_global.index_select(0, edges[:, 1])
    t_i = t_global.index_select(0, edges[:, 0])
    t_j = t_global.index_select(0, edges[:, 1])
    if mode == 'rigid':
        M_pred = A_j @ A_i.mT
        quad = matrix_trace(cov_xx, dim1=-2, dim2=-1)   # tr(M^T M cov_xx) = tr(cov_xx) for a rotation
    else:
        M_pred = A_j @ safe_inv(A_i)
        quad = ((M_pred.mT @ M_pred) * cov_xx).sum(dim=(-2, -1))   # tr(M^T M cov_xx), cov_xx symmetric
    t_pred = t_j - (M_pred @ t_i[..., None]).squeeze(-1)
    a = (M_pred @ mean_x[..., None]).squeeze(-1) + t_pred - mean_y
    # cov_yx is the (weight-normalized) centered cross-covariance, so <M_pred, cov_yx> is the rotation term.
    cross = (M_pred * cov_yx).sum(dim=(-2, -1))
    res2 = a.square().sum(dim=-1) + quad + matrix_trace(cov_yy, dim1=-2, dim2=-1) - 2.0 * cross
    return res2.clamp_min(0.0)


def pose_graph_optimization_gnc(
    num_nodes: int,
    edges: Tensor,
    cov_yx: Tensor,
    cov_xx: Tensor,
    cov_yy: Tensor,
    mean_x: Tensor,
    mean_y: Tensor,
    w: Tensor,
    edge_weights: Optional[Tensor] = None,
    *,
    mode: Literal['rigid', 'similar'] = 'rigid',
    threshold: float = 0.05,
    niter: int = 10,
    gnc_iters: int = 20,
    gnc_factor: float = 1.4,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """Robust pose graph optimization with Graduated Non-Convexity (GNC-TLS) for outlier edge rejection.

    Wraps `pose_graph_optimization` in an outer loop that re-weights each edge by a Truncated Least
    Squares (TLS) surrogate, gradually annealed from a near-convex problem to the true (non-convex)
    truncated cost via the control parameter `mu` (Yang et al., "Graduated Non-Convexity for Robust
    Spatial Perception", RA-L 2020). Bad edges (wrong loop closures / outlier constraints) converge
    to weight 0 and are effectively removed.

    The outlier test uses the *exact* per-edge mean squared point residual (see
    `_pose_graph_optimization_moment_residual`), so `threshold` is a plain point distance in the units of
    your correspondences — no rotation/translation scale to juggle. Each GNC iteration solves the
    weighted graph, evaluates the residual, then updates each edge's weight in closed form:

        `w_ij = 1`                                          if `r_ij^2 <= mu/(mu+1) * c^2`
        `w_ij = sqrt(mu(mu+1) c^2 / r_ij^2) - mu`           if in the ambiguous band
        `w_ij = 0`                                          if `r_ij^2 >= (mu+1)/mu * c^2`

    where `c = threshold`. `mu` starts small (near-convex) and is multiplied by `gnc_factor` each
    iteration to recover the truncated cost.

    Parameters
    ----
    - `num_nodes`: number of nodes `N` in the pose graph.
    - `edges`: (E, 2) edge list. Each edge is a pair of node indices `i -> j`.
    - `cov_yx`: (E, 3, 3) centered cross-covariance per edge (see `pose_graph_edge_moments`).
    - `cov_xx`: (E, 3, 3) centered covariance of the source (node `i`) points per edge.
    - `cov_yy`: (E, 3, 3) centered covariance of the destination (node `j`) points per edge.
    - `mean_x`: (E, 3) source (node `i`) centroid per edge.
    - `mean_y`: (E, 3) destination (node `j`) centroid per edge.
    - `w`: (E,) total correspondence weight per edge (translation information).
    - `edge_weights`: optional (E,) per-edge confidence prior, multiplied with the GNC weight. If None, uniform.
    - `mode`: `'rigid'` for `(R, t)` poses (default), or `'similar'` to additionally solve a per-node scale
        (see `pose_graph_optimization`). The residual threshold stays a plain point distance either way.
    - `threshold`: inlier point distance `c`. An edge is treated as an inlier when its RMS point
        residual is roughly below `threshold`; set it to your expected correspondence noise.
    - `niter`: inner Procrustes iterations per `pose_graph_optimization` solve.
    - `gnc_iters`: maximum number of outer GNC iterations.
    - `gnc_factor`: `mu` growth factor per outer iteration (`> 1` for GNC-TLS).
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `poses_global`: (N, 4, 4) global poses for each node.

        `relative[i->j] ≈ poses_global[j] @ poses_global[i].inv()`
    - `weights`: (E,) GNC inlier weights in `[0, 1]` (independent of the input `edge_weights`). 1 marks a
        confident inlier, 0 an edge rejected as an outlier; intermediate values are still in the
        ambiguous band. Threshold this to classify edges.
    """
    if edge_weights is None:
        edge_weights = torch.ones(edges.shape[0], device=cov_yx.device, dtype=cov_yx.dtype)

    barc2 = threshold ** 2

    # Non-robust initialization, used both as the starting estimate and to initialize mu.
    poses_global = pose_graph_optimization(num_nodes, edges, cov_yx, cov_xx, cov_yy, mean_x, mean_y, w, edge_weights=edge_weights, mode=mode, niter=niter, eps=eps)
    res2 = _pose_graph_optimization_moment_residual(poses_global, edges, cov_yx, cov_xx, cov_yy, mean_x, mean_y, mode=mode)

    # GNC-TLS mu initialization. If every residual is already within the inlier band there is no
    # outlier to reject and the surrogate is degenerate, so return the plain estimate directly.
    max_res2 = res2.max()
    denom = 2.0 * max_res2 - barc2
    if denom <= 0:
        return poses_global, torch.ones_like(edge_weights)

    mu = (barc2 / denom).clamp_min(eps)
    w_gnc = torch.ones_like(edge_weights)
    for _ in range(gnc_iters):
        th_lo = mu / (mu + 1.0) * barc2
        th_hi = (mu + 1.0) / mu * barc2
        w_mid = (torch.sqrt(mu * (mu + 1.0) * barc2 / res2.clamp_min(eps)) - mu).clamp(0.0, 1.0)
        w_gnc_new = torch.where(res2 <= th_lo, torch.ones_like(edge_weights), torch.where(res2 >= th_hi, torch.zeros_like(edge_weights), w_mid))

        if torch.max(torch.abs(w_gnc_new - w_gnc)) < 1e-6:
            w_gnc = w_gnc_new
            break
        w_gnc = w_gnc_new

        poses_global = pose_graph_optimization(num_nodes, edges, cov_yx, cov_xx, cov_yy, mean_x, mean_y, w, edge_weights=edge_weights * w_gnc, mode=mode, niter=niter, eps=eps)
        res2 = _pose_graph_optimization_moment_residual(poses_global, edges, cov_yx, cov_xx, cov_yy, mean_x, mean_y, mode=mode)
        mu = mu * gnc_factor

    return poses_global, w_gnc

