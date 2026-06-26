import math

import numpy as np
from numpy import ndarray
from typing import *
from numbers import Number
from ..helpers import no_warnings

from .transforms import make_affine_matrix, transform_points
from .utils import safe_inv, vector_outer


__all__ = [
    'kabsch',
    'umeyama',
    'affine_umeyama',
    'solve_pose',
    'solve_pose_ransac',
    'segment_solve_pose',
    'solve_poses_sequential',
    'segment_solve_poses_sequential',
]


def kabsch(cov: ndarray) -> ndarray:
    U, _, Vh = np.linalg.svd(cov)
    Vh[..., 2, :] *= np.sign(np.linalg.det(U @ Vh))[..., None]
    R = U @ Vh
    return R


def umeyama(cov_yx: ndarray, cov_xx: Optional[ndarray] = None, cov_yy: Optional[ndarray] = None, mean_x: Optional[ndarray] = None, mean_y: Optional[ndarray] = None) -> Tuple[ndarray, ndarray, ndarray]:
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
    dtype = cov_yx.dtype
    R = kabsch(cov_yx)
    if cov_xx is not None and cov_yy is None:
        s = np.trace(cov_yx @ R.swapaxes(-2, -1), axis1=-2, axis2=-1) / np.maximum(np.trace(cov_xx, axis1=-2, axis2=-1), np.finfo(dtype).tiny)
    if cov_xx is None and cov_yy is not None:
        s = np.trace(cov_yy, axis1=-2, axis2=-1) / np.maximum(np.trace(cov_yx @ R.swapaxes(-2, -1), axis1=-2, axis2=-1), np.finfo(dtype).tiny)
    elif cov_xx is not None and cov_yy is not None:
        x_fnorm = np.maximum(np.trace(cov_xx, axis1=-2, axis2=-1), np.finfo(dtype).tiny)
        y_fnorm = np.maximum(np.trace(cov_yy, axis1=-2, axis2=-1), np.finfo(dtype).tiny)
        s = np.sqrt(y_fnorm / x_fnorm)
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


def _sym_sqrt_and_inv_sqrt(mat: ndarray, eps: float) -> Tuple[ndarray, ndarray]:
    """Symmetric square root and inverse square root of a batch of SPD matrices, from a single eigendecomposition."""
    L, V = np.linalg.eigh(mat)
    sqrt_L = np.sqrt(np.maximum(L, eps))
    mat_sqrt = (V * sqrt_L[..., None, :]) @ V.swapaxes(-2, -1)
    mat_inv_sqrt = (V * (1.0 / sqrt_L)[..., None, :]) @ V.swapaxes(-2, -1)
    return mat_sqrt, mat_inv_sqrt


def affine_umeyama(cov_yx: ndarray, cov_xx: ndarray, cov_yy: ndarray, mean_x: ndarray, mean_y: ndarray, lam: float = 1e-2) -> Tuple[ndarray, ndarray]:
    """
    Extended Procrustes analysis to solve for affine transformation `A` and translation `t` such that `y_i ~= A x_i + t`.

    The inverse-consistency constraint (the inverse map `A^{-1}` should align `y` back onto `x`) is
    satisfied *exactly* in closed form by whitening both point clouds to unit covariance and solving
    an orthogonal Procrustes problem in the whitened space, where the optimal map is a rotation `Q`
    (so `(A^{-1})` is automatically the consistent inverse):

        `A = cov_yy^{1/2} @ Q @ cov_xx^{-1/2}`,   `Q = polar(cov_yy^{-1/2} @ cov_yx @ cov_xx^{-1/2})`

    No iteration and no penalty annealing.

    Parameters
    ----
    - `cov_yx`: (..., 3, 3) covariance matrix between y and x points.
    - `cov_xx`: (..., 3, 3) covariance matrix of x points.
    - `cov_yy`: (..., 3, 3) covariance matrix of y points.
    - `mean_x`: (..., 3) mean of x points.
    - `mean_y`: (..., 3) mean of y points.
    - `lam`: rigidity regularization weight. Shrinks the whitening toward isotropic, biasing `A`
        toward a similarity (rotation + uniform scale) transform and stabilizing the inverse sqrt
        for degenerate (e.g. near-planar) inputs.

    Returns
    ----
    - `A`: (..., 3, 3) affine transformation matrix.
    - `t`: (..., 3) translation vector.
    """
    dtype = cov_yx.dtype
    eps = np.finfo(dtype).tiny
    n = cov_xx.shape[-1]
    I = np.eye(n, dtype=dtype)
    tr_xx = np.maximum(np.trace(cov_xx, axis1=-2, axis2=-1), eps)
    tr_yy = np.maximum(np.trace(cov_yy, axis1=-2, axis2=-1), eps)

    # Mild rigidity / numerical ridge: shrink the whitening toward isotropic.
    reg_xx = cov_xx + lam * (tr_xx / n)[..., None, None] * I
    reg_yy = cov_yy + lam * (tr_yy / n)[..., None, None] * I

    _, cov_xx_inv_sqrt = _sym_sqrt_and_inv_sqrt(reg_xx, eps)
    cov_yy_sqrt, cov_yy_inv_sqrt = _sym_sqrt_and_inv_sqrt(reg_yy, eps)

    M = cov_yy_inv_sqrt @ cov_yx @ cov_xx_inv_sqrt
    U, _, Vh = np.linalg.svd(M)
    Q = U @ Vh

    A = cov_yy_sqrt @ Q @ cov_xx_inv_sqrt
    t = mean_y - transform_points(mean_x, A)
    return A, t



def solve_pose(
    p: np.ndarray, 
    q: np.ndarray, 
    w: Optional[np.ndarray] = None, 
    sigma: Optional[np.ndarray] = None, 
    *,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2
) -> np.ndarray:
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
    
    Returns
    ----
    - `pose`: (..., 4, 4) transformations matrix from p to q.
    """
    if w is None:
        w = np.ones(p.shape[:-1], dtype=p.dtype)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=p.dtype)
        w = w / np.maximum(np.square(sigma), np.finfo(p.dtype).tiny)
    w_sum = np.maximum(np.sum(w, axis=-1), np.finfo(p.dtype).tiny)
    p_mean = np.sum(w[..., None] * p, axis=-2) / w_sum[..., None]
    q_mean = np.sum(w[..., None] * q, axis=-2) / w_sum[..., None]
    p = p - p_mean[..., None, :]
    q = q - q_mean[..., None, :]
    pw = p * w[..., None]
    qw = q * w[..., None]
    cov_qp = np.sum(vector_outer(qw, p), axis=-3) / w_sum[..., None, None]
    if mode == 'similar' or mode == 'affine':
        cov_pp = np.sum(vector_outer(pw, p), axis=-3) / w_sum[..., None, None]
        cov_qq = np.sum(vector_outer(qw, q), axis=-3) / w_sum[..., None, None]
    
    if mode == 'rigid':
        _, R, t = umeyama(cov_qp, mean_x=p_mean, mean_y=q_mean)
        pose = make_affine_matrix(R, t)
    elif mode == 'similar':
        s, R, t = umeyama(cov_qp, cov_xx=cov_pp, cov_yy=cov_qq, mean_x=p_mean, mean_y=q_mean)
        pose = make_affine_matrix(s * R, t)
    elif mode == 'affine':
        A, t = affine_umeyama(cov_qp, cov_pp, cov_qq, p_mean, q_mean, lam=lam)
        pose = make_affine_matrix(A, t)
    
    return pose


def solve_pose_ransac(
    p: np.ndarray, 
    q: np.ndarray, 
    w: Optional[np.ndarray] = None, 
    sigma: Optional[np.ndarray] = None, 
    *,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    threshold: Union[float, np.ndarray] = 0.05,
    num_samples: int = 32,
    sample_size: Optional[int] = None,
    lam: float = 1e-2, 
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Robustly solve for the pose (transformation from p to q) given point correspondences using RANSAC.

    Hypotheses are sampled from minimal subsets, scored by a truncated soft-inlier cost, and the best
    one is refit on all of its inliers. Vectorized over hypotheses and leading batch dimensions.

    The solve minimizes `sum_i (w_i / sigma_i^2) ||pose @ p_i - q_i||^2` (as in `solve_pose`), while the
    inlier test is the purely geometric `||pose @ p_i - q_i|| < threshold_i`. `w`, `sigma`, and
    `threshold` act independently.

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
    - `threshold`: inlier distance threshold (scalar or per-point array, broadcastable to (..., N)). A
        correspondence is an inlier when `||pose @ p_i - q_i|| < threshold_i`. For a relative tolerance
        pass `relative_threshold * ||p_i||`.
    - `num_samples`: number of RANSAC hypotheses per batch element. Compute/memory scale linearly with it.
    - `sample_size`: size of each minimal sample. If None, defaults to 3 for 'rigid'/'similar' and 4 for 'affine'.
    - `lam`: regularization weight for 'affine' mode.
    - `rng`: optional random generator for reproducible sampling.

    Returns
    ----
    - `pose`: (..., 4, 4) transformations matrix from p to q.
    - `inliers`: (..., N) boolean mask of inliers w.r.t. the returned pose.
    """
    if sample_size is None:
        sample_size = 4 if mode == 'affine' else 3
    if rng is None:
        rng = np.random.default_rng()
    batch_shape = p.shape[:-2]
    N = p.shape[-2]
    B = math.prod(batch_shape)

    p_flat = p.reshape(B, N, 3)
    q_flat = q.reshape(B, N, 3)
    if w is None:
        w_flat = np.ones((B, N), dtype=p.dtype)
    else:
        w_flat = w.reshape(B, N)

    # Optional per-point noise scale, passed through to every solve (identical meaning to solve_pose).
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=p.dtype)
        sigma_flat = np.broadcast_to(sigma, p.shape[:-1]).reshape(B, N)  # (B, N)
    else:
        sigma_flat = None

    # Inlier threshold: a purely geometric gate (scalar or per-point), never folded into the solve.
    tiny = np.finfo(p.dtype).tiny
    threshold = np.maximum(np.asarray(threshold, dtype=p.dtype), tiny)
    threshold_b = np.broadcast_to(threshold, p.shape[:-1]).reshape(B, N)[:, None, :] if threshold.ndim > 0 else threshold  # (B, 1, N) or scalar

    # Draw `num_samples` minimal subsets without replacement per batch element, sampling each point
    # with probability proportional to its confidence `w`, so high-confidence correspondences are
    # more likely to seed a hypothesis. `np.random.choice` can't draw a batch of independent subsets
    # in one vectorized call, so we use the Gumbel-top-k trick (Efraimidis-Spirakis): perturbing each
    # `log(w_i)` by i.i.d. Gumbel noise and taking the top-`sample_size` keys yields exactly weighted
    # sampling without replacement. Weights are floored to `tiny` so a draw is always possible even
    # when fewer than `sample_size` points have nonzero weight. Uniform `w` -> uniform sampling.
    u = np.maximum(rng.random((B, num_samples, N)).astype(p.dtype), tiny)
    keys = np.log(np.maximum(w_flat[:, None, :], tiny)) - np.log(-np.log(u))  # log(w_i) + Gumbel noise
    idx = np.argpartition(keys, -sample_size, axis=-1)[..., -sample_size:].astype(np.int32)  # (B, num_samples, sample_size)
    p_s = np.take_along_axis(p_flat[:, None, :, :], idx[..., None], axis=2)  # (B, num_samples, sample_size, 3)
    q_s = np.take_along_axis(q_flat[:, None, :, :], idx[..., None], axis=2)
    w_s = np.take_along_axis(w_flat[:, None, :], idx, axis=2)
    sigma_s = np.take_along_axis(sigma_flat[:, None, :], idx, axis=2) if sigma_flat is not None else None

    # Solve a candidate pose for every hypothesis.
    pose_h = solve_pose(p_s, q_s, w_s, sigma_s, mode=mode, lam=lam)  # (B, num_samples, 4, 4)

    # Score hypotheses by a robust, truncated soft-inlier cost. The threshold normalizes the raw
    # residual (purely geometric); the confidence `w` weights each point's contribution but does
    # NOT relax its threshold. Each correspondence contributes `w_i * min(1, residual_i / threshold_i)`.
    p_t = transform_points(p_flat[:, None, :, :], pose_h[:, :, None, :, :])     # (B, num_samples, N, 3)
    residual = np.linalg.norm(p_t - q_flat[:, None, :, :], axis=-1)             # (B, num_samples, N)
    normalized_residual = residual / threshold_b                               # (B, num_samples, N)
    inliers = normalized_residual < 1.0  # (B, num_samples, N)
    inlier_cost = (w_flat[:, None, :] * np.minimum(normalized_residual, 1.0)).mean(axis=-1)  # (B, num_samples)

    best = np.argmin(inlier_cost, axis=-1)  # (B,)
    best_inliers = np.take_along_axis(inliers, best[:, None, None], axis=1)[:, 0]  # (B, N)

    # Refit on all inliers of the best hypothesis (same w / sigma weighting as the hypotheses).
    refit_w = w_flat * best_inliers.astype(w_flat.dtype)
    pose = solve_pose(p_flat, q_flat, refit_w, sigma_flat, mode=mode, lam=lam)  # (B, 4, 4)

    pose = pose.reshape(*batch_shape, 4, 4)
    best_inliers = best_inliers.reshape(*batch_shape, N)
    return pose, best_inliers


def segment_solve_pose(
    p: np.ndarray, 
    q: np.ndarray, 
    w: Optional[np.ndarray] = None, 
    sigma: Optional[np.ndarray] = None, 
    *,
    offsets: np.ndarray, 
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2
) -> np.ndarray:
    """Solve for the pose (transformation from p to q) given weighted point correspondences.

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
    
    Returns
    ----
    - `pose`: (S, 4, 4) transformations matrix from p to q.
    """
    if w is None:
        w = np.ones(p.shape[:-1], dtype=p.dtype)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=p.dtype)
        w = w / np.maximum(np.square(sigma), np.finfo(p.dtype).tiny)

    lengths = np.diff(offsets)
    w_sum = np.maximum(np.add.reduceat(w, offsets[:-1], axis=0), np.finfo(p.dtype).tiny)
    p_mean = np.add.reduceat(w[..., None] * p, offsets[:-1], axis=0) / w_sum[:, None]
    q_mean = np.add.reduceat(w[..., None] * q, offsets[:-1], axis=0) / w_sum[:, None]
    p = p - np.repeat(p_mean, lengths, axis=0)
    q = q - np.repeat(q_mean, lengths, axis=0)
    pw = p * w[..., None]
    qw = q * w[..., None]
    cov_qp = np.add.reduceat(vector_outer(qw, p), offsets[:-1], axis=0) / w_sum[:, None, None]
    if mode == 'similar' or mode == 'affine':
        cov_pp = np.add.reduceat(vector_outer(pw, p), offsets[:-1], axis=0) / w_sum[:, None, None]    
        cov_qq = np.add.reduceat(vector_outer(qw, q), offsets[:-1], axis=0) / w_sum[:, None, None]
    
    if mode == 'rigid':
        _, R, t = umeyama(cov_qp, mean_x=p_mean, mean_y=q_mean)
        pose = make_affine_matrix(R, t)
    elif mode == 'similar':
        s, R, t = umeyama(cov_qp, cov_xx=cov_pp, cov_yy=cov_qq, mean_x=p_mean, mean_y=q_mean)
        pose = make_affine_matrix(s * R, t)
    elif mode == 'affine':
        A, t = affine_umeyama(cov_qp, cov_pp, cov_qq, p_mean, q_mean, lam=lam)
        pose = make_affine_matrix(A, t)
    
    return pose


def solve_poses_sequential(
    trajectories: ndarray,
    weights: Optional[ndarray] = None,
    noise_scales: Optional[ndarray] = None,
    *,
    accum: Optional[Tuple[ndarray, ...]] = None,
    min_valid_size: int = 3,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2
) -> Tuple[ndarray, Tuple[ndarray, ...], Tuple[ndarray, ndarray, ndarray, ndarray]]:
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

    Returns
    ----
    - `poses`: (T, ..., 4, 4) transformations from canonical to each frame.
    - `valid`: (T, ...) boolean mask indicating valid segments
    - `stats`: canonical statistics of each group,
        It is a tuple of:
        - `mu`: (..., 3) weighted mean of points
        - `cov`: (..., 3, 3) weighted covariance of points
        - `tot_w`: (...,) total weight of points
        - `nnz`: (...,) number of non-zero weight points
    - `canonical_points`: (..., N, 3) canonical points.
    - `err`: (..., N,) per-point RMS error over all time := sqrt(sum_over_time(per_point_weights * per_point_squared_error) / per_point_nnz)
        Use this to filter outliers as needed.
    - `accum`: per point accumulated statistics. Just pass it to the next call for incremental solving.
        It is a tuple of:
        - `accum_sqrtw`: (..., N,) sum of sqrt(weights)
        - `accum_sqrtwx`: (..., N, 3) sum of sqrt(weights) * x
        - `accum_sqrtwxx`: (...N, 3, 3) sum of sqrt(weights) * outer(x - mean_sqrtwx, x - mean_sqrtwx)
        - `accum_w`: (..., N,) sum of weights
        - `accum_wx`: (..., N, 3) sum of weights * x
        - `accum_wxx`: (..., N, 3, 3) sum of weights * outer(x - mean_wx, x - mean_wx)
        - `accum_nnz`: (..., N,) number of non-zero weight accumulations

    Example
    ----
    ```
    accum = None
    poses, valid = [], []
    for new_trajectories_chunk in data_stream:
        # new_trajectories_chunk: (T_chunk, N, 3)
        poses_chunk, valid_chunk, stats, canonical_points, err, accum = solve_poses_sequential(
            new_trajectories_chunk,
            accum=accum,
        )
        poses.append(poses_chunk)
        valid.append(valid_chunk)
        # `stats`, `canonical_points` and `err` are returned and updated every chunk.
    poses = np.concatenate(poses, axis=0)   # (T_all, 4, 4), poses over all frames
    valid = np.concatenate(valid, axis=0)   # (T_all,), poses' validity over all frames
    """
    dtype = trajectories.dtype
    num_frames = trajectories.shape[0]
    num_points = trajectories.shape[-2]
    batch_shape = trajectories.shape[1:-2]

    if weights is None:
        weights = np.ones((num_frames, *batch_shape, num_points), dtype=dtype)
    if noise_scales is not None:
        noise_scales = np.asarray(noise_scales, dtype=dtype)
        weights = weights / np.maximum(np.square(noise_scales), np.finfo(dtype).tiny)

    poses = np.zeros((num_frames, *batch_shape, 4, 4), dtype=dtype)
        
    if accum is not None:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = [a.copy() for a in accum]
    else:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = \
            np.zeros((*batch_shape, num_points,), dtype=dtype), \
            np.zeros((*batch_shape, num_points, 3), dtype=dtype), \
            np.zeros((*batch_shape, num_points, 3, 3), dtype=dtype), \
            np.zeros((*batch_shape, num_points,), dtype=dtype), \
            np.zeros((*batch_shape, num_points, 3), dtype=dtype), \
            np.zeros((*batch_shape, num_points, 3, 3), dtype=dtype), \
            np.zeros((*batch_shape, num_points,), dtype=dtype)
    
    for i in range(num_frames):
        # Compute weighted statistics
        mean_sqrtwx = accum_sqrtwx / np.maximum(accum_sqrtw, np.finfo(trajectories.dtype).tiny)[..., None]
        wi, yi = weights[i], trajectories[i]
        sqrtwi = np.sqrt(wi)
        w = sqrtwi * accum_sqrtw
        sum_w = np.sum(w, axis=-1) + np.finfo(dtype).tiny
        center_x = np.sum(sqrtwi[..., None] * accum_sqrtwx, axis=-2) / sum_w[..., None]
        center_y = np.sum(w[..., None] * yi, axis=-2) / sum_w[..., None]
        xc = mean_sqrtwx - center_x[..., None, :]
        yc = yi - center_y[..., None, :]
        cov_yx = np.einsum('...i,...ij,...ik->...jk', w, yc, xc) / sum_w[..., None, None]
        if mode == 'affine' or mode == 'similar':
            cov_xx = (np.einsum('...i,...ij,...ik->...jk', w, xc, xc) + np.einsum('...i,...ijk->...jk', sqrtwi, accum_sqrtwxx)) / sum_w[..., None, None]
            cov_yy = np.einsum('...i,...ij,...ik->...jk', w, yc, yc) / sum_w[..., None, None]
        
        # Solve for pose
        if mode == 'rigid':
            _, R, t = umeyama(cov_yx, mean_x=center_x, mean_y=center_y)
            poses[i] = make_affine_matrix(R, t)
        elif mode == 'similar':
            s, R, t = umeyama(cov_yx, cov_xx=cov_xx, mean_x=center_x, mean_y=center_y)
            poses[i] = make_affine_matrix(s * R, t)
        elif mode == 'affine':
            A, t = affine_umeyama(cov_yx, cov_xx, cov_yy, center_x, center_y, lam=lam)
            poses[i] = make_affine_matrix(A, t)

        xi = transform_points(yi, safe_inv(poses[i])[..., None, :, :])

        # Update accum
        old_mean_sqrtwx, old_accum_sqrtw = mean_sqrtwx.copy(), accum_sqrtw.copy()
        accum_sqrtw += sqrtwi
        accum_sqrtwx += sqrtwi[..., None] * xi
        mean_sqrtwx = accum_sqrtwx / np.maximum(accum_sqrtw, np.finfo(dtype).tiny)[..., None]
        accum_sqrtwxx += old_accum_sqrtw[..., None, None] * vector_outer(mean_sqrtwx - old_mean_sqrtwx) + sqrtwi[..., None, None] * vector_outer(xi - mean_sqrtwx)

        mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[..., None]
        old_mean_wx, old_accum_w = mean_wx.copy(), accum_w.copy()
        accum_w += wi
        accum_wx += wi[..., None] * xi
        mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[..., None]
        accum_wxx += old_accum_w[..., None, None] * vector_outer(mean_wx - old_mean_wx) + wi[..., None, None] * vector_outer(xi - mean_wx)
        accum_nnz += wi > 0

    tot_w = np.sum(accum_w, axis=-1)
    mu = np.sum(accum_wx, axis=-2) / np.maximum(tot_w, np.finfo(dtype).tiny)[..., None]
    mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[..., None]
    sigma = np.sum(accum_wxx + accum_w[..., None, None] * vector_outer(mu[..., None, :] - mean_wx), axis=-3) / np.maximum(tot_w, np.finfo(dtype).tiny)[..., None, None]
    nnz = np.sum(accum_nnz, axis=-1)
    valid = np.sum(weights > 0, axis=-1) >= min_valid_size
    err = np.sqrt(np.trace(accum_wxx, axis1=-2, axis2=-1) / np.maximum(accum_nnz, np.finfo(dtype).tiny))

    return poses, valid, (mu, sigma, tot_w, nnz), mean_wx, err, (accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz)


def segment_solve_poses_sequential(
    trajectories: ndarray,
    weights: Optional[ndarray] = None,
    offsets: ndarray = None,
    noise_scales: Optional[ndarray] = None,
    *,
    accum: Optional[Tuple[ndarray, ...]] = None,
    min_valid_size: int = 3,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2
) -> Tuple[ndarray, Tuple[ndarray, ...], Tuple[ndarray, ndarray, ndarray, ndarray]]:
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
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed. 
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `lam`: rigidity regularization weight for 'affine' mode.

    Returns
    ----
    - `poses`: (T, S, 4, 4) transformations from canonical to each frame.
    - `valid`: (T, S) boolean mask indicating valid segments
    - `stats`: canonical statistics of each group,
        It is a tuple of:
        - `mu`: (S, 3) weighted mean of points
        - `cov`: (S, 3, 3) weighted covariance of points
        - `tot_w`: (S,) total weight of points
        - `nnz`: (S,) number of non-zero weight points
    - `canonical_points`: (N, 3) canonical points.
    - `err`: (N,) per-point RMS error over all time := sqrt(sum_over_time(per_point_weights * per_point_squared_error) / per_point_nnz)
        Use this to filter outliers as needed.
    - `accum`: per point accumulated statistics. Just pass it to the next call for incremental solving.
        It is a tuple of:
        - `accum_sqrtw`: (N,) sum of sqrt(weights)
        - `accum_sqrtwx`: (N, 3) sum of sqrt(weights) * x
        - `accum_sqrtwxx`: (N, 3, 3) sum of sqrt(weights) * outer(x - mean_sqrtwx, x - mean_sqrtwx)
        - `accum_w`: (N,) sum of weights
        - `accum_wx`: (N, 3) sum of weights * x
        - `accum_wxx`: (N, 3, 3) sum of weights * outer(x - mean_wx, x - mean_wx)
        - `accum_nnz`: (N,) number of non-zero weight accumulations
    """
    dtype = trajectories.dtype
    num_frames = trajectories.shape[0]
    num_points = trajectories.shape[1]

    if weights is None:
        weights = np.ones((num_frames, num_points), dtype=dtype)
    if noise_scales is not None:
        noise_scales = np.asarray(noise_scales, dtype=dtype)
        weights = weights / np.maximum(np.square(noise_scales), np.finfo(dtype).tiny)

    num_segments = len(offsets) - 1
    lengths = np.diff(offsets)
    poses = np.zeros((num_frames, num_segments, 4, 4), dtype=dtype)
        
    if accum is not None:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = [a.copy() for a in accum]
    else:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = \
            np.zeros((num_points,), dtype=dtype), \
            np.zeros((num_points, 3), dtype=dtype), \
            np.zeros((num_points, 3, 3), dtype=dtype), \
            np.zeros((num_points,), dtype=dtype), \
            np.zeros((num_points, 3), dtype=dtype), \
            np.zeros((num_points, 3, 3), dtype=dtype), \
            np.zeros((num_points,), dtype=dtype)
    
    for i in range(num_frames):
        # Compute weighted statistics
        mean_sqrtwx = accum_sqrtwx / np.maximum(accum_sqrtw, np.finfo(trajectories.dtype).tiny)[..., None]
        wi, yi = weights[i], trajectories[i]
        sqrtwi = np.sqrt(wi)
        w = sqrtwi * accum_sqrtw
        sum_w = np.add.reduceat(w, offsets[:-1], axis=0) + np.finfo(dtype).tiny
        center_x = np.add.reduceat(sqrtwi[:, None] * accum_sqrtwx, offsets[:-1], axis=0) / sum_w[:, None]
        center_y = np.add.reduceat(w[:, None] * yi, offsets[:-1], axis=0) / sum_w[:, None]
        center_x_broadcast = np.repeat(center_x, lengths, axis=0)
        center_y_broadcast = np.repeat(center_y, lengths, axis=0)
        xc = mean_sqrtwx - center_x_broadcast
        yc = yi - center_y_broadcast
        cov_yx = np.add.reduceat(w[:, None, None] * vector_outer(yc, xc), offsets[:-1], axis=0) / sum_w[:, None, None]
        if mode == 'affine' or mode == 'similar':
            cov_xx = np.add.reduceat(sqrtwi[:, None, None] * accum_sqrtwxx + w[:, None, None] * vector_outer(xc), offsets[:-1], axis=0) / sum_w[:, None, None]
            cov_yy = np.add.reduceat(w[:, None, None] * vector_outer(yc), offsets[:-1], axis=0) / sum_w[:, None, None]
        
        # Solve for pose
        if mode == 'rigid':
            _, R, t = umeyama(cov_yx, mean_x=center_x, mean_y=center_y)
            poses[i] = make_affine_matrix(R, t)
        elif mode == 'similar':
            s, R, t = umeyama(cov_yx, cov_xx=cov_xx, mean_x=center_x, mean_y=center_y)
            poses[i] = make_affine_matrix(s * R, t)
        elif mode == 'affine':
            A, t = affine_umeyama(cov_yx, cov_xx, cov_yy, center_x, center_y, lam=lam)
            poses[i] = make_affine_matrix(A, t)

        xi = transform_points(yi, np.repeat(safe_inv(poses[i]), lengths, axis=0))

        # Update accum
        old_mean_sqrtwx, old_accum_sqrtw = mean_sqrtwx.copy(), accum_sqrtw.copy()
        accum_sqrtw += sqrtwi
        accum_sqrtwx += sqrtwi[..., None] * xi
        mean_sqrtwx = accum_sqrtwx / np.maximum(accum_sqrtw, np.finfo(dtype).tiny)[..., None]
        accum_sqrtwxx += old_accum_sqrtw[..., None, None] * vector_outer(mean_sqrtwx - old_mean_sqrtwx) + sqrtwi[..., None, None] * vector_outer(xi - mean_sqrtwx)

        mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[..., None]
        old_mean_wx, old_accum_w = mean_wx.copy(), accum_w.copy()
        accum_w += wi
        accum_wx += wi[..., None] * xi
        mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[..., None]
        accum_wxx += old_accum_w[..., None, None] * vector_outer(mean_wx - old_mean_wx) + wi[..., None, None] * vector_outer(xi - mean_wx)
        accum_nnz += wi > 0

    tot_w = np.add.reduceat(accum_w, offsets[:-1], axis=0)
    mu = np.add.reduceat(accum_wx, offsets[:-1], axis=0) / np.maximum(tot_w, np.finfo(dtype).tiny)[:, None]
    mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[:, None]
    mu_broadcast = np.repeat(mu, lengths, axis=0)
    sigma = np.add.reduceat(accum_wxx + accum_w[:, None, None] * vector_outer(mu_broadcast - mean_wx), offsets[:-1], axis=0) / np.maximum(tot_w, np.finfo(dtype).tiny)[:, None, None]
    nnz = np.add.reduceat(accum_nnz, offsets[:-1], axis=0)
    valid = np.add.reduceat(weights > 0, offsets[:-1], axis=1) >= min_valid_size
    err = np.sqrt(np.trace(accum_wxx, axis1=-2, axis2=-1) / np.maximum(accum_nnz, np.finfo(dtype).tiny))

    return poses, valid, (mu, sigma, tot_w, nnz), mean_wx, err, (accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz)