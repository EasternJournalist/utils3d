import numpy as np


def linear_spline(x: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Linear spline interpolation.

    ### Parameters:
    - `x`: np.ndarray, shape (..., n,): the x-coordinates of the data points.
    - `t`: np.ndarray, shape (..., n,): the knot vector. NOTE: t must be sorted in ascending order.
    - `s`: np.ndarray, shape (..., m,): the y-coordinates of the data points.
    
    ### Returns:
    - `y`: np.ndarray, shape (..., m): the interpolated values.
    """
    


def _solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = b.shape[-1]
    cc = np.zeros_like(b)
    dd = np.zeros_like(b)
    cc[..., 0] = c[..., 0] / b[..., 0]    
    dd[..., 0] = d[..., 0] / b[..., 0]
    for i in range(1, n):
        cc[..., i] = c[..., i] / (b[..., i] - a[..., i - 1] * cc[..., i - 1])
        dd[..., i] = (d[..., i] - a[..., i - 1] * dd[..., i - 1]) / (b[..., i] - a[..., i - 1] * cc[..., i - 1])
    x = np.zeros_like(b)
    x[..., -1] = dd[..., -1]
    for i in range(n - 2, -1, -1):
        x[..., i] = dd[..., i] - cc[..., i] * x[..., i + 1]
    return x


def cubic_spline_interpolate(x: np.ndarray, t: np.ndarray, s: np.ndarray, v0: np.ndarray = None, vn: np.ndarray = None) -> np.ndarray:
    """
    Cubic spline interpolation.

    ### Parameters:
    - `x`: np.ndarray, shape (..., n,): the x-coordinates of the data points.
    - `t`: np.ndarray, shape (n,): the knot vector. NOTE: t must be sorted in ascending order.
    - `s`: np.ndarray, shape (..., m,): the y-coordinates of the data points.
    - `v0`: np.ndarray, shape (...,): the value of the derivative at the first knot, as the boundary condition. If None, it is set to zero.
    - `vn`: np.ndarray, shape (...,): the value of the derivative at the last knot, as the boundary condition. If None, it is set to zero.
    
    ### Returns:
    - `y`: np.ndarray, shape (..., m): the interpolated values.
    """
    h = t[..., 1:] - t[..., :-1]
    mu = h[..., :-1] / (h[..., :-1] + h[..., 1:])
    la = 1 - mu
    d = (x[..., 1:] - x[..., :-1]) / h
    d = 6 * (d[..., 1:] - d[..., :-1]) / (t[..., 2:] - t[..., :-2])

    mu = np.concatenate([mu, np.ones_like(mu[..., :1])], axis=-1)
    la = np.concatenate([np.ones_like(la[..., :1]), la], axis=-1)
    d = np.concatenate([(((x[..., 1] - x[..., 0]) / h[0] - v0) / h[0])[..., None], d, ((vn - (x[..., -1] - x[..., -2]) / h[-1]) / h[-1])[..., None]], axis=-1)

    M = _solve_tridiagonal(mu, np.full_like(d, fill_value=2), la, d)

    i = np.searchsorted(t, s, side='left')

