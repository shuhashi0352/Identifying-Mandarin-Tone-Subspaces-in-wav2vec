import numpy as np

def resample_f0_helper(f0_z, n_points=10):
    """
    Resample one z-scored F0 contour to a fixed number of points.

    Parameters
    ----------
    f0_z : array-like
        1D contour with possible NaNs.
    n_points : int
        Number of normalized frame positions to resample to.

    Returns
    -------
    np.ndarray
        Shape [n_points].
    """
    f0_z = np.asarray(f0_z, dtype=float)

    # keep only voiced points
    valid_mask = ~np.isnan(f0_z)
    valid_f0 = f0_z[valid_mask]

    if len(valid_f0) == 0:
        raise ValueError("Contour has no voiced F0 values.")

    if len(valid_f0) == 1:
        return np.full(n_points, valid_f0[0], dtype=float)

    # original contour positions in normalized time
    old_x = np.linspace(0.0, 1.0, num=len(valid_f0))
    new_x = np.linspace(0.0, 1.0, num=n_points)

    # linear interpolation
    resampled = np.interp(new_x, old_x, valid_f0)

    return resampled

def resample_f0(z_results, n_points=10):
    """
    Add fixed-length interpolated contour to each item.
    """
    out = []

    for item in z_results:
        new_item = item.copy()
        new_item["f0_z_resampled"] = resample_f0_helper(item["f0_z"], n_points=n_points)
        out.append(new_item)

    n_voiced_counts = [np.sum(~np.isnan(item["f0_z"])) for item in z_results]

    print("min:", np.min(n_voiced_counts))
    print("max:", np.max(n_voiced_counts))
    print("mean:", np.mean(n_voiced_counts))
    print("median:", np.median(n_voiced_counts))

    return out