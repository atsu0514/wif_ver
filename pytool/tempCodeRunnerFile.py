def match_rate_percent(true_vals: np.ndarray, pred_vals: np.ndarray, tol_abs: float = 3.0) -> float:
    """
    真値の±tol_abs（例: 3.0）以内を「一致」とみなした割合(%)。
    """
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    abs_err = np.abs(true_vals - pred_vals)
    return float((abs_err <= tol_abs).mean() * 100.0)
