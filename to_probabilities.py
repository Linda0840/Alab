def to_probabilities(x):
    """
    Convert arbitrary feature values into normalized probabilities.
    The output sums to 1 and can be directly compared to thresholds.
    """
    x = np.asarray(x, dtype=float)
    # numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
