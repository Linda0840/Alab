import numpy as np
import pandas as pd

def downsample_flag(df, flag_col, keep_ratio=0.2, seed=None):
    """
    Downsample only the rows where df[flag_col] == True.
    Keep all other rows unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    flag_col : str
        Column containing True/False or 1/0 flags.
    keep_ratio : float
        Fraction of True-flag rows to keep.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame.
    """

    if seed is not None:
        np.random.seed(seed)

    # Boolean mask
    mask_true = df[flag_col] == True
    df_true = df[mask_true]
    df_false = df[~mask_true]

    # Number of True rows to keep
    n_true = len(df_true)
    keep_n = int(n_true * keep_ratio)

    # Randomly choose which True rows to keep
    keep_idx = np.random.choice(df_true.index, size=keep_n, replace=False)

    # Construct final df
    df_downsampled = pd.concat([df_false, df.loc[keep_idx]], axis=0)

    return df_downsampled
