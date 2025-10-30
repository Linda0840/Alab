import pandas as pd

def merge_on_column(result_df: pd.DataFrame, feature_df: pd.DataFrame, merge_col: str) -> pd.DataFrame:
    """
    Perform an inner merge between result_df and feature_df based on a specified column.

    Parameters
    ----------
    result_df : pd.DataFrame
        The main dataframe with ID columns and target.
    feature_df : pd.DataFrame
        The dataframe containing a 'feature' column to merge with.
    merge_col : str
        One of ['APPROVAL_ID_INFO', 'APPLICATION_ID_INFO', 
                'FIRST_APPROVAL_ID_INFO', 'DECISION_CHAIN_ID_INFO'].

    Returns
    -------
    pd.DataFrame
        The merged dataframe with all NaN rows removed.
    """
    valid_cols = [
        'APPROVAL_ID_INFO', 
        'APPLICATION_ID_INFO', 
        'FIRST_APPROVAL_ID_INFO', 
        'DECISION_CHAIN_ID_INFO'
    ]
    
    if merge_col not in valid_cols:
        raise ValueError(f"merge_col must be one of {valid_cols}")
    
    merged = pd.merge(result_df, feature_df, how='inner', left_on=merge_col, right_on='feature')
    merged = merged.dropna().reset_index(drop=True)
    return merged
