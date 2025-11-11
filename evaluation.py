### This Evaluation Function requires your y_pred to be BINARY ############
### DO NOT CHANGE OTHER IMPLEMENTATIONS ###################################

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def evaluate_fraud_model(y_true, y_pred):
    """
    y_true & y_pred: MUST BE BINARY
    Evaluate model performance for fraud detection.
    Terminates if NaN values are detected in either input array.
    Reports specific reason(s) if rejected.
    """
    # Convert to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check for NaNs
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("❌ Terminated: Input arrays contain NaN values. Please clean your data before evaluation.")


    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "Precision": precision,
        "Recall": recall,
        "Accuracy": acc,
        "F1": f1
    }
