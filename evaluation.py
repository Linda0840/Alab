### This Evaluation Function only requires to change threshold for proper evaluations ############
### It is interpretable as it will compare your feature with every threshold I set in this scripy ######
### DO NOT CHANGE OTHER IMPLEMENTATIONS ###################################

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def evaluate_fraud_model(y_true, y_pred_prob, threshold=0.5):
    """
    Evaluate model performance for fraud detection.
    Terminates if NaN values are detected in either input array.
    Reports specific reason(s) if rejected.
    """
    # Convert to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)

    # Check for NaNs
    if np.isnan(y_true).any() or np.isnan(y_pred_prob).any():
        raise ValueError("❌ Terminated: Input arrays contain NaN values. Please clean your data before evaluation.")

    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob >= threshold).astype(int)

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
