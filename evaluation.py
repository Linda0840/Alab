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

    # Reasonable thresholds for fraud detection
    precision_req = 0.7
    recall_req = 0.65
    acc_req = 0.75
    f1_req = 0.7

    # Collect unmet criteria
    failed = []
    if precision < precision_req:
        failed.append(f"Precision below threshold ({precision:.3f} < {precision_req})")
    if recall < recall_req:
        failed.append(f"Recall below threshold ({recall:.3f} < {recall_req})")
    if acc < acc_req:
        failed.append(f"Accuracy below threshold ({acc:.3f} < {acc_req})")
    if f1 < f1_req:
        failed.append(f"F1 Score below threshold ({f1:.3f} < {f1_req})")

    # Decision
    if not failed:
        decision = "PASS ✅"
        reason = "All metrics meet or exceed required thresholds."
    else:
        decision = "REJECT ❌"
        reason = " | ".join(failed)
    
    # Display results
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Decision  : {decision}")
    print(f"Reason    : {reason}")

    return {
        "Precision": precision,
        "Recall": recall,
        "Accuracy": acc,
        "F1": f1,
        "Decision": decision,
        "Reason": reason
    }
