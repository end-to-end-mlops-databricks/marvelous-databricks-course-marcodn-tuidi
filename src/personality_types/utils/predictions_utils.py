from typing import Any, Dict

import numpy as np


def custom_predictions(
    y_pred: np.ndarray, y_pred_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Generates a dictionary containing predicted classes and their associated
    maximum probabilities.

    Args:
        y_pred (np.ndarray): Array of predicted class labels.
        y_pred_prob (np.ndarray): Array of predicted class probabilities

    Returns:
        Dict[str, Any]: A dictionary with:
            - "class" (np.ndarray): The predicted class labels.
            - "prob" (np.ndarray): The maximum probability for each sample.
    """
    y_pred_max_prob = np.max(y_pred_prob, axis=1)
    out_dict = {"class": y_pred, "prob": y_pred_max_prob}
    return out_dict
