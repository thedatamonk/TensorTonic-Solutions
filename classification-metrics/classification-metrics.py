import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))   # find all unique classes across both y_true and y_pred
    accuracy = np.mean(y_true == y_pred) # find all instances where y_true = y_pred

    if average == "binary":
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average == "micro":
        # global TP/FP/FN aggregation across all classes
        tp_sum = 0
        fp_sum = 0
        fn_sum = 0
        # iterate over all the prediction classes
        # compute the true positives, false positives, and false negatives for each class
        # and add them to the global sums
        for c in classes:
            tp_sum += np.sum((y_pred == c) & (y_true == c))
            fp_sum += np.sum((y_pred == c) & (y_true != c))
            fn_sum += np.sum((y_pred != c) & (y_true == c))

        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average in ("macro", "weighted"):
        precisions = []
        recalls = []
        f1s = []
        supports = []

        # iterate over all the prediction classes
        # compute the true positives, false positives, and false negatives for each class
        # we also compute the precision, recall, and F1 for each class
        # and store them for each class.
        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            supports.append(np.sum(y_true == c))

        # the stored precision, recall, and F1 values for each class
        # are then averaged according to the specified method:
        # - "macro": simple average of the metrics across classes
        # - "weighted": average weighted by the number of true instances for each class (support
        if average == "macro":
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)
        else:  # weighted
            total = np.sum(supports)
            weights = np.array(supports) / total if total > 0 else np.zeros(len(supports))
            precision = np.dot(weights, precisions)
            recall = np.dot(weights, recalls)
            f1 = np.dot(weights, f1s)
    else:
        raise ValueError(f"Unknown average mode: {average!r}")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }