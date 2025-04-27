import numpy as np

def oversample_minority_classes(X, y, target_count=None):
    """
    Oversamples minority classes to balance dataset.
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    rng = np.random.default_rng(seed=42)

    if target_count is None:
        target_count = max(class_counts)  # Default to matching majority class

    X_oversampled = []
    y_oversampled = []

    for cls in unique_classes:
        idx = np.nonzero(y == cls)[0]
        n_samples_needed = target_count - len(idx)

        if n_samples_needed > 0:
            sampled_idx = rng.choice(idx, n_samples_needed, replace=True)
            X_oversampled.append(np.concatenate([X[idx], X[sampled_idx]], axis=0))
            y_oversampled.append(np.concatenate([y[idx], y[sampled_idx]], axis=0))
        else:
            X_oversampled.append(X[idx])
            y_oversampled.append(y[idx])

    X_final = np.concatenate(X_oversampled, axis=0)
    y_final = np.concatenate(y_oversampled, axis=0)

    return X_final, y_final