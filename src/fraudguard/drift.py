import numpy as np

# Common rule of thumb for the Population Stability Index (PSI).
PSI_WARNING = 0.10
PSI_RETRAIN = 0.25


def psi(reference, current, bins=10):
    """Population Stability Index: how much a distribution moved from `reference` to `current`.

    Bins are the reference deciles, so each holds about 10% of the reference data.
    0 means no change; above 0.10 is a moderate shift and above 0.25 a large one.
    """
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, edges)[0] / len(reference)
    current_share = np.histogram(current, edges)[0] / len(current)

    # Avoid log(0) for empty bins.
    reference_share = np.clip(reference_share, 1e-6, None)
    current_share = np.clip(current_share, 1e-6, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def drift_status(value):
    """Turn a PSI value into an action."""
    if value >= PSI_RETRAIN:
        return "retrain"
    if value >= PSI_WARNING:
        return "warning"
    return "ok"
