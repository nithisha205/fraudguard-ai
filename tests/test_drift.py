import numpy as np
import pytest

from drift import PSI_RETRAIN, PSI_WARNING, drift_status, psi

rng = np.random.default_rng(42)


def test_psi_is_near_zero_for_the_same_distribution():
    assert psi(rng.normal(size=20_000), rng.normal(size=20_000)) < 0.01


def test_psi_detects_a_shifted_distribution():
    assert psi(rng.normal(size=20_000), rng.normal(loc=1.0, size=20_000)) > PSI_RETRAIN


def test_psi_handles_values_outside_the_reference_range():
    reference = rng.uniform(0, 1, size=5_000)
    assert np.isfinite(psi(reference, reference + 10))


def test_psi_handles_repeated_values():
    # Many identical reference values give duplicate decile edges.
    reference = np.r_[np.zeros(9_000), rng.normal(size=1_000)]
    assert psi(reference, reference) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("value", "status"),
    [(0.0, "ok"), (PSI_WARNING, "warning"), (0.2, "warning"), (PSI_RETRAIN, "retrain"), (1.5, "retrain")],
)
def test_drift_status(value, status):
    assert drift_status(value) == status
