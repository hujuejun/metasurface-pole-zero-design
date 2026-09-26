import numpy as np
import pytest

from pzid.roots import (
    match_root_sets,
    refine_scalar_root,
    refine_scalar_root_complex_secant,
    refine_scalar_root_multistart,
    refine_scalar_roots,
)


def test_refine_two_analytic_poles_and_zeros() -> None:
    poles = np.asarray((1.8 - 0.08j, 2.5 - 0.03j))
    zeros = np.asarray((1.82 + 0.02j, 2.47 - 0.01j))

    def response(frequency: complex) -> complex:
        return complex(
            np.prod(frequency - zeros) / np.prod(frequency - poles)
        )

    pole_results = refine_scalar_roots(
        response,
        (1.75 - 0.05j, 2.49 - 0.04j),
        kind="pole",
    )
    zero_results = refine_scalar_roots(
        response,
        (1.78 + 0.01j, 2.50 - 0.02j),
        kind="zero",
    )
    assert all(result.success for result in (*pole_results, *zero_results))
    assert np.allclose([result.frequency for result in pole_results], poles)
    assert np.allclose([result.frequency for result in zero_results], zeros)


@pytest.mark.parametrize("kind,target", (("pole", 1.8 - 0.08j), ("zero", 1.82 + 0.02j)))
def test_complex_secant_refines_analytic_channel_root(kind: str, target: complex) -> None:
    pole = 1.8 - 0.08j
    zero = 1.82 + 0.02j
    result = refine_scalar_root_complex_secant(
        lambda frequency: (frequency - zero) / (frequency - pole),
        target + 0.03 + 0.01j,
        target - 0.02 + 0.015j,
        kind=kind,
        search_radius=0.1,
        tolerance=1e-11,
    )
    assert result.success
    assert result.frequency == pytest.approx(target, abs=1e-10)
    assert result.evaluations < 12


def test_match_root_sets_uses_global_scaled_assignment() -> None:
    references = (1.0 - 0.1j, 2.0 - 0.01j)
    candidates = (2.01 - 0.012j, 0.98 - 0.09j)
    assert match_root_sets(references, candidates, scales=(0.2, 0.02)) == (1, 0)


def test_refinement_rejects_duplicate_convergence() -> None:
    with pytest.raises(ValueError, match="same scalar root"):
        refine_scalar_roots(
            lambda frequency: frequency - (1.0 + 0.2j),
            (0.9 + 0.1j, 1.1 + 0.3j),
            kind="zero",
        )


def test_bounded_refinement_stays_in_local_root_box() -> None:
    result = refine_scalar_root(
        lambda frequency: (frequency - 1.0) * (frequency - 3.0),
        1.2 + 0.1j,
        kind="zero",
        search_radius=0.5,
    )
    assert result.success
    assert result.frequency == pytest.approx(1.0 + 0.0j, abs=1e-9)


def test_multistart_selects_true_root_over_bounded_stationary_point() -> None:
    result = refine_scalar_root_multistart(
        lambda frequency: frequency - (2.0 + 0.1j),
        (0.0j, 1.9 + 0.0j),
        kind="zero",
        search_radius=0.3,
    )
    assert result.success
    assert result.frequency == pytest.approx(2.0 + 0.1j, abs=1e-9)
    assert result.residual_norm < 1e-9


def test_multistart_can_prefer_a_real_axis_root() -> None:
    result = refine_scalar_root_multistart(
        lambda frequency: (frequency - 1.0) * (frequency - (2.0 + 0.4j)),
        (2.0 + 0.3j, 1.1 + 0.0j),
        kind="zero",
        search_radius=0.5,
        prefer_real_axis=True,
    )
    assert result.success
    assert result.frequency == pytest.approx(1.0 + 0.0j, abs=1e-8)
