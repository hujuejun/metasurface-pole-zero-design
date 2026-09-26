"""Local refinement and continuation matching of scalar-channel roots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, linear_sum_assignment, root


ComplexResponse = Callable[[complex], complex]
RootKind = Literal["zero", "pole"]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ScalarRootResult:
    """One independently refined zero or pole of a scalar response."""

    frequency: complex
    kind: RootKind
    success: bool
    message: str
    evaluations: int
    residual_norm: float


def refine_scalar_root_complex_secant(
    response: ComplexResponse,
    first: complex,
    second: complex,
    *,
    kind: RootKind,
    tolerance: float = 1e-8,
    max_evaluations: int = 20,
    search_radius: float | None = None,
) -> ScalarRootResult:
    """Refine an analytic scalar root with one new solve per iteration.

    Unlike a generic two-real-variable finite-difference root finder, the
    complex secant update uses analyticity and is therefore appropriate for
    expensive complex-frequency Maxwell solves.  For a pole, the reciprocal
    response is refined.  An optional circular trust region is centered on
    ``first`` and clips, rather than silently accepts, divergent updates.
    """

    if kind not in {"zero", "pole"}:
        raise ValueError("kind must be 'zero' or 'pole'")
    if tolerance <= 0.0 or max_evaluations < 2:
        raise ValueError("root controls must be positive")
    if search_radius is not None and search_radius <= 0.0:
        raise ValueError("search_radius must be positive")
    center = complex(first)
    previous = complex(first)
    current = complex(second)
    if not np.isfinite(previous) or not np.isfinite(current) or previous == current:
        raise ValueError("two distinct finite starting frequencies are required")

    def value(frequency: complex) -> complex:
        result = complex(response(frequency))
        return 1.0 / result if kind == "pole" else result

    previous_value = value(previous)
    current_value = value(current)
    evaluations = 2
    message = "maximum evaluations reached"
    success = abs(current_value) <= tolerance
    while not success and evaluations < max_evaluations:
        denominator = current_value - previous_value
        scale = max(abs(current_value), abs(previous_value), 1.0)
        if abs(denominator) <= 1e-14 * scale:
            message = "complex secant denominator vanished"
            break
        candidate = current - current_value * (current - previous) / denominator
        if search_radius is not None and abs(candidate - center) > search_radius:
            candidate = center + search_radius * (candidate - center) / abs(
                candidate - center
            )
        if abs(candidate - current) <= tolerance * max(abs(candidate), 1.0):
            current = candidate
            current_value = value(current)
            evaluations += 1
            success = abs(current_value) <= max(tolerance, 1e-10)
            message = (
                "frequency and residual converged"
                if success
                else "frequency step stagnated"
            )
            break
        previous, previous_value = current, current_value
        current = candidate
        current_value = value(current)
        evaluations += 1
        success = abs(current_value) <= tolerance
    if success and message == "maximum evaluations reached":
        message = "complex residual converged"
    return ScalarRootResult(
        frequency=current,
        kind=kind,
        success=success,
        message=message,
        evaluations=evaluations,
        residual_norm=float(abs(current_value)),
    )


def refine_scalar_root(
    response: ComplexResponse,
    initial: complex,
    *,
    kind: RootKind,
    tolerance: float = 1e-10,
    max_evaluations: int = 100,
    search_radius: float | None = None,
) -> ScalarRootResult:
    """Refine one scalar zero, or a pole through the reciprocal response."""

    if kind not in {"zero", "pole"}:
        raise ValueError("kind must be 'zero' or 'pole'")
    if tolerance <= 0.0 or max_evaluations < 1:
        raise ValueError("root controls must be positive")
    if search_radius is not None and search_radius <= 0.0:
        raise ValueError("search_radius must be positive")
    initial_value = complex(initial)
    if not np.isfinite(initial_value):
        raise ValueError("initial root guess must be finite")
    scale = max(abs(initial_value.real), 1.0)

    def equations(values: NDArray[np.float64]) -> NDArray[np.float64]:
        frequency = scale * complex(values[0], values[1])
        value = complex(response(frequency))
        if kind == "pole":
            value = 1.0 / value
        return np.asarray((value.real, value.imag), dtype=float)

    initial_vector = np.asarray(
        (initial_value.real / scale, initial_value.imag / scale)
    )
    if search_radius is None:
        solution = root(
            equations,
            initial_vector,
            method="hybr",
            options={"xtol": tolerance, "maxfev": max_evaluations},
        )
        evaluations = int(solution.nfev)
    else:
        normalized_radius = float(search_radius) / scale
        solution = least_squares(
            equations,
            initial_vector,
            bounds=(
                initial_vector - normalized_radius,
                initial_vector + normalized_radius,
            ),
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
            max_nfev=max_evaluations,
        )
        evaluations = int(solution.nfev)
    frequency = scale * complex(solution.x[0], solution.x[1])
    return ScalarRootResult(
        frequency=frequency,
        kind=kind,
        success=bool(solution.success),
        message=str(solution.message),
        evaluations=evaluations,
        residual_norm=float(np.linalg.norm(solution.fun)),
    )


def refine_scalar_roots(
    response: ComplexResponse,
    guesses: Sequence[complex],
    *,
    kind: RootKind,
    tolerance: float = 1e-10,
    max_evaluations: int = 100,
    search_radii: Sequence[float] | None = None,
) -> tuple[ScalarRootResult, ...]:
    """Refine a collection of roots while retaining the supplied identities."""

    if search_radii is None:
        radii: tuple[float | None, ...] = (None,) * len(guesses)
    else:
        if len(search_radii) != len(guesses):
            raise ValueError("search_radii must contain one value per guess")
        radii = tuple(float(value) for value in search_radii)
    results = tuple(
        refine_scalar_root(
            response,
            guess,
            kind=kind,
            tolerance=tolerance,
            max_evaluations=max_evaluations,
            search_radius=radius,
        )
        for guess, radius in zip(guesses, radii, strict=True)
    )
    frequencies = np.asarray(
        [result.frequency for result in results], dtype=np.complex128
    )
    for index in range(frequencies.size):
        for other in range(index):
            scale = max(abs(frequencies[index]), abs(frequencies[other]), 1.0)
            if abs(frequencies[index] - frequencies[other]) < 1e-8 * scale:
                raise ValueError(
                    "multiple guesses converged to the same scalar root"
                )
    return results


def refine_scalar_root_multistart(
    response: ComplexResponse,
    initials: Sequence[complex],
    *,
    kind: RootKind,
    tolerance: float = 1e-10,
    max_evaluations: int = 100,
    search_radius: float | None = None,
    prefer_real_axis: bool = False,
) -> ScalarRootResult:
    """Return the lowest-residual refinement from several local guesses.

    A bounded least-squares solve can report convergence at its search-box
    boundary even when no root is present there.  Ranking independent local
    refinements by the actual complex-response residual makes scalar-channel
    verification robust to that benign optimizer status.
    """

    guesses = tuple(complex(value) for value in initials)
    if not guesses:
        raise ValueError("initials must contain at least one guess")
    candidates = []
    residual_ceiling = max(100.0 * tolerance, 1e-8)
    for guess in guesses:
        candidate = refine_scalar_root(
            response,
            guess,
            kind=kind,
            tolerance=tolerance,
            max_evaluations=max_evaluations,
            search_radius=search_radius,
        )
        candidates.append(candidate)
        if (
            prefer_real_axis
            and candidate.success
            and candidate.residual_norm <= residual_ceiling
            and abs(candidate.frequency.imag) <= 1e-7
        ):
            break
    if prefer_real_axis:
        viable = tuple(
            result
            for result in candidates
            if result.success and result.residual_norm <= residual_ceiling
        )
        if viable:
            return min(
                viable,
                key=lambda result: (
                    abs(result.frequency.imag),
                    result.residual_norm,
                    abs(result.frequency - guesses[0]),
                ),
            )
    return min(
        tuple(candidates),
        key=lambda result: (
            result.residual_norm,
            not result.success,
            abs(result.frequency - guesses[0]),
        ),
    )


def match_root_sets(
    references: Sequence[complex],
    candidates: Sequence[complex],
    *,
    scales: Sequence[float] | None = None,
) -> tuple[int, ...]:
    """Match continued roots by minimum scaled complex-frequency distance.

    The returned tuple contains the candidate index assigned to each reference
    root.  Using linewidths as ``scales`` prevents a broad resonance from
    dominating a nearby narrow one during continuation.
    """

    reference_values = np.asarray(references, dtype=np.complex128)
    candidate_values = np.asarray(candidates, dtype=np.complex128)
    if reference_values.ndim != 1 or candidate_values.ndim != 1:
        raise ValueError("root sets must be one-dimensional")
    if reference_values.size != candidate_values.size or reference_values.size == 0:
        raise ValueError("root sets must have the same nonzero size")
    if scales is None:
        scale_values = np.ones(reference_values.size, dtype=float)
    else:
        scale_values = np.asarray(scales, dtype=float)
        if scale_values.shape != reference_values.shape:
            raise ValueError("scales must contain one value per reference root")
        if np.any(~np.isfinite(scale_values)) or np.any(scale_values <= 0.0):
            raise ValueError("root-matching scales must be finite and positive")
    costs = np.abs(
        reference_values[:, np.newaxis] - candidate_values[np.newaxis, :]
    ) / scale_values[:, np.newaxis]
    rows, columns = linear_sum_assignment(costs)
    assignment = np.empty(reference_values.size, dtype=int)
    assignment[rows] = columns
    return tuple(int(value) for value in assignment)
