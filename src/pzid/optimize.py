"""Small-residual-space Levenberg-Marquardt optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


RealArray = NDArray[np.float64]
ResidualFunction = Callable[[RealArray], tuple[RealArray, RealArray]]


@dataclass(frozen=True)
class LMOptions:
    """Controls for bounded underdetermined Levenberg-Marquardt."""

    max_iterations: int = 100
    max_attempts_per_iteration: int = 12
    initial_damping: float = 0.1
    damping_increase: float = 10.0
    damping_decrease: float = 0.3
    minimum_damping: float = 1e-12
    residual_tolerance: float = 1e-8
    step_tolerance: float = 1e-10
    active_set_bounds: bool = False
    active_set_max_iterations: int = 20
    bound_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.max_attempts_per_iteration < 1:
            raise ValueError("iteration limits must be positive")
        if self.active_set_max_iterations < 1:
            raise ValueError("active_set_max_iterations must be positive")
        if self.initial_damping <= 0.0:
            raise ValueError("initial_damping must be positive")
        if self.damping_increase <= 1.0:
            raise ValueError("damping_increase must exceed one")
        if not 0.0 < self.damping_decrease < 1.0:
            raise ValueError("damping_decrease must lie inside (0, 1)")
        if self.minimum_damping <= 0.0:
            raise ValueError("minimum_damping must be positive")
        if self.residual_tolerance <= 0.0 or self.step_tolerance <= 0.0:
            raise ValueError("tolerances must be positive")
        if self.bound_tolerance < 0.0:
            raise ValueError("bound_tolerance must be nonnegative")


@dataclass(frozen=True)
class LMIteration:
    """Accepted iteration record."""

    iteration: int
    cost: float
    residual_norm: float
    step_norm: float
    damping: float
    rejected_attempts: int


@dataclass(frozen=True)
class LMResult:
    """Optimization result and full accepted-step history."""

    x: RealArray
    residual: RealArray
    jacobian: RealArray
    history: tuple[LMIteration, ...]
    converged: bool
    reason: str


def complex_system_to_real(
    residual: ArrayLike,
    jacobian: ArrayLike,
) -> tuple[RealArray, RealArray]:
    """Stack complex equalities as consecutive real and imaginary rows."""

    complex_residual = np.asarray(residual, dtype=np.complex128).ravel()
    complex_jacobian = np.asarray(jacobian, dtype=np.complex128)
    if complex_jacobian.shape[0] != complex_residual.size:
        raise ValueError("jacobian first axis must match the residual size")
    flattened_jacobian = complex_jacobian.reshape(complex_residual.size, -1)
    return (
        np.concatenate((complex_residual.real, complex_residual.imag)),
        np.vstack((flattened_jacobian.real, flattened_jacobian.imag)),
    )


def underdetermined_lm_step(
    residual: ArrayLike,
    jacobian: ArrayLike,
    damping: float,
) -> RealArray:
    r"""Return the Chen et al. dual-space damped least-squares step.

    This implements their Eq. (20), with
    ``Delta = ||J||_F**2 / n_variables * I``. The residual count must not
    exceed the variable count.
    """

    values = np.asarray(residual, dtype=float).ravel()
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != values.size:
        raise ValueError("jacobian must have one row per residual")
    equations, variables = matrix.shape
    if equations > variables:
        raise ValueError("the dual update requires an underdetermined system")
    if damping < 0.0:
        raise ValueError("damping must be nonnegative")
    scale = float(np.linalg.norm(matrix, ord="fro") ** 2 / variables)
    if scale == 0.0:
        raise ValueError("cannot form a step from a zero Jacobian")
    dual_matrix = matrix @ matrix.T + damping * scale * np.eye(equations)
    dual = np.linalg.solve(dual_matrix, -values)
    return np.asarray(matrix.T @ dual)


def damped_least_squares_step(
    residual: ArrayLike,
    jacobian: ArrayLike,
    damping: float,
) -> RealArray:
    """Return a damped LM step for either residual/variable ordering.

    The dual Chen update is retained for an underdetermined system.  A small
    physical-parameter problem can instead have more declared residual rows
    than coordinates, in which case the mathematically equivalent damped
    primal system is used.
    """

    values = np.asarray(residual, dtype=float).ravel()
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != values.size:
        raise ValueError("jacobian must have one row per residual")
    equations, variables = matrix.shape
    if equations <= variables:
        return underdetermined_lm_step(values, matrix, damping)
    if damping < 0.0:
        raise ValueError("damping must be nonnegative")
    scale = float(np.linalg.norm(matrix, ord="fro") ** 2 / variables)
    if scale == 0.0:
        raise ValueError("cannot form a step from a zero Jacobian")
    primal = matrix.T @ matrix + damping * scale * np.eye(variables)
    return np.asarray(np.linalg.solve(primal, -(matrix.T @ values)))


def active_set_underdetermined_lm_step(
    residual: ArrayLike,
    jacobian: ArrayLike,
    x: ArrayLike,
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
    damping: float,
    *,
    max_iterations: int = 20,
    bound_tolerance: float = 1e-12,
) -> RealArray:
    """Return a dual LM step after removing outward-pointing bound variables.

    A posteriori clipping of a minimum-norm underdetermined step can retain a
    large component in directions that are unavailable at binary density
    bounds.  This small active-set loop recomputes the dual step after freezing
    variables whose proposed direction points outside their current bound.
    It leaves interior variables and inward-pointing boundary variables free.
    """

    values = np.asarray(residual, dtype=float).ravel()
    matrix = np.asarray(jacobian, dtype=float)
    current = np.asarray(x, dtype=float).ravel()
    lower = np.broadcast_to(np.asarray(lower_bound, dtype=float), current.shape)
    upper = np.broadcast_to(np.asarray(upper_bound, dtype=float), current.shape)
    if matrix.ndim != 2 or matrix.shape != (values.size, current.size):
        raise ValueError("jacobian shape must match residual and x")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if bound_tolerance < 0.0:
        raise ValueError("bound_tolerance must be nonnegative")

    free = np.ones(current.size, dtype=bool)
    step = np.zeros_like(current)
    for _ in range(max_iterations):
        if not np.any(free):
            step.fill(0.0)
            return step
        reduced = matrix[:, free]
        if not np.any(reduced):
            # All remaining directions are locally insensitive.  A zero step
            # lets the outer solver terminate cleanly by its step-tolerance
            # rule instead of raising from an empty/zero reduced system.
            step.fill(0.0)
            return step
        if np.count_nonzero(free) < values.size:
            # The M2/M5 systems are strongly underdetermined, so reaching this
            # branch means that the active bounds have removed that property.
            # A damped primal solve remains well defined in that rare case.
            scale = float(np.linalg.norm(reduced, ord="fro") ** 2 / values.size)
            primal = reduced.T @ reduced + damping * scale * np.eye(
                reduced.shape[1]
            )
            reduced_step = np.linalg.solve(primal, -(reduced.T @ values))
        else:
            reduced_step = underdetermined_lm_step(
                values, reduced, damping
            )
        step.fill(0.0)
        step[free] = reduced_step
        blocked = free & (
            ((current <= lower + bound_tolerance) & (step < 0.0))
            | ((current >= upper - bound_tolerance) & (step > 0.0))
        )
        if not np.any(blocked):
            return step
        free[blocked] = False
    return step


def solve_underdetermined_lm(
    function: ResidualFunction,
    initial_x: ArrayLike,
    *,
    lower_bound: ArrayLike | float = 0.0,
    upper_bound: ArrayLike | float = 1.0,
    options: LMOptions | None = None,
    allow_overdetermined: bool = False,
) -> LMResult:
    """Minimize a real residual system with projected, damped LM steps.

    The default preserves the Chen dual-space underdetermined formulation.
    ``allow_overdetermined`` enables a primal step for a small physical-
    coordinate problem with more residual rows than design coordinates.
    """

    controls = options or LMOptions()
    x = np.asarray(initial_x, dtype=float).ravel().copy()
    lower = np.broadcast_to(np.asarray(lower_bound, dtype=float), x.shape)
    upper = np.broadcast_to(np.asarray(upper_bound, dtype=float), x.shape)
    if np.any(lower > upper):
        raise ValueError("lower_bound must not exceed upper_bound")
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError("initial_x must satisfy the bounds")

    residual, jacobian = function(x)
    residual = np.asarray(residual, dtype=float).ravel()
    jacobian = np.asarray(jacobian, dtype=float)
    damping = max(controls.initial_damping, controls.minimum_damping)
    history: list[LMIteration] = []

    for iteration in range(controls.max_iterations):
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= controls.residual_tolerance:
            return LMResult(
                x, residual, jacobian, tuple(history), True, "residual tolerance"
            )
        cost = 0.5 * residual_norm**2
        accepted = False
        last_step_norm = 0.0
        for rejected in range(controls.max_attempts_per_iteration):
            if controls.active_set_bounds:
                step = active_set_underdetermined_lm_step(
                    residual,
                    jacobian,
                    x,
                    lower,
                    upper,
                    damping,
                    max_iterations=controls.active_set_max_iterations,
                    bound_tolerance=controls.bound_tolerance,
                )
            else:
                if allow_overdetermined:
                    step = damped_least_squares_step(
                        residual, jacobian, damping
                    )
                else:
                    step = underdetermined_lm_step(
                        residual, jacobian, damping
                    )
            trial_x = np.clip(x + step, lower, upper)
            projected_step = trial_x - x
            last_step_norm = float(np.linalg.norm(projected_step))
            if last_step_norm <= controls.step_tolerance * (
                1.0 + float(np.linalg.norm(x))
            ):
                return LMResult(
                    x, residual, jacobian, tuple(history), False, "step tolerance"
                )
            trial_residual, trial_jacobian = function(trial_x)
            trial_residual = np.asarray(trial_residual, dtype=float).ravel()
            trial_jacobian = np.asarray(trial_jacobian, dtype=float)
            trial_cost = 0.5 * float(trial_residual @ trial_residual)
            if trial_cost < cost:
                x = trial_x
                residual = trial_residual
                jacobian = trial_jacobian
                history.append(
                    LMIteration(
                        iteration=iteration,
                        cost=trial_cost,
                        residual_norm=float(np.linalg.norm(trial_residual)),
                        step_norm=last_step_norm,
                        damping=damping,
                        rejected_attempts=rejected,
                    )
                )
                damping = max(
                    damping * controls.damping_decrease,
                    controls.minimum_damping,
                )
                accepted = True
                break
            damping *= controls.damping_increase
        if not accepted:
            return LMResult(
                x, residual, jacobian, tuple(history), False, "no acceptable step"
            )

    return LMResult(
        x, residual, jacobian, tuple(history), False, "iteration limit"
    )
