"""Adaptive HSE core evolution with a configurable exterior update policy."""

from __future__ import annotations

from typing import Any

import numpy as np

from run_hse_late import force_within_tolerance
from sidm_engine import (
    TINY,
    PhysicalParams,
    Scales,
    State,
    StepControls,
    evolve_one_step,
    predict_step_heat_update,
)
from tools.diagnose_hse_direct_solver import (
    HSEDirectProblem,
    damped_newton,
    radii_to_free_boundary_coordinates,
)


# The last movable HSE face separates two HSE cells, even though the outer face
# of the last cell is supplied by the dynamic predictor.  It must therefore be
# included in the hydrostatic force solve.  Excluding it allowed an arbitrarily
# large pressure and density cliff to pass the HSE acceptance gate.
HYBRID_HSE_FORCE_EXCLUDE_OUTER_FACES = 0
HYBRID_HSE_EXPANSION_RETRY_STEPS = 1000


def central_density_shape_metrics(
    state: State,
    core_face_end: int,
    cell_count: int = 8,
    fractional_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Measure outward density inversions in the resolved central HSE cells."""

    stop = min(max(0, int(cell_count)), int(core_face_end), state.rho.size - 1)
    if stop < 2:
        return {
            "hse_central_density_shape_ok": True,
            "hse_central_density_shape_cells": stop,
            "hse_central_density_max_log_increase": 0.0,
            "hse_central_density_max_increase_pair": -1,
            "hse_central_density_peak_cell": 0,
        }
    rho = np.maximum(state.rho[:stop], np.finfo(np.float64).tiny)
    log_increase = np.diff(np.log(rho))
    pair = int(np.argmax(log_increase))
    maximum = float(log_increase[pair])
    allowed = float(np.log1p(max(0.0, fractional_tolerance)))
    return {
        "hse_central_density_shape_ok": bool(maximum <= allowed),
        "hse_central_density_shape_cells": stop,
        "hse_central_density_max_log_increase": maximum,
        "hse_central_density_max_fractional_increase": float(np.expm1(maximum)),
        "hse_central_density_max_increase_pair": pair,
        "hse_central_density_peak_cell": int(np.argmax(rho)),
        "hse_central_density_fractional_tolerance": float(fractional_tolerance),
    }


def hse_boundary_contract_face(
    core_face_end: int,
    transition_faces: int = HYBRID_HSE_FORCE_EXCLUDE_OUTER_FACES,
) -> int:
    """Return the outermost interior face used for HSE contraction checks."""

    return max(1, int(core_face_end) - 1 - max(0, int(transition_faces)))


def hse_expansion_retry_ready(
    accepted_steps: int,
    retry_after_step: int,
    current_core_face_end: int,
    rejected_from_core_face_end: int | None,
) -> bool:
    """Permit a rejected expansion again after the state has had time to evolve."""

    return bool(
        rejected_from_core_face_end is None
        or int(current_core_face_end) != int(rejected_from_core_face_end)
        or int(accepted_steps) >= int(retry_after_step)
    )


def select_hysteretic_core_face_end(
    lambda_over_h: np.ndarray,
    previous_core_face_end: int | None,
    enter_ratio: float = 0.8,
    exit_ratio: float = 1.25,
    min_core_face_end: int = 3,
    allow_expand: bool = True,
    force_contract: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Track the contiguous SMFP front without one-cell switch chatter."""

    ratio = np.asarray(lambda_over_h, dtype=np.float64)
    if ratio.ndim != 1 or ratio.size < min_core_face_end:
        raise ValueError("lambda_over_h does not contain a resolvable core")
    if not (0.0 < enter_ratio < 1.0 < exit_ratio):
        raise ValueError("hysteresis requires 0 < enter_ratio < 1 < exit_ratio")

    raw = 0
    for value in ratio:
        if np.isfinite(value) and value <= 1.0:
            raw += 1
        else:
            break

    maximum = ratio.size
    if previous_core_face_end is None:
        selected = raw
        action = "initialize"
    else:
        selected = min(max(int(previous_core_face_end), 0), maximum)
        action = "hold"
        if force_contract and selected > 0:
            selected -= 1
            action = "contract_one_face_nonquasistatic"
        elif (
            raw > selected
            and selected < maximum
            and ratio[selected] <= enter_ratio
            and allow_expand
        ):
            selected += 1
            action = "expand_one_face"
        elif raw > selected and ratio[selected] <= enter_ratio and not allow_expand:
            action = "hold_expansion_not_quasistatic"
        elif raw < selected and selected > 0 and ratio[selected - 1] >= exit_ratio:
            selected -= 1
            action = "contract_one_face"

    if selected < min_core_face_end:
        if previous_core_face_end is None:
            selected = 0
            action = "disable_unresolved_core"
        else:
            selected = min_core_face_end
            action = "hold_minimum_hse_core"
    metrics = {
        "hybrid_smfp_raw_core_face_end": int(raw),
        "hybrid_previous_core_face_end": (
            int(previous_core_face_end)
            if previous_core_face_end is not None
            else None
        ),
        "hybrid_selected_core_face_end": int(selected),
        "hybrid_boundary_action": action,
        "hybrid_boundary_expand_allowed": bool(allow_expand),
        "hybrid_boundary_force_contract": bool(force_contract),
        "hybrid_hysteresis_enter_lambda_over_h": float(enter_ratio),
        "hybrid_hysteresis_exit_lambda_over_h": float(exit_ratio),
        "hybrid_boundary_last_inside_lambda_over_h": (
            float(ratio[selected - 1]) if selected > 0 else None
        ),
        "hybrid_boundary_first_outside_lambda_over_h": (
            float(ratio[selected]) if 0 <= selected < maximum else None
        ),
    }
    return selected, metrics


def evolve_adaptive_hybrid_step(
    state: State,
    a_face: np.ndarray,
    dt_t0: float,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls,
    core_face_end: int,
    force_exclude_boundary_faces: int,
    fixed_point_iterations: int,
    max_newton_iterations: int,
    finite_difference_step: float,
    trust_radius: float,
    min_line_search: float,
    force_max_tol: float,
    force_median_tol: float,
    closure_tol: float = 1.0e-3,
    central_density_shape_cells: int = 8,
    central_density_inversion_tol: float = 0.01,
    gamma_a1_power: int = 2,
    exterior_update_policy: str = "dynamic",
) -> tuple[State, dict[str, Any]]:
    """Advance one step with an HSE core and a selected exterior policy.

    With ``dynamic``, faces strictly outside ``core_face_end`` follow the normal
    momentum update.  With ``frozen``, their radii, velocities, and thermal
    state are held at the old state and serve only as the outer HSE reservoir.
    The SMFP boundary radius is solved as a free HSE coordinate using the first
    exterior cell as the pressure-matching condition.  The heat predictor is
    evaluated from the old state once and supplied to both parts of the split
    step.
    """

    n_face = state.r.size
    core_face_end = int(core_face_end)
    if not 3 <= core_face_end <= n_face:
        raise ValueError(
            f"core_face_end must be in [3, {n_face}], got {core_face_end}"
        )
    if not np.isfinite(dt_t0) or dt_t0 <= 0.0:
        raise ValueError("dt_t0 must be finite and positive")
    if exterior_update_policy not in {"dynamic", "frozen"}:
        raise ValueError(
            "exterior_update_policy must be either 'dynamic' or 'frozen'"
        )

    frozen_exterior = exterior_update_policy == "frozen"
    if frozen_exterior:
        exterior_trial = state.copy()
        q_new = None
        delta_eps_heat = None
    else:
        q_new, delta_eps_heat = predict_step_heat_update(
            state,
            a_face,
            dt_t0,
            params,
            scales,
            controls,
        )
        exterior_trial = evolve_one_step(
            state,
            a_face,
            dt_t0,
            params,
            scales,
            controls,
            dynamic_face_start=core_face_end + 1,
            q_new_override=q_new,
            delta_eps_heat_override=delta_eps_heat,
        )

    problem = HSEDirectProblem(
        state,
        a_face,
        dt_t0,
        params,
        controls,
        core_face_end,
        force_exclude_boundary_faces,
        fixed_point_iterations,
        freeze_exterior_thermal=frozen_exterior,
        radius_template=exterior_trial.r,
        q_new_override=q_new,
        delta_eps_heat_override=delta_eps_heat,
        exterior_state=exterior_trial,
        force_exclude_inner=force_exclude_boundary_faces,
        force_exclude_outer=HYBRID_HSE_FORCE_EXCLUDE_OUTER_FACES,
        close_excluded_residuals=True,
        gamma_a1_power=gamma_a1_power,
        free_outer_boundary=True,
    )
    q_new = problem.q_new.copy()
    coordinates = radii_to_free_boundary_coordinates(
        exterior_trial.r,
        core_face_end,
    )
    _, projected, history = damped_newton(
        problem,
        coordinates,
        max_newton_iterations,
        finite_difference_step,
        trust_radius,
        min_line_search,
        force_max_tol,
        force_median_tol,
        minimum_iterations=0,
        closure_tolerance=closure_tol,
    )
    final = history[-1]

    if projected.r[core_face_end] >= exterior_trial.r[core_face_end + 1]:
        raise RuntimeError("free HSE boundary crossed the first dynamic face")

    projected.u[:core_face_end] = 0.0
    dt_code = dt_t0 * scales.dt_code_per_t0
    projected.u[core_face_end] = (
        (projected.r[core_face_end] - state.r[core_face_end])
        * scales.r0_scale**2
        / (max(float(projected.ephi[core_face_end]), TINY) * dt_code)
    )
    projected.u[core_face_end + 1 :] = exterior_trial.u[core_face_end + 1 :]
    projected.r[core_face_end + 1 :] = exterior_trial.r[core_face_end + 1 :]

    exterior_radius_exact = bool(
        np.array_equal(
            projected.r[core_face_end + 1 :],
            exterior_trial.r[core_face_end + 1 :],
        )
    )
    exterior_velocity_exact = bool(
        np.array_equal(
            projected.u[core_face_end + 1 :],
            exterior_trial.u[core_face_end + 1 :],
        )
    )
    heat_flux_exact = bool(np.array_equal(projected.q, q_new))
    if not exterior_radius_exact or not exterior_velocity_exact or not heat_flux_exact:
        raise RuntimeError("hybrid HSE projection violated an interface invariant")

    density_shape = central_density_shape_metrics(
        projected,
        core_face_end,
        cell_count=central_density_shape_cells,
        fractional_tolerance=central_density_inversion_tol,
    )
    closure_converged = force_within_tolerance(final["closure_max"], closure_tol)
    last_hse_cell = core_face_end - 1
    first_exterior_cell = core_face_end

    def log10_jump(values: np.ndarray) -> float:
        left = max(abs(float(values[last_hse_cell])), TINY)
        right = max(abs(float(values[first_exterior_cell])), TINY)
        return float(abs(np.log10(left / right)))

    previous_hse_width = max(
        float(projected.r[core_face_end - 1] - projected.r[core_face_end - 2]),
        TINY,
    )
    boundary_hse_width = max(
        float(projected.r[core_face_end] - projected.r[core_face_end - 1]),
        TINY,
    )
    first_exterior_width = max(
        float(projected.r[core_face_end + 1] - projected.r[core_face_end]),
        TINY,
    )
    metrics: dict[str, Any] = {
        "hse_solver": "adaptive-hybrid-direct-dense",
        "hse_iterations": int(final["iteration"]),
        "hse_minimum_iterations": 0,
        "hse_initial_projection_accepted": bool(
            int(final["iteration"]) == 0 and final.get("converged", False)
        ),
        "hse_evaluations": int(problem.evaluations),
        "hse_converged": bool(
            final.get("converged", False)
            and closure_converged
            and density_shape["hse_central_density_shape_ok"]
        ),
        "hse_converged_after_stalled_required_iteration": bool(
            final.get("converged_after_stalled_required_iteration", False)
        ),
        "hse_force_converged": bool(
            force_within_tolerance(final["force_max"], force_max_tol)
            and force_within_tolerance(final["force_median"], force_median_tol)
        ),
        "hse_force_residual_initial_max": float(history[0]["force_max"]),
        "hse_force_residual_initial_max_face": int(
            history[0]["force_max_face"]
        ),
        "hse_force_residual_initial_median": float(history[0]["force_median"]),
        "hse_force_residual_final_max": float(final["force_max"]),
        "hse_force_residual_final_max_face": int(final["force_max_face"]),
        "hse_force_gate_first_face": int(final["force_gate_first_face"]),
        "hse_force_gate_last_face": int(final["force_gate_last_face"]),
        "hse_force_residual_final_median": float(final["force_median"]),
        "hse_force_residual_full_movable_max": float(
            final["full_movable_force_max"]
        ),
        "hse_interface_force_residual": float(final["interface_force_residual"]),
        "hse_force_max_tol": float(force_max_tol),
        "hse_force_median_tol": float(force_median_tol),
        "hse_closure_converged": closure_converged,
        "hse_closure_max": float(final["closure_max"]),
        "hse_closure_tol": float(closure_tol),
        "hse_central_shape_drift_max": float(final["central_shape_drift_max"]),
        "hse_central_shape_constraint_count": int(
            final["central_shape_constraint_count"]
        ),
        "hse_interface_anchor_max": float(final["interface_anchor_max"]),
        "hse_interface_anchor_count": int(final["interface_anchor_count"]),
        "hse_force_exclude_inner_faces": int(force_exclude_boundary_faces),
        "hse_force_exclude_outer_faces": int(
            HYBRID_HSE_FORCE_EXCLUDE_OUTER_FACES
        ),
        "hse_core_face_end": core_face_end,
        "hse_gamma_a1_power": int(gamma_a1_power),
        "hse_core_face_end_fixed": False,
        "hse_core_boundary_radius_rs": float(projected.r[core_face_end]),
        "hse_relaxed_face_count": int(problem.relaxed_face_count),
        "hse_interface_force_included": True,
        "hybrid_exterior_update_policy": exterior_update_policy,
        "hybrid_dynamic_face_start": (
            None if frozen_exterior else core_face_end + 1
        ),
        "hybrid_dynamic_cell_start": None if frozen_exterior else core_face_end,
        "hybrid_boundary_displacement_rs": float(
            projected.r[core_face_end] - state.r[core_face_end]
        ),
        "hybrid_boundary_velocity": float(projected.u[core_face_end]),
        "hybrid_interface_velocity_jump": float(
            projected.u[core_face_end + 1] - projected.u[core_face_end]
        ),
        "hybrid_interface_density_log10_jump": log10_jump(projected.rho),
        "hybrid_interface_pressure_log10_jump": log10_jump(projected.pressure),
        "hybrid_interface_epsilon_log10_jump": log10_jump(projected.epsilon),
        "hybrid_last_hse_width_ratio_to_previous": float(
            boundary_hse_width / previous_hse_width
        ),
        "hybrid_first_exterior_width_ratio_to_last_hse": float(
            first_exterior_width / boundary_hse_width
        ),
        "hybrid_interface_heat_flux": float(projected.q[core_face_end]),
        "hybrid_inner_neighbor_heat_flux": float(projected.q[core_face_end - 1]),
        "hybrid_outer_neighbor_heat_flux": float(projected.q[core_face_end + 1]),
        "hybrid_exterior_radius_exact": exterior_radius_exact,
        "hybrid_exterior_velocity_exact": exterior_velocity_exact,
        "hybrid_single_heat_flux_exact": heat_flux_exact,
        "hse_direct_mass_fixed_point_change": float(
            final["mass_fixed_point_change"]
        ),
        "hse_direct_ephi_fixed_point_change": float(
            final["ephi_fixed_point_change"]
        ),
        "hse_direct_fixed_point_iterations_used": int(
            final["fixed_point_iterations_used"]
        ),
        "hse_direct_fixed_point_iterations_configured": int(
            final["fixed_point_iterations_configured"]
        ),
        "hse_direct_fixed_point_terminated_exactly": bool(
            final["fixed_point_terminated_exactly"]
        ),
        "dt_t0": float(dt_t0),
        **density_shape,
    }
    return projected, metrics
