"""Quasistatic HSE relaxation and force-convergence utilities.

The standalone entry relaxes a low-Mach checkpoint after each heat update.
State-change guards control the timestep. The subdivision runner imports
the force-convergence utilities from this module."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from sidm_engine import (
    TINY,
    PhysicalParams,
    State,
    StepControls,
    avg_cell_to_face,
    avg_face_to_cell,
    baryon_enclosed_mass,
    check_state,
    check_step_transition,
    compute_e_a,
    compute_phi_gradient,
    compute_predicted_heat_flux,
    compute_scales,
    heat_energy_delta,
    horizon_metric,
    implicit_heat_predictor,
    integrate_ephi,
    iterate_mass_gamma,
    total_compactness_metric,
    update_gamma,
    update_dm_misner_sharp_mass,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=Path("grid_outputs"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-t0", type=float, default=580.0)
    parser.add_argument("--dt-t0", type=float, default=1e-4)
    parser.add_argument("--min-dt-t0", type=float, default=1e-12)
    parser.add_argument("--retry-shrink", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--save-interval-t0", type=float, default=0.1)
    parser.add_argument("--hse-max-iter", type=int, default=200)
    parser.add_argument("--hse-rel-tol", type=float, default=1e-6)
    parser.add_argument("--hse-force-max-tol", type=float, default=5e-2)
    parser.add_argument("--hse-force-median-tol", type=float, default=1e-2)
    parser.add_argument("--hse-force-exclude-boundary-faces", type=int, default=2)
    parser.add_argument("--hse-pseudo-time-factor", type=float, default=0.5)
    parser.add_argument("--hse-radius-step-limit", type=float, default=0.05)
    parser.add_argument(
        "--hse-core-radius-rs",
        type=float,
        default=float("inf"),
        help=(
            "Only relax faces initially inside this radius (in Rs units); the first face "
            "outside the radius anchors the core boundary. An infinite radius selects the "
            "whole-halo HSE diagnostic."
        ),
    )
    parser.add_argument("--implicit-heat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--implicit-heat-theta",
        type=float,
        default=1.0,
        help="Theta for implicit heat integration: 1.0 backward Euler, 0.5 Crank-Nicolson.",
    )
    parser.add_argument("--heat-flux-outer-taper-width", type=int, default=80)
    parser.add_argument("--post-step-smoothing-passes", type=int, default=0)
    parser.add_argument("--max-fractional-epsilon-change", type=float, default=0.2)
    parser.add_argument("--max-fractional-density-change", type=float, default=0.2)
    parser.add_argument("--min-shell-width-ratio", type=float, default=0.0)
    parser.add_argument("--max-density-code", type=float, default=1e100)
    parser.add_argument("--max-epsilon-code", type=float, default=1e100)
    parser.add_argument("--max-mass-code", type=float, default=10.0)
    parser.add_argument("--sigma0", type=float, default=5.0)
    parser.add_argument("--rs-kpc", type=float, default=2.6)
    parser.add_argument("--halo-mass-msun", type=float, default=6.3e9)
    parser.add_argument(
        "--baryon-profile",
        choices=[
            "none",
            "plummer",
            "hernquist",
            "powerlaw",
            "feng2021",
            "powerlaw-softened",
            "softened-powerlaw",
            "feng2021-softened",
            "softened-feng2021",
        ],
        default="none",
    )
    parser.add_argument("--baryon-mass-fraction", type=float, default=0.0)
    parser.add_argument("--baryon-scale-radius-rs", type=float, default=0.1)
    parser.add_argument("--baryon-powerlaw-index", type=float, default=0.6)
    parser.add_argument("--horizon-mass-source", choices=["dm", "total"], default="dm")
    return parser.parse_args()


def array_to_state(arr: np.ndarray) -> State:
    return State(
        u=np.ascontiguousarray(arr[0].astype(np.float64)),
        r=np.ascontiguousarray(arr[1].astype(np.float64)),
        rho=np.ascontiguousarray(arr[2].astype(np.float64)),
        epsilon=np.ascontiguousarray(arr[3].astype(np.float64)),
        pressure=np.ascontiguousarray(arr[4].astype(np.float64)),
        enthalpy=np.ascontiguousarray(arr[5].astype(np.float64)),
        ephi=np.ascontiguousarray(arr[6].astype(np.float64)),
        mass=np.ascontiguousarray(arr[7].astype(np.float64)),
        gamma_lorentz=np.ascontiguousarray(arr[8].astype(np.float64)),
        e_a=np.ascontiguousarray(arr[9].astype(np.float64)),
        q=np.ascontiguousarray(arr[10].astype(np.float64)),
    )


def state_to_array(state: State) -> np.ndarray:
    return np.vstack(
        [
            state.u,
            state.r,
            state.rho,
            state.epsilon,
            state.pressure,
            state.enthalpy,
            state.ephi,
            state.mass,
            state.gamma_lorentz,
            state.e_a,
            state.q,
        ]
    )


def load_state(
    grid_dir: Path,
    resume_from: Path,
) -> tuple[np.ndarray, State, float, int, dict[str, object]]:
    a_grid = np.load(grid_dir / "A.npy").astype(np.float64)
    state = array_to_state(np.load(resume_from / "checkpoint_state.npy"))
    meta = json.loads((resume_from / "checkpoint_meta.json").read_text(encoding="utf-8"))
    return a_grid, state, float(meta["tau"]), int(meta.get("accepted_steps", 0)), meta


def full_cell_arrays(
    rho_cell: np.ndarray,
    eps_cell: np.ndarray,
    pressure_cell: np.ndarray,
    enthalpy_cell: np.ndarray,
    template: State,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = template.rho.copy()
    eps = template.epsilon.copy()
    pressure = template.pressure.copy()
    enthalpy = template.enthalpy.copy()
    rho[:-1] = rho_cell
    eps[:-1] = eps_cell
    pressure[:-1] = pressure_cell
    enthalpy[:-1] = enthalpy_cell
    rho[-1] = rho[-2]
    eps[-1] = eps[-2]
    pressure[-1] = pressure[-2]
    enthalpy[-1] = enthalpy[-2]
    return rho, eps, pressure, enthalpy


# The normalized force residual is assembled from three nearly cancelling
# terms. Keep only a floating-point comparison margin; the Newton projection
# must reduce any physically meaningful excess below the configured gate.
FORCE_TOLERANCE_RELATIVE_SLACK = 1e-6
FORCE_TOLERANCE_ABSOLUTE_SLACK = 1e-12


def force_tolerance_slack(tolerance: float) -> float:
    """Return a floating-point comparison margin, not a relaxed HSE target."""
    return max(
        1e-14,
        abs(float(tolerance)) * FORCE_TOLERANCE_RELATIVE_SLACK,
    ) + FORCE_TOLERANCE_ABSOLUTE_SLACK


def force_within_tolerance(value: float, tolerance: float) -> bool:
    return bool(
        np.isfinite(value)
        and float(value) <= float(tolerance) + force_tolerance_slack(tolerance)
    )


def hse_relax_step(
    state: State,
    a_face: np.ndarray,
    dt_t0: float,
    params: PhysicalParams,
    scales,
    controls: StepControls,
    max_iter: int,
    rel_tol: float,
    force_max_tol: float,
    force_median_tol: float,
    force_exclude_boundary_faces: int,
    pseudo_time_factor: float,
    radius_step_limit: float,
    core_radius_rs: float = float("inf"),
    core_face_end_override: int | None = None,
) -> tuple[State, dict[str, float | int | bool]]:
    dt_code = dt_t0 * scales.dt_code_per_t0
    if controls.implicit_heat_enabled:
        q_new, delta_eps_heat = implicit_heat_predictor(state, a_face, dt_code, params, scales, controls)
    else:
        q_new = compute_predicted_heat_flux(state, a_face, params, scales, controls)
        delta_eps_heat = heat_energy_delta(q_new, state, a_face, dt_code, scales)

    # In the quasi-static branch U is reset to zero, so the nonequilibrium
    # pressure factor is inactive.  The lapse equation still follows paper_prd
    # Appendix A2/A9 and must retain the heat-flux time derivative term.
    hse_params = replace(params, pressure_factor_enabled=False)
    n_face = state.r.size
    n_cell = n_face - 1
    if core_face_end_override is not None:
        core_face_end = int(core_face_end_override)
        if not 3 <= core_face_end <= n_face:
            raise ValueError(
                f"core_face_end_override must be in [3, {n_face}], got {core_face_end}"
            )
    elif np.isfinite(core_radius_rs):
        if core_radius_rs <= 0.0:
            raise ValueError("core_radius_rs must be positive")
        outside = np.flatnonzero(state.r > core_radius_rs)
        core_face_end = int(outside[0]) if outside.size else n_face
    else:
        core_face_end = n_face
    # Faces 1..core_face_end-1 move; face core_face_end anchors the interface.
    core_face_end = min(n_face, max(3, core_face_end))
    core_cell_end = min(n_cell, core_face_end)
    relaxed_face_count = max(0, core_face_end - 1)
    if core_face_end < n_face:
        # Preserve the luminosity through the core boundary, while suppressing
        # thermal evolution in the frozen exterior.
        q_new[core_face_end + 1 :] = 0.0
        delta_eps_heat = heat_energy_delta(q_new, state, a_face, dt_code, scales)
        delta_eps_heat[core_cell_end:] = 0.0
    da = np.diff(a_face)
    old_rho = state.rho[:-1]
    old_eps = state.epsilon[:-1]
    old_pressure = state.pressure[:-1]

    r_iter = state.r.copy()
    r_iter[0] = 0.0
    ephi_iter = state.ephi.copy()
    mass_iter = state.mass.copy()
    u_zero = np.zeros_like(state.u)
    max_rel_change = np.inf
    converged = False
    initial_force_residual_max: float | None = None
    initial_force_residual_median: float | None = None
    final_force_residual_max: float | None = None
    final_force_residual_median: float | None = None
    final_accel_maxabs: float | None = None
    final_radius_repairs = 0
    force_exclude = max(0, int(force_exclude_boundary_faces))
    force_max_slack = force_tolerance_slack(force_max_tol)
    force_median_slack = force_tolerance_slack(force_median_tol)
    effective_pseudo_time_factor = float(pseudo_time_factor)
    min_pseudo_time_factor = max(abs(effective_pseudo_time_factor) * 2.0**-24, 1e-14)
    hse_backtracks = 0
    pending_radius_trial = False
    pending_rel_change = 0.0
    pending_radius_repairs = 0
    accepted_objective = np.inf
    accepted_force_residual_max = np.inf
    accepted_force_residual_median = np.inf
    accepted_r = r_iter.copy()
    accepted_ephi = ephi_iter.copy()
    accepted_mass = mass_iter.copy()

    last_rho = old_rho.copy()
    last_eps = old_eps.copy()
    last_pressure = old_pressure.copy()
    last_enthalpy = state.enthalpy[:-1].copy()
    last_gamma = state.gamma_lorentz.copy()

    for iteration in range(1, max_iter + 1):
        gamma_iter = np.ones_like(state.gamma_lorentz)
        valid = r_iter > 0.0
        radicand = np.ones_like(r_iter)
        radicand[valid] = 1.0 - 2.0 * mass_iter[valid] / np.maximum(r_iter[valid] * scales.r0_scale, TINY)
        gamma_iter[valid] = np.sqrt(np.maximum(radicand[valid], 0.0))
        gamma_iter[0] = 1.0
        gamma_cell = avg_face_to_cell(gamma_iter)

        shell_volume = (4.0 * np.pi / 3.0) * (r_iter[1:] ** 3 - r_iter[:-1] ** 3)
        shell_mass = gamma_cell * da
        rho_cell = shell_mass / np.maximum(shell_volume, TINY)

        vol_old = 1.0 / np.maximum(old_rho, TINY)
        vol_new = 1.0 / np.maximum(rho_cell, TINY)
        d_vol = vol_new - vol_old
        denom = 1.0 + 0.5 * (params.gamma - 1.0) * rho_cell * d_vol
        numerator = old_eps + delta_eps_heat - 0.5 * old_pressure * d_vol
        eps_cell = numerator / np.maximum(denom, params.energy_denominator_floor)
        eps_cell = np.maximum(eps_cell, params.epsilon_floor)
        pressure_cell = (params.gamma - 1.0) * rho_cell * eps_cell
        enthalpy_cell = 1.0 + (eps_cell + pressure_cell / np.maximum(rho_cell, TINY)) / scales.r0_scale

        rho_full, eps_full, pressure_full, enthalpy_full = full_cell_arrays(
            rho_cell, eps_cell, pressure_cell, enthalpy_cell, state
        )
        e_a_new = compute_e_a(q_new, r_iter, rho_cell)
        dphi_da = compute_phi_gradient(
            pressure_cell=pressure_cell,
            rho_cell=rho_cell,
            enthalpy_cell=enthalpy_cell,
            ephi_face=ephi_iter,
            e_a_old=state.e_a,
            e_a_new=e_a_new,
            a_face=a_face,
            dt_code=dt_code,
            params=hse_params,
            scales=scales,
        )
        ephi_iter = integrate_ephi(dphi_da, a_face, hse_params, scales)

        temp_state = State(
            u=u_zero.copy(),
            r=r_iter.copy(),
            rho=rho_full,
            epsilon=eps_full,
            pressure=pressure_full,
            enthalpy=enthalpy_full,
            ephi=ephi_iter.copy(),
            mass=mass_iter.copy(),
            gamma_lorentz=gamma_iter.copy(),
            e_a=e_a_new,
            q=q_new.copy(),
        )
        if params.mass_evolution_enabled:
            mass_iter = update_dm_misner_sharp_mass(a_face, temp_state, params, scales)
        else:
            mass_iter = state.mass.copy()

        pressure_face = avg_cell_to_face(pressure_cell)
        rho_face = avg_cell_to_face(rho_cell)
        acc = np.zeros_like(r_iter)
        term1 = (
            -gamma_iter[1:] ** 2
            * dphi_da[1:]
            * 4.0
            * np.pi
            * r_iter[1:] ** 2
            * rho_face[1:]
            / np.maximum(ephi_iter[1:], TINY)
        )
        baryon_mass_iter = baryon_enclosed_mass(r_iter, params)
        gravity_mass_iter = mass_iter + baryon_mass_iter
        term2 = gravity_mass_iter[1:] / np.maximum(r_iter[1:] ** 2, TINY)
        term3 = 4.0 * np.pi * pressure_face[1:] * r_iter[1:] / scales.r0_scale
        residual = np.abs(term1 + term2 + term3) / (
            np.abs(term1) + np.abs(term2) + np.abs(term3) + TINY
        )
        # residual[k] belongs to face k+1. The fixed interface and exterior
        # are not part of the quasistatic force-balance solve.
        residual_eval = residual[:relaxed_face_count]
        if force_exclude > 0 and residual_eval.size > 2 * force_exclude:
            residual_eval = residual_eval[force_exclude:-force_exclude]
        finite_residual = residual_eval[np.isfinite(residual_eval)]
        if finite_residual.size:
            current_force_residual_max = float(np.max(finite_residual))
            current_force_residual_median = float(np.median(finite_residual))
        else:
            current_force_residual_max = np.inf
            current_force_residual_median = np.inf
        if iteration == 1:
            initial_force_residual_max = current_force_residual_max
            initial_force_residual_median = current_force_residual_median
        force_objective = max(
            current_force_residual_max / max(force_max_tol, TINY),
            current_force_residual_median / max(force_median_tol, TINY),
        )

        if pending_radius_trial and (
            not np.isfinite(force_objective)
            or force_objective >= accepted_objective * (1.0 - 1e-12)
        ):
            # Reject an over-shooting root step.  The thermal update and
            # physical dt remain fixed; only the numerical HSE correction is
            # damped before retrying from the last accepted radius profile.
            r_iter = accepted_r.copy()
            ephi_iter = accepted_ephi.copy()
            mass_iter = accepted_mass.copy()
            effective_pseudo_time_factor *= 0.5
            hse_backtracks += 1
            pending_radius_trial = False
            final_force_residual_max = accepted_force_residual_max
            final_force_residual_median = accepted_force_residual_median
            if abs(effective_pseudo_time_factor) < min_pseudo_time_factor:
                break
            continue

        accepted_objective = force_objective
        accepted_force_residual_max = current_force_residual_max
        accepted_force_residual_median = current_force_residual_median
        accepted_r = r_iter.copy()
        accepted_ephi = ephi_iter.copy()
        accepted_mass = mass_iter.copy()
        if pending_radius_trial:
            max_rel_change = pending_rel_change
            final_radius_repairs = pending_radius_repairs
            pending_radius_trial = False
        final_force_residual_max = current_force_residual_max
        final_force_residual_median = current_force_residual_median
        # The thermal update is part of every quasistatic step, including a
        # force-balanced first iterate.  Save it before the early convergence
        # exit so HSE suppresses shell acceleration without suppressing heat
        # conduction itself.
        last_rho = rho_cell
        last_eps = eps_cell
        last_pressure = pressure_cell
        last_enthalpy = enthalpy_cell
        last_gamma = gamma_iter
        force_converged = (
            current_force_residual_max <= force_max_tol + force_max_slack
            and current_force_residual_median <= force_median_tol + force_median_slack
        )
        if force_converged:
            if iteration == 1:
                max_rel_change = 0.0
            converged = True
            break

        acc[1:] = -ephi_iter[1:] * (term1 + term2 + term3) / scales.r0_scale
        finite_acc = acc[np.isfinite(acc)]
        final_accel_maxabs = float(np.max(np.abs(finite_acc))) if finite_acc.size else np.inf

        t_dyn_sq = r_iter[1:] ** 3 / np.maximum(gravity_mass_iter[1:], TINY)
        # ``acc`` is the physical code acceleration and contains e^phi/R0.
        # The HSE iteration is a dimensionless root solve, so remove that
        # prefactor before applying the dynamical-time preconditioner.
        root_scale = scales.r0_scale / np.maximum(ephi_iter[1:], TINY)
        delta_r = np.zeros_like(r_iter)
        delta_r[1:core_face_end] = (
            effective_pseudo_time_factor
            * t_dyn_sq[:relaxed_face_count]
            * root_scale[:relaxed_face_count]
            * acc[1:core_face_end]
        )
        # A fraction of radius is much larger than a cell width in the resolved
        # core and can make neighbouring Lagrangian faces cross.  Limit each
        # pseudo-step by the narrower adjacent shell instead.
        shell_width = np.diff(r_iter[: core_face_end + 1])
        local_width = np.minimum(shell_width[:-1], shell_width[1:])
        max_step = radius_step_limit * np.maximum(local_width, TINY)
        delta_r[1:core_face_end] = np.clip(
            delta_r[1:core_face_end], -max_step, max_step
        )
        r_next = r_iter + delta_r
        r_next[0] = 0.0
        old_width = np.diff(r_iter)
        radius_repairs = 0
        for i in range(1, core_face_end):
            min_width = max(TINY, 1e-10 * max(1.0, abs(r_iter[i - 1])))
            if i - 1 < old_width.size:
                min_width = max(min_width, 1e-6 * max(old_width[i - 1], TINY))
            if r_next[i] <= r_next[i - 1] + min_width:
                r_next[i] = r_next[i - 1] + min_width
                radius_repairs += 1
        if core_face_end < n_face:
            interface_width = max(
                TINY,
                1e-6 * max(old_width[core_face_end - 1], TINY),
            )
            max_inner_radius = r_iter[core_face_end] - interface_width
            if r_next[core_face_end - 1] >= max_inner_radius:
                r_next[core_face_end - 1] = max_inner_radius
                radius_repairs += 1
                for i in range(core_face_end - 2, 0, -1):
                    min_width = max(TINY, 1e-6 * max(old_width[i], TINY))
                    max_radius = r_next[i + 1] - min_width
                    if r_next[i] >= max_radius:
                        r_next[i] = max_radius
                        radius_repairs += 1
        actual_rel = np.abs(r_next[1:core_face_end] - r_iter[1:core_face_end]) / np.maximum(
            r_iter[1:core_face_end], TINY
        )
        pending_rel_change = float(np.max(actual_rel)) if actual_rel.size else 0.0
        pending_radius_repairs = radius_repairs
        pending_radius_trial = True
        r_iter = r_next

        # Force balance is evaluated at the start of the next iteration.  The
        # radius-change tolerance remains diagnostic; it is not an additional
        # physical condition once the requested force residual is satisfied.

    force_physical_converged = bool(
        final_force_residual_max is not None
        and final_force_residual_median is not None
        and final_force_residual_max <= force_max_tol + force_max_slack
        and final_force_residual_median <= force_median_tol + force_median_slack
    )

    rho_full, eps_full, pressure_full, enthalpy_full = full_cell_arrays(
        last_rho, last_eps, last_pressure, last_enthalpy, state
    )
    e_a_final = compute_e_a(q_new, r_iter, last_rho)
    final_state = State(
        u=u_zero,
        r=r_iter,
        rho=rho_full,
        epsilon=eps_full,
        pressure=pressure_full,
        enthalpy=enthalpy_full,
        ephi=ephi_iter,
        mass=mass_iter,
        gamma_lorentz=last_gamma,
        e_a=e_a_final,
        q=q_new,
    )
    final_state = iterate_mass_gamma(final_state, a_face, params, scales)

    metrics = {
        "hse_iterations": int(iteration),
        "hse_converged": bool(converged),
        "hse_max_rel_radius_change": float(max_rel_change),
        "hse_force_residual_initial_max": float(initial_force_residual_max)
        if initial_force_residual_max is not None
        else None,
        "hse_force_residual_initial_median": float(initial_force_residual_median)
        if initial_force_residual_median is not None
        else None,
        "hse_force_residual_final_max": float(final_force_residual_max)
        if final_force_residual_max is not None
        else None,
        "hse_force_residual_final_median": float(final_force_residual_median)
        if final_force_residual_median is not None
        else None,
        "hse_force_residual_improved": (
            bool(final_force_residual_max < initial_force_residual_max)
            if final_force_residual_max is not None and initial_force_residual_max is not None
            else False
        ),
        "hse_force_converged": bool(force_physical_converged),
        "hse_force_max_tol": float(force_max_tol),
        "hse_force_median_tol": float(force_median_tol),
        "hse_force_max_comparison_slack": float(force_max_slack),
        "hse_force_median_comparison_slack": float(force_median_slack),
        "hse_force_exclude_boundary_faces": int(force_exclude),
        "hse_core_radius_rs_requested": float(core_radius_rs),
        "hse_core_face_end_fixed": bool(core_face_end_override is not None),
        "hse_core_face_end": int(core_face_end),
        "hse_core_cell_end": int(core_cell_end),
        "hse_core_boundary_radius_rs": float(state.r[core_face_end])
        if core_face_end < n_face
        else float(state.r[-1]),
        "hse_relaxed_face_count": int(relaxed_face_count),
        "hse_accel_final_maxabs": float(final_accel_maxabs)
        if final_accel_maxabs is not None
        else None,
        "hse_radius_repairs_final": int(final_radius_repairs),
        "hse_backtracks": int(hse_backtracks),
        "hse_effective_pseudo_time_factor": float(effective_pseudo_time_factor),
        "dt_t0": float(dt_t0),
    }
    return final_state, metrics


def state_metrics(
    state: State,
    params: PhysicalParams,
    scales,
    horizon_mass_source: str,
) -> dict[str, float | int]:
    compactness_dm, idx_dm = horizon_metric(state, scales)
    compactness_total, idx_total = total_compactness_metric(state, params, scales)
    if horizon_mass_source == "total":
        compactness, idx = compactness_total, idx_total
    else:
        compactness, idx = compactness_dm, idx_dm
    baryon_mass = baryon_enclosed_mass(state.r, params)
    return {
        "compactness_max": float(compactness),
        "compactness_idx": int(idx),
        "compactness_dm_max": float(compactness_dm),
        "compactness_dm_idx": int(idx_dm),
        "compactness_total_proxy_max": float(compactness_total),
        "compactness_total_proxy_idx": int(idx_total),
        "mass_at_compactness_idx": float(state.mass[idx]),
        "radius_at_compactness_idx": float(state.r[idx]),
        "baryon_mass_at_compactness_idx": float(baryon_mass[idx]),
        "baryon_mass_outer": float(baryon_mass[-1]),
        "rho_max": float(np.max(state.rho)),
        "epsilon_max": float(np.max(state.epsilon)),
        "ephi_min": float(np.min(state.ephi)),
        "ephi_max": float(np.max(state.ephi)),
    }


def write_checkpoint(
    output_dir: Path,
    state: State,
    tau: float,
    accepted_steps: int,
    dt_t0: float,
    core_face_end: int | None = None,
    core_radius_rs: float = float("inf"),
) -> None:
    state_path = output_dir / "checkpoint_state.npy"
    state_temporary = state_path.with_name(f".{state_path.name}.tmp")
    with state_temporary.open("wb") as stream:
        np.save(stream, state_to_array(state))
    state_temporary.replace(state_path)
    meta = {
        "tau": float(tau),
        "accepted_steps": int(accepted_steps),
        "dt_t0": float(dt_t0),
        "mode": "hse_late",
        "hse_core_face_end": int(core_face_end) if core_face_end is not None else None,
        "hse_core_radius_rs_requested": float(core_radius_rs)
        if np.isfinite(core_radius_rs)
        else None,
    }
    meta_path = output_dir / "checkpoint_meta.json"
    meta_temporary = meta_path.with_name(f".{meta_path.name}.tmp")
    meta_temporary.write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    meta_temporary.replace(meta_path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = PhysicalParams(
        rs_kpc=args.rs_kpc,
        halo_mass_msun=args.halo_mass_msun,
        sigma0_cgs=args.sigma0,
        baryon_profile=args.baryon_profile,
        baryon_mass_fraction=args.baryon_mass_fraction,
        baryon_scale_radius_rs=args.baryon_scale_radius_rs,
        baryon_powerlaw_index=args.baryon_powerlaw_index,
    )
    scales = compute_scales(params)
    controls = StepControls(
        implicit_heat_enabled=args.implicit_heat,
        implicit_heat_theta=args.implicit_heat_theta,
        heat_flux_outer_taper_width=max(0, args.heat_flux_outer_taper_width),
        post_step_smoothing_passes=max(0, args.post_step_smoothing_passes),
        max_fractional_epsilon_change=args.max_fractional_epsilon_change,
        max_fractional_density_change=args.max_fractional_density_change,
        min_shell_width_ratio=args.min_shell_width_ratio,
        dt_t0_min=args.min_dt_t0,
        retry_shrink=args.retry_shrink,
        max_retries=args.max_retries,
        max_density_code=args.max_density_code,
        max_epsilon_code=args.max_epsilon_code,
        max_mass_code=args.max_mass_code,
    )
    a_grid, state, tau, accepted_steps, resume_meta = load_state(args.grid_dir, args.resume_from)
    fixed_core_face_end: int | None = None
    saved_core_face_end = resume_meta.get("hse_core_face_end")
    saved_core_radius = resume_meta.get("hse_core_radius_rs_requested")
    if saved_core_face_end is not None:
        if (
            np.isfinite(args.hse_core_radius_rs)
            and saved_core_radius is not None
            and not np.isclose(float(saved_core_radius), args.hse_core_radius_rs, rtol=0.0, atol=0.0)
        ):
            raise ValueError(
                "requested hse_core_radius_rs does not match the resumed fixed core boundary"
            )
        fixed_core_face_end = int(saved_core_face_end)
    elif np.isfinite(args.hse_core_radius_rs):
        outside = np.flatnonzero(state.r > args.hse_core_radius_rs)
        fixed_core_face_end = int(outside[0]) if outside.size else state.r.size
        fixed_core_face_end = min(state.r.size, max(3, fixed_core_face_end))
    start_tau = tau
    start_wall = time.time()
    next_save_tau = tau
    progress_path = args.output_dir / "progress.jsonl"
    status = "running"
    reason = "running"
    last_hse_metrics: dict[str, float | int | bool] = {}

    write_checkpoint(
        args.output_dir,
        state,
        tau,
        accepted_steps,
        args.dt_t0,
        fixed_core_face_end,
        args.hse_core_radius_rs,
    )

    with progress_path.open("a", encoding="utf-8") as progress:
        while tau < args.max_t0:
            remaining_to_target = args.max_t0 - tau
            target_tol_t0 = max(args.min_dt_t0, 1e-12 * max(1.0, abs(args.max_t0)))
            if remaining_to_target <= target_tol_t0:
                tau = args.max_t0
                break
            dt = min(args.dt_t0, remaining_to_target)
            accepted = False
            failure_reason = ""
            trial = state
            hse_metrics: dict[str, float | int | bool] = {}
            for retry in range(max(1, args.max_retries + 1)):
                if dt < args.min_dt_t0:
                    failure_reason = f"dt below min_dt_t0: {dt:.6e}"
                    break
                trial, hse_metrics = hse_relax_step(
                    state,
                    a_grid,
                    dt,
                    params,
                    scales,
                    controls,
                    max_iter=args.hse_max_iter,
                    rel_tol=args.hse_rel_tol,
                    force_max_tol=args.hse_force_max_tol,
                    force_median_tol=args.hse_force_median_tol,
                    force_exclude_boundary_faces=args.hse_force_exclude_boundary_faces,
                    pseudo_time_factor=args.hse_pseudo_time_factor,
                    radius_step_limit=args.hse_radius_step_limit,
                    core_radius_rs=args.hse_core_radius_rs,
                    core_face_end_override=fixed_core_face_end,
                )
                ok, msg = check_state(trial, controls, scales)
                if ok:
                    ok, msg = check_step_transition(state, trial, controls)
                if ok and bool(hse_metrics.get("hse_converged", False)):
                    accepted = True
                    break
                failure_reason = msg
                if ok and not bool(hse_metrics.get("hse_converged", False)):
                    failure_reason = "HSE relaxation did not converge"
                dt *= args.retry_shrink

            if not accepted:
                status = "failed"
                reason = failure_reason
                last_hse_metrics = hse_metrics
                break

            state = trial
            tau += dt
            accepted_steps += 1
            last_hse_metrics = hse_metrics
            metrics = state_metrics(state, params, scales, args.horizon_mass_source)
            compactness = float(metrics["compactness_max"])
            event = {
                "tau": float(tau),
                "accepted_steps": int(accepted_steps),
                "dt_t0": float(dt),
                "status": "accepted",
                "wall_seconds": time.time() - start_wall,
                **metrics,
                **hse_metrics,
            }
            progress.write(json.dumps(event, sort_keys=True) + "\n")
            progress.flush()
            if compactness >= 1.0:
                status = "horizon"
                reason = "horizon"
                break
            if tau >= next_save_tau + args.save_interval_t0 or tau >= args.max_t0:
                write_checkpoint(
                    args.output_dir,
                    state,
                    tau,
                    accepted_steps,
                    dt,
                    fixed_core_face_end,
                    args.hse_core_radius_rs,
                )
                next_save_tau = tau

        if status == "running":
            status = "max_t0_reached"
            reason = "max_t0_reached"

    write_checkpoint(
        args.output_dir,
        state,
        tau,
        accepted_steps,
        args.dt_t0,
        fixed_core_face_end,
        args.hse_core_radius_rs,
    )
    np.save(args.output_dir / "final_state.npy", state_to_array(state))
    summary = {
        "status": status,
        "reason": reason,
        "tau_start": float(start_tau),
        "tau_final": float(tau),
        "dt_t0_requested": float(args.dt_t0),
        "accepted_steps": int(accepted_steps),
        "wall_seconds": time.time() - start_wall,
        "controls": {
            "implicit_heat": bool(args.implicit_heat),
            "implicit_heat_theta": float(args.implicit_heat_theta),
            "heat_flux_outer_taper_width": int(args.heat_flux_outer_taper_width),
            "max_fractional_epsilon_change": float(args.max_fractional_epsilon_change),
            "max_fractional_density_change": float(args.max_fractional_density_change),
            "max_density_code": float(args.max_density_code),
            "max_epsilon_code": float(args.max_epsilon_code),
            "hse_max_iter": int(args.hse_max_iter),
            "hse_rel_tol": float(args.hse_rel_tol),
            "hse_force_max_tol": float(args.hse_force_max_tol),
            "hse_force_median_tol": float(args.hse_force_median_tol),
            "hse_force_exclude_boundary_faces": int(args.hse_force_exclude_boundary_faces),
            "hse_pseudo_time_factor": float(args.hse_pseudo_time_factor),
            "hse_radius_step_limit": float(args.hse_radius_step_limit),
            "hse_core_radius_rs": float(args.hse_core_radius_rs),
            "hse_core_face_end": fixed_core_face_end,
        },
        "last_hse": last_hse_metrics,
        **state_metrics(state, params, scales, args.horizon_mass_source),
        "horizon_mass_source": args.horizon_mass_source,
        "params": params.__dict__,
        "scales": scales.__dict__,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status in {"horizon", "max_t0_reached"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
