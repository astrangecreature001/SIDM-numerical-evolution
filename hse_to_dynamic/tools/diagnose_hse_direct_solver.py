"""Constrained direct solver for the HSE core, with a standalone diagnostic entry.

The HSE evolution module imports this solver. Logarithmic shell widths keep
the core ordered while the signed, normalized force residual is solved."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_hse_late import (  # noqa: E402
    array_to_state,
    force_within_tolerance,
    full_cell_arrays,
    state_to_array,
)
from sidm_engine import (  # noqa: E402
    TINY,
    PhysicalParams,
    State,
    StepControls,
    avg_cell_to_face,
    avg_face_to_cell,
    baryon_enclosed_mass,
    compute_e_a,
    compute_phi_gradient,
    compute_predicted_heat_flux,
    compute_scales,
    heat_energy_delta,
    implicit_heat_predictor,
    integrate_ephi,
    update_dm_misner_sharp_mass,
)


def dataclass_from_mapping(cls, mapping: dict[str, object]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in mapping.items() if key in allowed})


def radii_to_coordinates(r_face: np.ndarray, core_face_end: int) -> np.ndarray:
    widths = np.diff(r_face[: core_face_end + 1])
    if np.any(widths <= 0.0):
        raise ValueError("core shell widths must be positive")
    return np.log(widths[:-1] / widths[-1])


def coordinates_to_radii(
    coordinates: np.ndarray,
    template_r: np.ndarray,
    core_face_end: int,
) -> np.ndarray:
    logits = np.concatenate([coordinates, np.zeros(1, dtype=np.float64)])
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    widths = template_r[core_face_end] * weights / np.sum(weights)
    radii = template_r.copy()
    radii[0] = 0.0
    radii[1:core_face_end] = np.cumsum(widths)[:-1]
    radii[core_face_end] = template_r[core_face_end]
    return radii


def radii_to_free_boundary_coordinates(
    r_face: np.ndarray,
    core_face_end: int,
) -> np.ndarray:
    """Encode every HSE shell width while leaving the outer radius free."""

    widths = np.diff(r_face[: core_face_end + 1])
    if widths.size != core_face_end or np.any(widths <= 0.0):
        raise ValueError("free-boundary HSE shell widths must be positive")
    radius_scale = max(float(r_face[core_face_end]), TINY)
    return np.log(widths / radius_scale)


def free_boundary_coordinates_to_radii(
    coordinates: np.ndarray,
    template_r: np.ndarray,
    core_face_end: int,
) -> np.ndarray:
    """Decode positive HSE widths without fixing the SMFP boundary radius."""

    if coordinates.size != core_face_end:
        raise ValueError("free-boundary coordinate count must equal core_face_end")
    radius_scale = max(float(template_r[core_face_end]), TINY)
    widths = radius_scale * np.exp(coordinates)
    radii = template_r.copy()
    radii[0] = 0.0
    radii[1 : core_face_end + 1] = np.cumsum(widths)
    return radii


class HSEDirectProblem:
    def __init__(
        self,
        state: State,
        a_face: np.ndarray,
        dt_t0: float,
        params: PhysicalParams,
        controls: StepControls,
        core_face_end: int,
        force_exclude: int,
        fixed_point_iterations: int,
        solve_full_residual: bool = False,
        freeze_exterior_thermal: bool = True,
        radius_template: np.ndarray | None = None,
        q_new_override: np.ndarray | None = None,
        delta_eps_heat_override: np.ndarray | None = None,
        exterior_state: State | None = None,
        force_exclude_inner: int | None = None,
        force_exclude_outer: int | None = None,
        stop_on_exact_fixed_point: bool = True,
        close_excluded_residuals: bool = False,
        gamma_a1_power: int = 2,
        free_outer_boundary: bool = False,
    ) -> None:
        self.state = state
        self.a_face = a_face
        self.params = params
        self.hse_params = replace(params, pressure_factor_enabled=False)
        self.controls = controls
        self.scales = compute_scales(params)
        self.dt_code = dt_t0 * self.scales.dt_code_per_t0
        self.core_face_end = int(core_face_end)
        self.core_cell_end = min(state.r.size - 1, self.core_face_end)
        self.free_outer_boundary = bool(free_outer_boundary)
        # With a free outer boundary there is one radius unknown per HSE shell,
        # so the physical interface force is the final hydrostatic equation.
        # The fixed-radius problem retains only the strictly internal
        # faces for compatibility with the standalone diagnostic tool.
        self.relaxed_face_count = (
            self.core_face_end
            if self.free_outer_boundary
            else self.core_face_end - 1
        )
        self.force_exclude = max(0, int(force_exclude))
        self.force_exclude_inner = (
            self.force_exclude
            if force_exclude_inner is None
            else max(0, int(force_exclude_inner))
        )
        self.force_exclude_outer = (
            self.force_exclude
            if force_exclude_outer is None
            else max(0, int(force_exclude_outer))
        )
        self.fixed_point_iterations = max(1, int(fixed_point_iterations))
        self.solve_full_residual = bool(solve_full_residual)
        self.freeze_exterior_thermal = bool(freeze_exterior_thermal)
        self.stop_on_exact_fixed_point = bool(stop_on_exact_fixed_point)
        self.close_excluded_residuals = bool(close_excluded_residuals)
        if int(gamma_a1_power) not in (1, 2):
            raise ValueError("gamma_a1_power must be 1 or 2")
        self.gamma_a1_power = int(gamma_a1_power)
        self.radius_template = (
            state.r.copy()
            if radius_template is None
            else np.ascontiguousarray(radius_template, dtype=np.float64).copy()
        )
        if self.radius_template.shape != state.r.shape:
            raise ValueError("radius_template must match the state radius shape")
        self.exterior_state = exterior_state
        if exterior_state is not None and exterior_state.r.shape != state.r.shape:
            raise ValueError("exterior_state must match the state shape")
        self.da = np.diff(a_face)
        self.old_rho = state.rho[:-1]
        self.old_eps = state.epsilon[:-1]
        self.old_pressure = state.pressure[:-1]
        self.central_shape_reference = self._central_regularity_residuals(
            state.r,
            self.old_rho,
            self.force_exclude_inner,
        )
        if (q_new_override is None) != (delta_eps_heat_override is None):
            raise ValueError("q_new_override and delta_eps_heat_override must be provided together")
        if q_new_override is not None and delta_eps_heat_override is not None:
            q_new = np.ascontiguousarray(q_new_override, dtype=np.float64).copy()
            delta_eps_heat = np.ascontiguousarray(
                delta_eps_heat_override,
                dtype=np.float64,
            ).copy()
            if q_new.shape != state.q.shape or delta_eps_heat.shape != state.epsilon[:-1].shape:
                raise ValueError("thermal overrides have incompatible shapes")
        elif controls.implicit_heat_enabled:
            q_new, delta_eps_heat = implicit_heat_predictor(
                state,
                a_face,
                self.dt_code,
                params,
                self.scales,
                controls,
            )
        else:
            q_new = compute_predicted_heat_flux(state, a_face, params, self.scales, controls)
            delta_eps_heat = heat_energy_delta(
                q_new,
                state,
                a_face,
                self.dt_code,
                self.scales,
            )
        self.q_new = q_new.copy()
        self.delta_eps_heat = delta_eps_heat.copy()
        if self.freeze_exterior_thermal and self.core_face_end < state.r.size:
            self.q_new[self.core_face_end + 1 :] = 0.0
            self.delta_eps_heat = heat_energy_delta(
                self.q_new,
                state,
                a_face,
                self.dt_code,
                self.scales,
            )
            self.delta_eps_heat[self.core_cell_end :] = 0.0
        self.evaluations = 0

    @staticmethod
    def _central_regularity_residuals(
        r_face: np.ndarray,
        rho_cell: np.ndarray,
        count: int,
    ) -> np.ndarray:
        """Measure local central-shape curvature in regular spherical coordinates.

        A smooth spherical scalar has ``log(rho) = log(rho_c) + a r^2 +
        O(r^4)`` near the origin.  Each residual below measures the departure
        of one cell from linear interpolation in ``r^2`` between its two
        neighbours.  Cell radii are volume midpoints so unequal Lagrangian
        shell masses do not masquerade as geometric irregularity.
        """

        usable = min(max(0, int(count)), max(0, rho_cell.size - 2))
        if usable == 0:
            return np.empty(0, dtype=np.float64)
        r_mid = np.cbrt(0.5 * (r_face[:-1] ** 3 + r_face[1:] ** 3))
        x = r_mid**2
        log_rho = np.log(np.maximum(rho_cell, TINY))
        residuals = np.empty(usable, dtype=np.float64)
        for offset in range(usable):
            i = offset + 1
            span = max(float(x[i + 1] - x[i - 1]), TINY)
            weight = float(x[i] - x[i - 1]) / span
            interpolated = (1.0 - weight) * log_rho[i - 1] + weight * log_rho[i + 1]
            residuals[offset] = log_rho[i] - interpolated
        return residuals

    def _outer_interface_residuals(
        self,
        r_face: np.ndarray,
        count: int,
    ) -> np.ndarray:
        """Anchor omitted hybrid transition faces to the dynamic predictor."""

        usable = min(max(0, int(count)), self.relaxed_face_count)
        if usable == 0:
            return np.empty(0, dtype=np.float64)
        first_face = self.relaxed_face_count - usable + 1
        faces = np.arange(first_face, self.relaxed_face_count + 1, dtype=np.int64)
        return np.log(
            np.maximum(r_face[faces], TINY)
            / np.maximum(self.radius_template[faces], TINY)
        )

    def evaluate(self, coordinates: np.ndarray) -> tuple[np.ndarray, State, dict[str, float]]:
        self.evaluations += 1
        if self.free_outer_boundary:
            r_iter = free_boundary_coordinates_to_radii(
                coordinates,
                self.radius_template,
                self.core_face_end,
            )
        else:
            r_iter = coordinates_to_radii(
                coordinates,
                self.radius_template,
                self.core_face_end,
            )
        mass_iter = self.state.mass.copy()
        ephi_iter = self.state.ephi.copy()
        u_zero = np.zeros_like(self.state.u)
        if self.exterior_state is not None:
            u_zero[self.core_face_end :] = self.exterior_state.u[self.core_face_end :]
        mass_change = np.inf
        ephi_change = np.inf
        fixed_point_iterations_used = 0
        fixed_point_terminated_exactly = False

        for _ in range(self.fixed_point_iterations):
            previous_mass = mass_iter.copy()
            previous_ephi = ephi_iter.copy()
            gamma_iter = np.ones_like(self.state.gamma_lorentz)
            valid = r_iter > 0.0
            radicand = np.ones_like(r_iter)
            radicand[valid] = (
                1.0
                + (u_zero[valid] / self.scales.r0_scale) ** 2
                - 2.0
                * mass_iter[valid]
                / np.maximum(r_iter[valid] * self.scales.r0_scale, TINY)
            )
            gamma_iter[valid] = np.sqrt(np.maximum(radicand[valid], 0.0))
            gamma_iter[0] = 1.0
            gamma_cell = avg_face_to_cell(gamma_iter)
            shell_volume = (4.0 * np.pi / 3.0) * (
                r_iter[1:] ** 3 - r_iter[:-1] ** 3
            )
            rho_cell = gamma_cell * self.da / np.maximum(shell_volume, TINY)

            d_vol = 1.0 / np.maximum(rho_cell, TINY) - 1.0 / np.maximum(
                self.old_rho,
                TINY,
            )
            denominator = 1.0 + 0.5 * (self.params.gamma - 1.0) * rho_cell * d_vol
            numerator = (
                self.old_eps
                + self.delta_eps_heat
                - 0.5 * self.old_pressure * d_vol
            )
            eps_cell = numerator / np.maximum(
                denominator,
                self.params.energy_denominator_floor,
            )
            eps_cell = np.maximum(eps_cell, self.params.epsilon_floor)
            pressure_cell = (self.params.gamma - 1.0) * rho_cell * eps_cell
            if self.exterior_state is not None and self.core_cell_end < pressure_cell.size:
                exterior_base = (
                    (self.params.gamma - 1.0)
                    * self.exterior_state.rho[:-1]
                    * self.exterior_state.epsilon[:-1]
                )
                exterior_factor = self.exterior_state.pressure[:-1] / np.maximum(
                    exterior_base,
                    TINY,
                )
                pressure_cell[self.core_cell_end :] *= exterior_factor[
                    self.core_cell_end :
                ]
            enthalpy_cell = 1.0 + (
                eps_cell + pressure_cell / np.maximum(rho_cell, TINY)
            ) / self.scales.r0_scale
            rho, eps, pressure, enthalpy = full_cell_arrays(
                rho_cell,
                eps_cell,
                pressure_cell,
                enthalpy_cell,
                self.state,
            )
            e_a_new = compute_e_a(self.q_new, r_iter, rho_cell)
            dphi_da = compute_phi_gradient(
                pressure_cell=pressure_cell,
                rho_cell=rho_cell,
                enthalpy_cell=enthalpy_cell,
                ephi_face=ephi_iter,
                e_a_old=self.state.e_a,
                e_a_new=e_a_new,
                a_face=self.a_face,
                dt_code=self.dt_code,
                params=self.hse_params,
                scales=self.scales,
            )
            ephi_iter = integrate_ephi(
                dphi_da,
                self.a_face,
                self.hse_params,
                self.scales,
            )
            trial_state = State(
                u=u_zero.copy(),
                r=r_iter.copy(),
                rho=rho,
                epsilon=eps,
                pressure=pressure,
                enthalpy=enthalpy,
                ephi=ephi_iter.copy(),
                mass=mass_iter.copy(),
                gamma_lorentz=gamma_iter.copy(),
                e_a=e_a_new,
                q=self.q_new.copy(),
            )
            if self.params.mass_evolution_enabled:
                mass_iter = update_dm_misner_sharp_mass(
                    self.a_face,
                    trial_state,
                    self.params,
                    self.scales,
                )
            else:
                mass_iter = self.state.mass.copy()
            mass_change = float(
                np.max(
                    np.abs(mass_iter - previous_mass)
                    / np.maximum(np.abs(mass_iter), TINY)
                )
            )
            ephi_change = float(
                np.max(
                    np.abs(ephi_iter - previous_ephi)
                    / np.maximum(np.abs(ephi_iter), TINY)
                )
            )
            fixed_point_iterations_used += 1
            if (
                self.stop_on_exact_fixed_point
                and mass_change == 0.0
                and ephi_change == 0.0
            ):
                fixed_point_terminated_exactly = True
                break

        trial_state.mass = mass_iter
        trial_state.gamma_lorentz = gamma_iter
        pressure_face = avg_cell_to_face(pressure_cell)
        rho_face = avg_cell_to_face(rho_cell)
        term_pressure = (
            -(gamma_iter[1:] ** self.gamma_a1_power)
            * dphi_da[1:]
            * 4.0
            * np.pi
            * r_iter[1:] ** 2
            * rho_face[1:]
            / np.maximum(ephi_iter[1:], TINY)
        )
        gravity_mass = mass_iter + baryon_enclosed_mass(r_iter, self.params)
        term_gravity = gravity_mass[1:] / np.maximum(r_iter[1:] ** 2, TINY)
        term_pressure_gravity = (
            4.0 * np.pi * pressure_face[1:] * r_iter[1:] / self.scales.r0_scale
        )
        # Normalize by the inward support scale, not by the pressure term
        # itself.  The old L1 normalization became identically +1 whenever
        # all three terms had the same sign, giving Newton a zero Jacobian at
        # precisely the pressure-gradient inversions it needed to repair.
        signed = (term_pressure + term_gravity + term_pressure_gravity) / (
            np.abs(term_gravity) + np.abs(term_pressure_gravity) + TINY
        )
        movable = signed[: self.relaxed_face_count]
        gate = movable
        gate_start = 0
        gate_stop = movable.size
        if movable.size > self.force_exclude_inner + self.force_exclude_outer:
            gate_stop = (
                movable.size - self.force_exclude_outer
                if self.force_exclude_outer > 0
                else movable.size
            )
            gate_start = self.force_exclude_inner
            gate = movable[gate_start:gate_stop]
        gate_peak = int(np.argmax(np.abs(gate)))
        central_shape = self._central_regularity_residuals(
            r_iter,
            rho_cell,
            gate_start if self.close_excluded_residuals else 0,
        )
        central_shape_drift = central_shape.copy()
        if central_shape_drift.size:
            central_shape_drift -= self.central_shape_reference[: central_shape_drift.size]
        interface_anchor = self._outer_interface_residuals(
            r_iter,
            (movable.size - gate_stop) if self.close_excluded_residuals else 0,
        )
        closure_residual = np.concatenate([central_shape_drift, interface_anchor])
        closure_max = (
            float(np.max(np.abs(closure_residual)))
            if closure_residual.size
            else 0.0
        )
        metrics = {
            "force_max": float(np.max(np.abs(gate))),
            # signed[0] is physical face 1 because the central face is omitted.
            "force_max_face": int(1 + gate_start + gate_peak),
            "force_gate_first_face": int(1 + gate_start),
            "force_gate_last_face": int(gate_stop),
            "force_median": float(np.median(np.abs(gate))),
            "force_rms": float(np.sqrt(np.mean(gate**2))),
            "full_movable_force_max": float(np.max(np.abs(movable))),
            "interface_force_residual": (
                float(signed[self.core_face_end - 1])
                if self.free_outer_boundary
                else float("nan")
            ),
            "central_shape_drift_max": (
                float(np.max(np.abs(central_shape_drift)))
                if central_shape_drift.size
                else 0.0
            ),
            "central_shape_constraint_count": int(central_shape_drift.size),
            "interface_anchor_max": (
                float(np.max(np.abs(interface_anchor)))
                if interface_anchor.size
                else 0.0
            ),
            "interface_anchor_count": int(interface_anchor.size),
            "closure_max": closure_max,
            "mass_fixed_point_change": mass_change,
            "ephi_fixed_point_change": ephi_change,
            "fixed_point_iterations_used": int(fixed_point_iterations_used),
            "fixed_point_iterations_configured": int(self.fixed_point_iterations),
            "fixed_point_terminated_exactly": bool(fixed_point_terminated_exactly),
        }
        if self.solve_full_residual:
            solver_residual = movable
        elif self.close_excluded_residuals:
            solver_residual = np.concatenate([gate, closure_residual])
        else:
            solver_residual = gate
        return np.ascontiguousarray(solver_residual), trial_state, metrics


def damped_newton(
    problem: HSEDirectProblem,
    coordinates: np.ndarray,
    max_iterations: int,
    finite_difference_step: float,
    trust_radius: float,
    min_line_search: float,
    force_max_tolerance: float = 0.05,
    force_median_tolerance: float = 0.01,
    linear_damping: float = 0.0,
    basis: np.ndarray | None = None,
    minimum_iterations: int = 0,
    closure_tolerance: float | None = None,
) -> tuple[np.ndarray, State, list[dict[str, float | int | bool]]]:
    minimum_iterations = max(0, int(minimum_iterations))
    if minimum_iterations > max_iterations:
        raise ValueError("minimum_iterations cannot exceed max_iterations")
    y = coordinates.copy()
    residual, state, metrics = problem.evaluate(y)
    history: list[dict[str, float | int | bool]] = []

    def closure_ok(values: dict[str, float | int | bool]) -> bool:
        return bool(
            closure_tolerance is None
            or force_within_tolerance(
                float(values.get("closure_max", 0.0)),
                closure_tolerance,
            )
        )

    def objective(values: dict[str, float | int | bool]) -> float:
        components = [
            float(values["force_max"]) / max(force_max_tolerance, TINY),
            float(values["force_median"]) / max(force_median_tolerance, TINY),
        ]
        if closure_tolerance is not None:
            components.append(
                float(values.get("closure_max", 0.0))
                / max(closure_tolerance, TINY)
            )
        return max(components)

    for iteration in range(max_iterations + 1):
        merit = 0.5 * float(np.dot(residual, residual))
        gate_objective = objective(metrics)
        record: dict[str, float | int | bool] = {
            "iteration": iteration,
            "evaluations": problem.evaluations,
            "merit": merit,
            "force_gate_objective": gate_objective,
            **metrics,
        }
        history.append(record)
        if (
            iteration >= minimum_iterations
            and
            force_within_tolerance(metrics["force_max"], force_max_tolerance)
            and force_within_tolerance(
                metrics["force_median"], force_median_tolerance
            )
            and closure_ok(metrics)
        ):
            record["converged"] = True
            return y, state, history
        if iteration == max_iterations:
            break

        direction_count = y.size if basis is None else basis.shape[1]
        jacobian = np.empty((residual.size, direction_count), dtype=np.float64)
        for column in range(direction_count):
            step = finite_difference_step
            perturbed_plus = y.copy()
            perturbed_minus = y.copy()
            if basis is None:
                step *= max(1.0, abs(float(y[column])))
                perturbed_plus[column] += step
                perturbed_minus[column] -= step
            else:
                perturbation = step * basis[:, column]
                perturbed_plus += perturbation
                perturbed_minus -= perturbation
            residual_plus, _, _ = problem.evaluate(perturbed_plus)
            residual_minus, _, _ = problem.evaluate(perturbed_minus)
            jacobian[:, column] = (residual_plus - residual_minus) / (2.0 * step)
        if linear_damping > 0.0:
            normal = jacobian.T @ jacobian
            diagonal_scale = max(float(np.max(np.diag(normal))), TINY)
            normal.flat[:: normal.shape[0] + 1] += linear_damping * diagonal_scale
            coefficients = np.linalg.solve(normal, -(jacobian.T @ residual))
        else:
            coefficients, *_ = np.linalg.lstsq(jacobian, -residual, rcond=1e-10)
        delta = coefficients if basis is None else basis @ coefficients
        delta_max = float(np.max(np.abs(delta))) if delta.size else 0.0
        if not np.all(np.isfinite(delta)) or delta_max == 0.0:
            record["linear_solve_failed"] = True
            if (
                iteration + 1 >= minimum_iterations
                and force_within_tolerance(
                    metrics["force_max"], force_max_tolerance
                )
                and force_within_tolerance(
                    metrics["force_median"], force_median_tolerance
                )
                and closure_ok(metrics)
            ):
                record["converged"] = True
                record["converged_after_stalled_required_iteration"] = True
                return y, state, history
            break
        if delta_max > trust_radius:
            delta *= trust_radius / delta_max

        alpha = 1.0
        accepted = False
        while alpha >= min_line_search:
            trial_y = y + alpha * delta
            trial_residual, trial_state, trial_metrics = problem.evaluate(trial_y)
            trial_merit = 0.5 * float(np.dot(trial_residual, trial_residual))
            trial_gate_objective = objective(trial_metrics)
            trial_force_converged = bool(
                force_within_tolerance(
                    trial_metrics["force_max"], force_max_tolerance
                )
                and force_within_tolerance(
                    trial_metrics["force_median"], force_median_tolerance
                )
                and closure_ok(trial_metrics)
            )
            gate_improved = bool(
                np.isfinite(trial_gate_objective)
                and trial_gate_objective < gate_objective * (1.0 - 1e-12)
            )
            if np.isfinite(trial_merit) and (
                trial_force_converged
                or (trial_merit < merit and gate_improved)
            ):
                y = trial_y
                residual = trial_residual
                state = trial_state
                metrics = trial_metrics
                accepted = True
                record["line_search_alpha"] = alpha
                record["coordinate_step_max"] = float(np.max(np.abs(alpha * delta)))
                record["line_search_force_gate_objective"] = trial_gate_objective
                break
            alpha *= 0.5
        if not accepted:
            record["line_search_failed"] = True
            if (
                iteration + 1 >= minimum_iterations
                and force_within_tolerance(
                    metrics["force_max"], force_max_tolerance
                )
                and force_within_tolerance(
                    metrics["force_median"], force_median_tolerance
                )
                and closure_ok(metrics)
            ):
                record["converged"] = True
                record["converged_after_stalled_required_iteration"] = True
                return y, state, history
            break
    return y, state, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--dt-t0", type=float, default=1e-3)
    parser.add_argument("--core-face-end", type=int, default=None)
    parser.add_argument("--force-exclude", type=int, default=2)
    parser.add_argument(
        "--tridiagonal-backend",
        choices=("thomas", "numba"),
        default="thomas",
    )
    parser.add_argument("--fixed-point-iterations", type=int, default=12)
    parser.add_argument("--solve-full-residual", action="store_true")
    parser.add_argument("--max-newton-iterations", type=int, default=12)
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--trust-radius", type=float, default=0.05)
    parser.add_argument("--min-line-search", type=float, default=2.0**-14)
    parser.add_argument("--linear-damping", type=float, default=0.0)
    parser.add_argument("--reduced-modes", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    meta = json.loads(
        (args.checkpoint_dir / "checkpoint_meta.json").read_text(encoding="utf-8")
    )
    params = dataclass_from_mapping(PhysicalParams, summary["params"])
    controls = dataclass_from_mapping(StepControls, summary["controls"])
    controls = replace(controls, tridiagonal_backend=args.tridiagonal_backend)
    a_face = np.load(args.grid_dir / "A.npy").astype(np.float64)
    state = array_to_state(
        np.load(args.checkpoint_dir / "checkpoint_state.npy").astype(np.float64)
    )
    core_face_end = args.core_face_end or int(meta["late_hse_core_face_end"])
    problem = HSEDirectProblem(
        state,
        a_face,
        args.dt_t0,
        params,
        controls,
        core_face_end,
        args.force_exclude,
        args.fixed_point_iterations,
        args.solve_full_residual,
    )
    initial_coordinates = radii_to_coordinates(state.r, core_face_end)
    basis = None
    if args.reduced_modes > 0:
        mode_count = min(args.reduced_modes, initial_coordinates.size)
        index = np.arange(initial_coordinates.size, dtype=np.float64) + 0.5
        basis = np.empty((initial_coordinates.size, mode_count), dtype=np.float64)
        basis[:, 0] = 1.0
        for mode in range(1, mode_count):
            basis[:, mode] = np.cos(
                np.pi * mode * index / initial_coordinates.size
            )
    coordinates, final_state, history = damped_newton(
        problem,
        initial_coordinates,
        args.max_newton_iterations,
        args.finite_difference_step,
        args.trust_radius,
        args.min_line_search,
        0.05,
        0.01,
        args.linear_damping,
        basis,
    )
    final_residual, final_state, final_metrics = problem.evaluate(coordinates)
    initial_metrics = history[0]
    shell_width = np.diff(final_state.r)
    max_density_change = float(
        np.max(
            np.abs(final_state.rho[:-1] - state.rho[:-1])
            / np.maximum(np.abs(state.rho[:-1]), TINY)
        )
    )
    max_epsilon_change = float(
        np.max(
            np.abs(final_state.epsilon[:-1] - state.epsilon[:-1])
            / np.maximum(np.abs(state.epsilon[:-1]), TINY)
        )
    )
    result = {
        "source_tau": float(meta["tau"]),
        "dt_t0": args.dt_t0,
        "core_face_end": core_face_end,
        "variables": int(initial_coordinates.size),
        "residual_equations": int(final_residual.size),
        "evaluations": problem.evaluations,
        "converged": bool(
            final_metrics["force_max"] <= 0.05
            and final_metrics["force_median"] <= 0.01
        ),
        "initial_force_max": initial_metrics["force_max"],
        "initial_force_median": initial_metrics["force_median"],
        "final_metrics": final_metrics,
        "minimum_shell_width": float(np.min(shell_width)),
        "strictly_ordered": bool(np.all(shell_width > 0.0)),
        "core_boundary_fixed": bool(
            final_state.r[core_face_end] == state.r[core_face_end]
            and np.array_equal(
                final_state.r[core_face_end:],
                state.r[core_face_end:],
            )
        ),
        "max_fractional_density_change": max_density_change,
        "max_fractional_epsilon_change": max_epsilon_change,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "direct_hse_state.npy", state_to_array(final_state))
    (args.output_dir / "direct_hse_diagnostic.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
