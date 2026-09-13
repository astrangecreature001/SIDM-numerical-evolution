"""Evolve an HSE core with a frozen exterior and recursive timestep subdivision.

Accepted child intervals advance the state and are checkpointed. Rejected
intervals are subdivided. When the next required interval falls below the
switch threshold, reconstruct shell velocities from the last two accepted
HSE states, rebuild the metric variables, and begin dynamical continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from display_diagnostics import smfp_range_metrics
from core_dynamic import evolve_core_dynamic_one_step
from hybrid_hse import evolve_adaptive_hybrid_step
from sidm_engine import (
    A1_GAMMA_POWER,
    PhysicalParams,
    Scales,
    State,
    StepControls,
    compute_cfl_dt_t0,
    compute_scales,
    iterate_mass_gamma,
    solve_tridiagonal,
)


SCHEMA = "sidm_hse_core_density_sublevels_dynamic_handoff_v1"


class HSETrialRejected(RuntimeError):
    def __init__(self, reason: str, details: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--gamma-a1-power", type=int, choices=(1, 2), required=True)
    parser.add_argument("--max-t0", type=float, default=float("inf"))
    parser.add_argument("--max-accepted-steps", type=int, default=0)
    parser.add_argument("--output-interval-t0", type=float, default=0.001)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--fixed-parent-step-t0", type=float, default=1.0e-5)
    parser.add_argument("--core-radius-limit-rs", type=float, default=1.0e-12)
    parser.add_argument("--hse-switch-dt-t0", type=float, default=1.0e-7)
    parser.add_argument("--density-half-factor", type=float, default=0.5)
    parser.add_argument("--density-relative-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--hse-regrow-factor", type=float, default=2.0)
    parser.add_argument("--hse-regrow-after-successes", type=int, default=4)
    parser.add_argument("--dt-cap-t0", type=float, default=float("inf"))
    parser.add_argument("--dt-t0-min", type=float, default=1.0e-24)
    parser.add_argument("--dynamic-energy-denominator-min", type=float, default=0.99)
    parser.add_argument("--horizon-compactness-threshold", type=float, default=1.0)
    parser.add_argument("--phi-heat-derivative-floor-t0", type=float, default=1.0e-7)
    parser.add_argument("--cfl-safety", type=float, default=0.8)
    parser.add_argument("--cfl-window-cells", type=int, default=1)
    parser.add_argument("--implicit-heat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tridiagonal-backend", choices=("thomas", "numba"), default="numba")
    parser.add_argument("--momentum-update-mode", choices=("euler", "heun"), default="heun")
    parser.add_argument("--outer-velocity-boundary-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outer-velocity-boundary-width", type=int, default=20)
    parser.add_argument("--outer-velocity-boundary-passes", type=int, default=2)
    parser.add_argument("--sigma0", type=float, default=5.0)
    parser.add_argument("--rs-kpc", type=float, default=2.6)
    parser.add_argument("--halo-mass-msun", type=float, default=6.3e9)
    parser.add_argument("--gamma-eos", type=float, default=5.0 / 3.0)
    parser.add_argument("--pressure-factor-model", choices=("off", "paper-cell", "flow-divergence"), default="flow-divergence")
    parser.add_argument("--pressure-factor-mach-transition", type=float, default=0.3)
    parser.add_argument("--hse-force-exclude-inner-faces", type=int, default=2)
    parser.add_argument("--hse-fixed-point-iterations", type=int, default=12)
    parser.add_argument("--hse-max-newton-iterations", type=int, default=64)
    parser.add_argument("--hse-finite-difference-step", type=float, default=1.0e-6)
    parser.add_argument("--hse-trust-radius", type=float, default=0.05)
    parser.add_argument("--hse-min-line-search", type=float, default=6.103515625e-5)
    parser.add_argument("--hse-force-max-tol", type=float, default=0.01)
    parser.add_argument("--hse-force-median-tol", type=float, default=0.001)
    parser.add_argument("--hse-closure-tol", type=float, default=0.001)
    parser.add_argument("--hse-boundary-max-face-change", type=int, default=1)
    parser.add_argument("--hse-boundary-probe-every-steps", type=int, default=1)
    parser.add_argument("--hse-nonconvergence-action", choices=("retry", "accept-valid"), default="retry")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.output_interval_t0,
        args.fixed_parent_step_t0,
        args.core_radius_limit_rs,
        args.hse_switch_dt_t0,
        args.dt_t0_min,
    )
    if any(value <= 0.0 for value in positive):
        raise ValueError("output, parent-step, radius, switch-step, and minimum-step values must be positive")
    if not math.isclose(args.density_half_factor, 0.5, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("recursive parent completion requires --density-half-factor 0.5")
    if args.density_relative_tolerance < 0.0:
        raise ValueError("density tolerance must be non-negative")
    if args.hse_regrow_factor <= 1.0 or args.hse_regrow_after_successes < 1:
        raise ValueError("HSE regrowth requires a factor > 1 and a positive success count")
    if not 0.0 < args.dynamic_energy_denominator_min < 1.0:
        raise ValueError("dynamic energy denominator threshold must lie in (0, 1)")
    if args.horizon_compactness_threshold <= 0.0:
        raise ValueError("horizon compactness threshold must be positive")
    if args.phi_heat_derivative_floor_t0 < 0.0:
        raise ValueError("phi heat-derivative floor must be non-negative")
    if not 0.0 < args.cfl_safety <= 1.0:
        raise ValueError("CFL safety must lie in (0, 1]")
    if args.progress_every < 1 or args.max_accepted_steps < 0:
        raise ValueError("progress cadence must be positive and step limit non-negative")
    if args.hse_boundary_max_face_change < 0 or args.hse_boundary_probe_every_steps < 1:
        raise ValueError("HSE boundary controls are invalid")


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, array)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def load_checkpoint(source: Path) -> tuple[State, dict[str, Any]]:
    if source.is_dir():
        state_path = source / "checkpoint_state.npy"
        meta_path = source / "checkpoint_meta.json"
    else:
        state_path = source
        if not source.name.endswith("_state.npy"):
            raise ValueError("checkpoint state filename must end in _state.npy")
        meta_path = source.with_name(source.name[:-10] + "_meta.json")
    state = State.from_initial(np.asarray(np.load(state_path), dtype=np.float64))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return state, meta


def assert_basic_state(state: State) -> None:
    if not np.all(np.isfinite(state.as_array())):
        raise FloatingPointError("state contains non-finite values")
    if not np.all(np.diff(state.r) > 0.0):
        raise FloatingPointError("shell radii are not strictly ordered")
    if np.any(state.rho[:-1] <= 0.0) or np.any(state.epsilon[:-1] <= 0.0):
        raise FloatingPointError("density or internal energy is non-positive")


def cell_radius_rs(state: State) -> np.ndarray:
    return np.cbrt(0.5 * (state.r[:-1] ** 3 + state.r[1:] ** 3))


def core_density_monotonicity(
    state: State,
    radius_limit_rs: float,
    tolerance: float,
) -> dict[str, Any]:
    radius = cell_radius_rs(state)
    rho = np.asarray(state.rho[:-1], dtype=np.float64)
    left = rho[:-1]
    right = rho[1:]
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0e-300)
    rise = (right - left) / scale
    eligible = (radius[:-1] < radius_limit_rs) & (radius[1:] < radius_limit_rs)
    nonfinite = ~(np.isfinite(left) & np.isfinite(right) & np.isfinite(rise))
    bad = np.flatnonzero(eligible & (nonfinite | (rise > tolerance)))
    eligible_index = np.flatnonzero(eligible)
    if eligible_index.size:
        finite_rise = np.where(np.isfinite(rise[eligible_index]), rise[eligible_index], np.inf)
        local_argmax = int(np.argmax(finite_rise))
        maximum_pair = int(eligible_index[local_argmax])
        maximum_rise = float(finite_rise[local_argmax])
    else:
        maximum_pair = None
        maximum_rise = None
    return {
        "core_density_monotonic": bool(bad.size == 0),
        "core_density_radius_limit_rs": float(radius_limit_rs),
        "core_density_cell_count": int(np.count_nonzero(radius < radius_limit_rs)),
        "core_density_pair_count": int(eligible_index.size),
        "core_density_nonmonotonic_count": int(bad.size),
        "core_density_nonfinite_pair_count": int(np.count_nonzero(eligible & nonfinite)),
        "core_density_first_nonmonotonic_cell": int(bad[0]) if bad.size else None,
        "core_density_max_relative_rise": maximum_rise,
        "core_density_max_relative_rise_cell": maximum_pair,
        "core_density_relative_tolerance": float(tolerance),
    }


def dynamic_state_guards(
    previous: State,
    candidate: State,
    gamma_eos: float,
    r0_scale: float,
) -> dict[str, Any]:
    rho_old = np.maximum(previous.rho[:-1], 1.0e-300)
    rho_new = np.maximum(candidate.rho[:-1], 1.0e-300)
    d_vol = 1.0 / rho_new - 1.0 / rho_old
    denominator = 1.0 + 0.5 * (gamma_eos - 1.0) * rho_new * d_vol

    raw_gamma_squared = np.ones_like(candidate.r)
    valid = candidate.r > 0.0
    raw_gamma_squared[valid] = (
        1.0
        + (candidate.u[valid] / r0_scale) ** 2
        - 2.0
        * candidate.mass[valid]
        / np.maximum(candidate.r[valid] * r0_scale, 1.0e-300)
    )
    compactness = np.zeros_like(candidate.r)
    compactness[valid] = (
        2.0
        * candidate.mass[valid]
        / np.maximum(candidate.r[valid] * r0_scale, 1.0e-300)
    )
    compactness_face = int(np.argmax(compactness))
    gamma_raw = np.sqrt(np.maximum(raw_gamma_squared, 0.0))
    u_over_c = candidate.u / r0_scale
    theta_plus_sign = u_over_c + gamma_raw
    theta_minus_sign = u_over_c - gamma_raw
    trapped = valid & (theta_plus_sign <= 0.0) & (theta_minus_sign < 0.0)
    trapped_faces = np.flatnonzero(trapped)
    return {
        "energy_denominator_min": float(np.min(denominator)),
        "energy_denominator_min_cell": int(np.argmin(denominator)),
        "raw_gamma_squared_min": float(np.min(raw_gamma_squared)),
        "raw_gamma_squared_min_face": int(np.argmin(raw_gamma_squared)),
        "max_abs_u_over_c": float(np.max(np.abs(u_over_c))),
        "compactness_max": float(compactness[compactness_face]),
        "compactness_face": compactness_face,
        "compactness_radius_rs": float(candidate.r[compactness_face]),
        "compactness_mass_code": float(candidate.mass[compactness_face]),
        "outermost_future_trapped_face": (
            int(trapped_faces[-1]) if trapped_faces.size else None
        ),
        "theta_plus_sign_at_compactness_max": float(
            theta_plus_sign[compactness_face]
        ),
        "theta_minus_sign_at_compactness_max": float(
            theta_minus_sign[compactness_face]
        ),
    }


def save_horizon_event(
    output: Path,
    previous: State,
    crossing: State,
    payload: dict[str, Any],
) -> Path:
    target = output / "horizon_events" / "first_apparent_horizon"
    atomic_save_npy(target / "pre_crossing_state.npy", previous.as_array())
    atomic_save_npy(target / "crossing_state.npy", crossing.as_array())
    atomic_write_json(target / "event.json", payload)
    atomic_save_npy(output / "latest_checkpoint" / "checkpoint_state.npy", crossing.as_array())
    atomic_write_json(output / "latest_checkpoint" / "checkpoint_meta.json", payload)
    return target


def velocity_sha256(state: State) -> str:
    return hashlib.sha256(np.ascontiguousarray(state.u, dtype=np.float64).tobytes()).hexdigest()


def save_formal_checkpoint(output: Path, state: State, payload: dict[str, Any], interval: float) -> None:
    tau = float(payload["tau"])
    index = int(round(tau / interval))
    stem = f"snapshot_{index:09d}_tau_{tau:.10f}"
    atomic_save_npy(output / "snapshots" / f"{stem}_state.npy", state.as_array())
    atomic_write_json(output / "snapshots" / f"{stem}_meta.json", payload)
    atomic_save_npy(output / "latest_checkpoint" / "checkpoint_state.npy", state.as_array())
    atomic_write_json(output / "latest_checkpoint" / "checkpoint_meta.json", payload)


def save_stop_checkpoint(output: Path, state: State, payload: dict[str, Any]) -> Path:
    tau = float(payload["tau"])
    target = output / "stop_checkpoints" / f"stop_tau_{tau:.12f}"
    atomic_save_npy(target / "checkpoint_state.npy", state.as_array())
    atomic_write_json(target / "checkpoint_meta.json", payload)
    atomic_save_npy(output / "latest_checkpoint" / "checkpoint_state.npy", state.as_array())
    atomic_write_json(output / "latest_checkpoint" / "checkpoint_meta.json", payload)
    return target


def save_child_checkpoint(
    output: Path,
    state: State,
    payload: dict[str, Any],
    level: int,
) -> None:
    target = output / "transient_checkpoints" / f"density_half_level_{level:02d}_latest"
    atomic_save_npy(target / "checkpoint_state.npy", state.as_array())
    atomic_write_json(target / "checkpoint_meta.json", payload)


def checkpoint_payload(
    *,
    tau: float,
    mode: str,
    branch_steps: int,
    hse_steps: int,
    dynamic_steps: int,
    density_half_events: int,
    hse_recovery_events: int,
    last_dt_t0: float,
    last_cfl_dt_t0: float,
    last_core_face_end: int,
    density: dict[str, Any],
    smfp: dict[str, Any],
    last_hse: dict[str, Any],
    transition: dict[str, Any] | None,
    dynamic_anchor_tau: float | None,
    dynamic_elapsed_t0: float,
    hse_velocity_previous_tau: float | None,
    hse_velocity_previous_state_path: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "hse" if mode == "hse" else "dynamic_nonhse",
        "evolution_mode": mode,
        "tau": float(tau),
        "branch_accepted_steps": int(branch_steps),
        "hse_steps": int(hse_steps),
        "dynamic_steps": int(dynamic_steps),
        "density_half_events": int(density_half_events),
        "hse_recovery_events": int(hse_recovery_events),
        "last_dt_t0": float(last_dt_t0),
        "last_cfl_dt_t0": float(last_cfl_dt_t0),
        "hse_core_face_end": int(last_core_face_end),
        "gamma_a1_power": int(args.gamma_a1_power),
        "output_interval_t0": float(args.output_interval_t0),
        "hse_parent_step_t0": float(args.fixed_parent_step_t0),
        "hse_switch_dt_t0": float(args.hse_switch_dt_t0),
        "hse_child_checkpoint_policy": "recursive binary completion; newest accepted state replaces the previous state at the same level",
        "dynamic_timestep_policy": "active-core CFL with frozen exterior after permanent HSE shutdown",
        "phi_heat_derivative_floor_t0": float(args.phi_heat_derivative_floor_t0),
        "dynamic_anchor_tau": dynamic_anchor_tau,
        "post_hse_dynamic_elapsed_t0": float(dynamic_elapsed_t0),
        "hse_to_dynamic_u_policy": "U reconstructed from the last two consecutive accepted HSE states using the exact discrete radius-update relation",
        "hse_to_dynamic_transition": transition,
        "hse_velocity_previous_tau": hse_velocity_previous_tau,
        "hse_velocity_previous_state_path": hse_velocity_previous_state_path,
        **density,
        **smfp,
        **last_hse,
    }


def run() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a_face = np.asarray(np.load(args.grid_dir / "A.npy"), dtype=np.float64)
    if not np.all(np.isfinite(a_face)) or not np.all(np.diff(a_face) > 0.0):
        raise ValueError("A grid must be finite and strictly increasing")
    state, resume_meta = load_checkpoint(args.resume_from)
    if state.r.shape != a_face.shape:
        raise ValueError("checkpoint and A grid have different shapes")
    assert_basic_state(state)
    if int(args.gamma_a1_power) != int(A1_GAMMA_POWER):
        raise ValueError(
            f"requested Gamma power {args.gamma_a1_power} does not match isolated engine power {A1_GAMMA_POWER}"
        )

    params = PhysicalParams(
        rs_kpc=args.rs_kpc,
        halo_mass_msun=args.halo_mass_msun,
        sigma0_cgs=args.sigma0,
        gamma=args.gamma_eos,
        pressure_factor_enabled=args.pressure_factor_model != "off",
        pressure_factor_model=args.pressure_factor_model,
        pressure_factor_mach_transition=args.pressure_factor_mach_transition,
        baryon_profile="none",
        baryon_mass_fraction=0.0,
        baryon_scale_radius_rs=0.0,
        phi_heat_derivative_floor_t0=args.phi_heat_derivative_floor_t0,
    )
    scales: Scales = compute_scales(params)
    controls = StepControls(
        cfl_safety=args.cfl_safety,
        cfl_window_cells=args.cfl_window_cells,
        implicit_heat_enabled=args.implicit_heat,
        tridiagonal_backend=args.tridiagonal_backend,
        dt_t0_min=args.dt_t0_min,
        retry_shrink=args.density_half_factor,
        max_retries=0,
        max_fractional_epsilon_change=0.0,
        max_fractional_density_change=0.0,
        max_abs_u_code=float("inf"),
        max_density_code=float("inf"),
        max_epsilon_code=float("inf"),
        max_mass_code=float("inf"),
        min_shell_width_ratio=0.0,
        momentum_update_mode=args.momentum_update_mode,
        outer_velocity_boundary_enabled=args.outer_velocity_boundary_enabled,
        outer_velocity_boundary_width=max(1, args.outer_velocity_boundary_width),
        outer_velocity_boundary_passes=max(0, args.outer_velocity_boundary_passes),
    )
    if args.tridiagonal_backend == "numba":
        solve_tridiagonal(np.empty(0), np.ones(1), np.empty(0), np.ones(1), backend="numba")

    tau = float(resume_meta["tau"])
    mode = str(resume_meta.get("evolution_mode", "hse"))
    if mode not in {"hse", "dynamic"}:
        mode = "hse"
    branch_steps = int(resume_meta.get("branch_accepted_steps", 0))
    hse_steps = int(resume_meta.get("hse_steps", 0))
    dynamic_steps = int(resume_meta.get("dynamic_steps", 0))
    density_half_events = int(resume_meta.get("density_half_events_branch", 0))
    hse_recovery_events = int(resume_meta.get("hse_recovery_events", 0))
    last_dt = float(resume_meta.get("last_dt_t0", 0.0))
    last_cfl = float(resume_meta.get("last_cfl_dt_t0", 0.0))
    last_core_face_end = int(resume_meta.get("hse_core_face_end", 3))
    last_hse: dict[str, Any] = {}
    adaptive_hse_parent_dt = float(
        resume_meta.get("adaptive_hse_parent_dt_t0", args.fixed_parent_step_t0)
    )
    adaptive_hse_success_streak = int(
        resume_meta.get("adaptive_hse_success_streak", 0)
    )
    adaptive_hse_parent_dt = min(
        max(adaptive_hse_parent_dt, args.hse_switch_dt_t0),
        args.fixed_parent_step_t0,
    )
    hse_parent_had_split = False
    transition: dict[str, Any] | None = resume_meta.get("hse_to_dynamic_transition")
    dynamic_anchor_tau = resume_meta.get("dynamic_anchor_tau")
    dynamic_anchor_tau = None if dynamic_anchor_tau is None else float(dynamic_anchor_tau)
    dynamic_elapsed_t0 = float(resume_meta.get("post_hse_dynamic_elapsed_t0", 0.0))
    dynamic_next_output_elapsed_t0: float | None = None
    hse_velocity_previous_state: State | None = None
    hse_velocity_previous_tau: float | None = None
    hse_velocity_previous_state_path: str | None = resume_meta.get(
        "hse_velocity_previous_state_path"
    )
    if hse_velocity_previous_state_path:
        previous_path = Path(hse_velocity_previous_state_path)
        if previous_path.exists():
            hse_velocity_previous_state = State.from_initial(
                np.asarray(np.load(previous_path), dtype=np.float64)
            )
            hse_velocity_previous_tau = float(
                resume_meta["hse_velocity_previous_tau"]
            )
        else:
            hse_velocity_previous_state_path = None
    if mode == "dynamic":
        if dynamic_anchor_tau is None:
            raise ValueError("dynamic checkpoint is missing dynamic_anchor_tau")
        logical_tau = float(dynamic_anchor_tau + dynamic_elapsed_t0)
        next_index = int(
            math.floor(logical_tau / args.output_interval_t0 + 1.0e-9)
        ) + 1
        dynamic_next_output_elapsed_t0 = float(
            next_index * args.output_interval_t0 - dynamic_anchor_tau
        )
    history = args.output_dir / "history.jsonl"
    latest_density = core_density_monotonicity(
        state, args.core_radius_limit_rs, args.density_relative_tolerance
    )
    latest_smfp = smfp_range_metrics(state, a_face, params, scales)
    if last_core_face_end < 3:
        last_core_face_end = int(latest_smfp["smfp_face_end"])

    config = {
        "schema": SCHEMA,
        "program": "run_hse_sublevel_then_dynamic.py",
        "grid_dir": str(args.grid_dir),
        "output_dir": str(args.output_dir),
        "resume_from": str(args.resume_from),
        "source_resume_tau": float(tau),
        "gamma_a1_power": int(args.gamma_a1_power),
        "no_baryon": True,
        "initial_mode": mode,
        "hse_exterior_update_policy": "frozen",
        "hse_parent_step_t0": float(args.fixed_parent_step_t0),
        "hse_density_domain": f"cell-centred adjacent pairs with both R/Rs < {args.core_radius_limit_rs:.17g}",
        "hse_density_relative_tolerance": float(args.density_relative_tolerance),
        "dynamic_energy_denominator_min": float(
            args.dynamic_energy_denominator_min
        ),
        "horizon_compactness_threshold": float(
            args.horizon_compactness_threshold
        ),
        "hse_rejection_policy": "recursively split both children and complete the original parent interval",
        "hse_regrowth_policy": (
            f"after {args.hse_regrow_after_successes} consecutive unsplit HSE intervals, "
            f"multiply the adaptive parent step by {args.hse_regrow_factor:.17g} up to the requested parent step"
        ),
        "hse_switch_condition": f"permanent dynamic handoff before attempting an HSE child below {args.hse_switch_dt_t0:.17g} t0",
        "child_checkpoint_retention": "one rolling state and metadata file per recursion level",
        "dynamic_policy_after_handoff": "active-core CFL dynamics with the exterior reservoir frozen; no HSE projection",
        "u_handoff_policy": "reconstruct every shell U from the last two consecutive accepted HSE states; frozen exterior shells have zero displacement and zero U",
        "output_interval_t0": float(args.output_interval_t0),
        "cfl_safety": float(args.cfl_safety),
        "hse_boundary_policy": "current contiguous central lambda/H <= 1 region, limited by configured face change",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(args.output_dir / "run_config.json", config)
    atomic_save_npy(args.output_dir / "A.npy", a_face)

    stop: dict[str, str | None] = {"signal": None}

    def request_stop(signum: int, _frame: Any) -> None:
        stop["signal"] = signal.Signals(signum).name

    for stop_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(stop_signal, request_stop)

    started_branch_steps = branch_steps
    failure: dict[str, Any] | None = None
    horizon_event: dict[str, Any] | None = None

    def step_limit_reached() -> bool:
        return bool(
            args.max_accepted_steps > 0
            and branch_steps - started_branch_steps >= args.max_accepted_steps
        )

    def choose_hse_boundary(current: State) -> tuple[int, int, dict[str, Any]]:
        metric = smfp_range_metrics(current, a_face, params, scales)
        raw = int(metric["smfp_face_end"])
        if raw < 3:
            raise HSETrialRejected("unresolved_smfp_core", {"raw_core_face_end": raw})
        probe_due = bool(
            raw == last_core_face_end
            or args.hse_boundary_probe_every_steps == 1
            or (hse_steps > 0 and hse_steps % args.hse_boundary_probe_every_steps == 0)
        )
        chosen = raw if probe_due else last_core_face_end
        if args.hse_boundary_max_face_change > 0 and probe_due:
            delta = int(args.hse_boundary_max_face_change)
            chosen = min(max(raw, max(3, last_core_face_end - delta)), last_core_face_end + delta)
        return raw, int(chosen), metric

    def one_hse_trial(current: State, step_dt: float) -> tuple[State, dict[str, Any], dict[str, Any], float, int]:
        raw, chosen, smfp_before = choose_hse_boundary(current)
        cfl_cell_start = chosen
        exterior_cfl = compute_cfl_dt_t0(
            current, a_face, params, scales, controls, cell_start=cfl_cell_start
        )

        def project(face_end: int) -> tuple[State, dict[str, Any]]:
            candidate, metrics = evolve_adaptive_hybrid_step(
                current,
                a_face,
                step_dt,
                params,
                scales,
                controls,
                face_end,
                args.hse_force_exclude_inner_faces,
                args.hse_fixed_point_iterations,
                args.hse_max_newton_iterations,
                args.hse_finite_difference_step,
                args.hse_trust_radius,
                args.hse_min_line_search,
                args.hse_force_max_tol,
                args.hse_force_median_tol,
                closure_tol=args.hse_closure_tol,
                central_density_shape_cells=8,
                central_density_inversion_tol=1.0e300,
                gamma_a1_power=args.gamma_a1_power,
                exterior_update_policy="frozen",
            )
            assert_basic_state(candidate)
            if not bool(metrics.get("hse_converged", False)) and args.hse_nonconvergence_action == "retry":
                raise HSETrialRejected("hse_nonconverged", metrics)
            return candidate, metrics

        fallback_used = False
        try:
            candidate, metrics = project(chosen)
        except Exception as first_error:
            if chosen == last_core_face_end:
                if isinstance(first_error, HSETrialRejected):
                    raise
                raise HSETrialRejected(
                    f"{type(first_error).__name__}: {first_error}",
                    {"raw_core_face_end": raw, "attempt_core_face_end": chosen},
                ) from first_error
            fallback_used = True
            try:
                chosen = int(last_core_face_end)
                candidate, metrics = project(chosen)
            except Exception as fallback_error:
                if isinstance(fallback_error, HSETrialRejected):
                    raise
                raise HSETrialRejected(
                    f"{type(fallback_error).__name__}: {fallback_error}",
                    {
                        "raw_core_face_end": raw,
                        "first_error": f"{type(first_error).__name__}: {first_error}",
                        "fallback_core_face_end": chosen,
                    },
                ) from fallback_error

        density = core_density_monotonicity(
            candidate, args.core_radius_limit_rs, args.density_relative_tolerance
        )
        metrics.update(
            {
                "hse_raw_core_face_end": int(raw),
                "hse_limited_core_face_end": int(chosen),
                "hse_boundary_was_limited": bool(chosen != raw),
                "hse_boundary_fallback_used": bool(fallback_used),
                "hse_exterior_cfl_cell_start": int(cfl_cell_start),
                "hse_exterior_cfl_dt_t0": float(exterior_cfl),
            }
        )
        if not density["core_density_monotonic"]:
            raise HSETrialRejected(
                "core_density_nonmonotonic",
                {"density": density, "hse": metrics},
            )
        return candidate, metrics, density, float(exterior_cfl), int(chosen)

    def switch_to_dynamic(reason: dict[str, Any]) -> None:
        nonlocal mode, state, transition, dynamic_anchor_tau
        nonlocal dynamic_elapsed_t0, dynamic_next_output_elapsed_t0
        if hse_velocity_previous_state is None or hse_velocity_previous_tau is None:
            raise RuntimeError(
                "HSE-to-dynamic handoff requires the preceding consecutive accepted HSE state"
            )
        accepted_hse_step_dt = float(tau - hse_velocity_previous_tau)
        if not np.isfinite(accepted_hse_step_dt) or accepted_hse_step_dt <= 0.0:
            raise RuntimeError("last accepted HSE step interval is not finite and positive")
        last_hse_delta_r = state.r - hse_velocity_previous_state.r
        reconstructed_u = (
            last_hse_delta_r
            * scales.r0_scale**2
            / (
                np.maximum(state.ephi, 1.0e-300)
                * accepted_hse_step_dt
                * scales.dt_code_per_t0
            )
        )
        target_u = np.zeros_like(state.u)
        target_u[1:] = reconstructed_u[1:]
        if not np.all(np.isfinite(target_u)):
            raise RuntimeError("last-step HSE velocity reconstruction produced non-finite U")
        if np.max(np.abs(target_u / scales.r0_scale)) >= 1.0:
            raise RuntimeError("last-step HSE velocity reconstruction is not subluminal")

        last_hse_state_array = state.as_array()
        dynamic_initial = state.copy()
        dynamic_initial.u[:] = target_u
        dynamic_initial = iterate_mass_gamma(dynamic_initial, a_face, params, scales)
        identical = bool(np.array_equal(dynamic_initial.u, target_u))
        max_abs_difference = float(np.max(np.abs(dynamic_initial.u - target_u)))
        target_u_hash = hashlib.sha256(
            np.ascontiguousarray(target_u, dtype=np.float64).tobytes()
        ).hexdigest()
        dynamic_u_hash = velocity_sha256(dynamic_initial)
        if not identical or dynamic_u_hash != target_u_hash:
            raise RuntimeError("HSE-to-dynamic handoff differs from reconstructed last-step U")
        state = dynamic_initial
        mode = "dynamic"
        dynamic_anchor_tau = float(tau)
        dynamic_elapsed_t0 = 0.0
        next_index = int(math.floor(tau / args.output_interval_t0 + 1.0e-9)) + 1
        dynamic_next_output_elapsed_t0 = float(
            next_index * args.output_interval_t0 - tau
        )
        transition = {
            "event": "hse_to_dynamic_handoff",
            "tau": float(tau),
            "reason": reason,
            "switch_threshold_dt_t0": float(args.hse_switch_dt_t0),
            "last_hse_velocity_previous_tau_t0": float(hse_velocity_previous_tau),
            "last_hse_velocity_step_dt_t0": accepted_hse_step_dt,
            "velocity_definition": "U=(R_n-R_nminus1)*r0_scale^2/(exp(phi_n)*Delta_tau*dt_code_per_t0)",
            "active_face_end": int(last_core_face_end),
            "all_shell_velocity_policy": "apply the same last-step displacement formula to every face; the frozen HSE exterior evaluates exactly to zero",
            "reconstructed_u_sha256": target_u_hash,
            "dynamic_initial_u_sha256": dynamic_u_hash,
            "u_matches_reconstruction_bitwise": identical,
            "u_max_abs_difference": max_abs_difference,
            "max_abs_u_over_c": float(np.max(np.abs(target_u / scales.r0_scale))),
            "inward_face_count": int(np.count_nonzero(target_u < 0.0)),
            "outward_face_count": int(np.count_nonzero(target_u > 0.0)),
            "zero_face_count": int(np.count_nonzero(target_u == 0.0)),
            "dynamic_clock_reset": True,
            "dynamic_anchor_tau": dynamic_anchor_tau,
            "transitioned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        target = args.output_dir / "transition_checkpoints" / f"hse_to_dynamic_tau_{tau:.12f}"
        atomic_save_npy(
            target / "previous_accepted_hse_state.npy",
            hse_velocity_previous_state.as_array(),
        )
        atomic_save_npy(target / "last_accepted_hse_state.npy", last_hse_state_array)
        atomic_save_npy(target / "checkpoint_state.npy", state.as_array())
        atomic_write_json(target / "checkpoint_meta.json", transition)
        append_jsonl(history, transition)

    def advance_hse_interval(
        current: State,
        start_tau: float,
        target_tau: float,
        level: int,
    ) -> tuple[State, float, bool]:
        nonlocal branch_steps, hse_steps, density_half_events, hse_recovery_events
        nonlocal last_dt, last_cfl, last_core_face_end, last_hse, latest_density, latest_smfp
        nonlocal hse_velocity_previous_state, hse_velocity_previous_tau
        nonlocal hse_velocity_previous_state_path
        nonlocal adaptive_hse_parent_dt, adaptive_hse_success_streak
        nonlocal hse_parent_had_split
        if stop["signal"] is not None or step_limit_reached():
            return current, start_tau, False
        step_dt = float(target_tau - start_tau)
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise FloatingPointError(f"invalid recursive HSE interval {step_dt}")
        try:
            candidate, metrics, density, exterior_cfl, chosen = one_hse_trial(current, step_dt)
        except HSETrialRejected as rejected:
            next_dt = step_dt * args.density_half_factor
            hse_parent_had_split = True
            adaptive_hse_success_streak = 0
            adaptive_hse_parent_dt = min(adaptive_hse_parent_dt, next_dt)
            event = {
                "event": "hse_child_split",
                "tau_before": float(start_tau),
                "target_tau": float(target_tau),
                "level": int(level),
                "rejected_dt_t0": float(step_dt),
                "next_child_dt_t0": float(next_dt),
                "reason": rejected.reason,
                "details": rejected.details,
            }
            if rejected.reason == "core_density_nonmonotonic":
                density_half_events += 1
            else:
                hse_recovery_events += 1
            append_jsonl(history, event)
            atomic_write_json(args.output_dir / "transient_checkpoints" / "latest_rejection.json", event)
            if next_dt < args.hse_switch_dt_t0:
                return current, start_tau, True
            midpoint = float(start_tau + next_dt)
            first_state, first_tau, switch = advance_hse_interval(
                current, start_tau, midpoint, level + 1
            )
            if switch or stop["signal"] is not None or step_limit_reached():
                return first_state, first_tau, switch
            return advance_hse_interval(first_state, first_tau, target_tau, level + 1)

        accepted_tau = float(target_tau)
        hse_velocity_previous_state = current.copy()
        hse_velocity_previous_tau = float(start_tau)
        velocity_pair_dir = args.output_dir / "hse_velocity_pair"
        previous_pair_path = velocity_pair_dir / "latest_previous_state.npy"
        current_pair_path = velocity_pair_dir / "latest_current_state.npy"
        atomic_save_npy(previous_pair_path, current.as_array())
        atomic_save_npy(current_pair_path, candidate.as_array())
        hse_velocity_previous_state_path = str(previous_pair_path)
        atomic_write_json(
            velocity_pair_dir / "latest_pair_meta.json",
            {
                "schema": "sidm_hse_consecutive_accepted_pair_v1",
                "previous_tau_t0": float(start_tau),
                "current_tau_t0": float(accepted_tau),
                "accepted_hse_step_dt_t0": float(step_dt),
                "previous_state_path": str(previous_pair_path),
                "current_state_path": str(current_pair_path),
            },
        )
        branch_steps += 1
        hse_steps += 1
        last_dt = float(step_dt)
        last_cfl = float(exterior_cfl)
        last_core_face_end = int(chosen)
        last_hse = metrics
        latest_density = density
        latest_smfp = smfp_range_metrics(candidate, a_face, params, scales)
        if level > 0:
            child_payload = checkpoint_payload(
                tau=accepted_tau,
                mode="hse",
                branch_steps=branch_steps,
                hse_steps=hse_steps,
                dynamic_steps=dynamic_steps,
                density_half_events=density_half_events,
                hse_recovery_events=hse_recovery_events,
                last_dt_t0=last_dt,
                last_cfl_dt_t0=last_cfl,
                last_core_face_end=last_core_face_end,
                density=latest_density,
                smfp=latest_smfp,
                last_hse=last_hse,
                transition=transition,
                dynamic_anchor_tau=dynamic_anchor_tau,
                dynamic_elapsed_t0=dynamic_elapsed_t0,
                hse_velocity_previous_tau=hse_velocity_previous_tau,
                hse_velocity_previous_state_path=hse_velocity_previous_state_path,
                args=args,
            )
            child_payload.update(
                {
                    "event": "accepted_hse_child_checkpoint",
                    "sublevel": int(level),
                    "child_start_tau": float(start_tau),
                    "child_target_tau": float(target_tau),
                    "adaptive_hse_parent_dt_t0": float(adaptive_hse_parent_dt),
                    "adaptive_hse_success_streak": int(adaptive_hse_success_streak),
                }
            )
            save_child_checkpoint(args.output_dir, candidate, child_payload, level)
        return candidate, accepted_tau, False

    def make_payload() -> dict[str, Any]:
        payload = checkpoint_payload(
            tau=tau,
            mode=mode,
            branch_steps=branch_steps,
            hse_steps=hse_steps,
            dynamic_steps=dynamic_steps,
            density_half_events=density_half_events,
            hse_recovery_events=hse_recovery_events,
            last_dt_t0=last_dt,
            last_cfl_dt_t0=last_cfl,
            last_core_face_end=last_core_face_end,
            density=latest_density,
            smfp=latest_smfp,
            last_hse=last_hse,
            transition=transition,
            dynamic_anchor_tau=dynamic_anchor_tau,
            dynamic_elapsed_t0=dynamic_elapsed_t0,
            hse_velocity_previous_tau=hse_velocity_previous_tau,
            hse_velocity_previous_state_path=hse_velocity_previous_state_path,
            args=args,
        )
        payload.update(
            {
                "adaptive_hse_parent_dt_t0": float(adaptive_hse_parent_dt),
                "adaptive_hse_success_streak": int(adaptive_hse_success_streak),
                "hse_regrow_factor": float(args.hse_regrow_factor),
                "hse_regrow_after_successes": int(args.hse_regrow_after_successes),
            }
        )
        return payload

    if math.isclose(
        tau / args.output_interval_t0,
        round(tau / args.output_interval_t0),
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        seed_payload = make_payload()
        seed_payload["event"] = "seed_checkpoint"
        save_formal_checkpoint(args.output_dir, state, seed_payload, args.output_interval_t0)
        append_jsonl(history, seed_payload)

    exit_code = 0
    try:
        while (
            tau < args.max_t0
            and stop["signal"] is None
            and not step_limit_reached()
            and horizon_event is None
        ):
            if mode == "hse":
                next_index = int(math.floor(tau / args.output_interval_t0 + 1.0e-9)) + 1
                checkpoint_tau = float(next_index * args.output_interval_t0)
                parent_dt = min(
                    adaptive_hse_parent_dt,
                    checkpoint_tau - tau,
                    args.max_t0 - tau,
                    args.dt_cap_t0,
                )
                if not np.isfinite(parent_dt) or parent_dt <= 0.0:
                    raise FloatingPointError(f"invalid HSE parent timestep {parent_dt}")
                parent_start = float(tau)
                target = float(tau + parent_dt)
                hse_parent_had_split = False
                state, tau, switch_requested = advance_hse_interval(
                    state, parent_start, target, 0
                )
                if switch_requested:
                    rejection = json.loads(
                        (args.output_dir / "transient_checkpoints" / "latest_rejection.json").read_text(encoding="utf-8")
                    )
                    switch_to_dynamic(
                        {
                            "trigger": "next_required_hse_child_below_threshold",
                            "latest_rejection": rejection,
                        }
                    )
                    continue
                if tau == parent_start:
                    break
                if not hse_parent_had_split:
                    adaptive_hse_success_streak += 1
                    if (
                        adaptive_hse_success_streak
                        >= args.hse_regrow_after_successes
                    ):
                        adaptive_hse_parent_dt = min(
                            args.fixed_parent_step_t0,
                            adaptive_hse_parent_dt * args.hse_regrow_factor,
                        )
                        adaptive_hse_success_streak = 0
            else:
                if dynamic_anchor_tau is None or dynamic_next_output_elapsed_t0 is None:
                    raise RuntimeError("dynamic continuation clock was not initialized")
                full_cfl = compute_cfl_dt_t0(
                    state,
                    a_face,
                    params,
                    scales,
                    controls,
                    cell_stop=last_core_face_end,
                )
                remaining_to_output = float(
                    dynamic_next_output_elapsed_t0 - dynamic_elapsed_t0
                )
                remaining_to_max = float(
                    args.max_t0 - dynamic_anchor_tau - dynamic_elapsed_t0
                )
                dynamic_dt = min(
                    full_cfl,
                    remaining_to_output,
                    remaining_to_max,
                    args.dt_cap_t0,
                )
                if not np.isfinite(dynamic_dt) or dynamic_dt < args.dt_t0_min:
                    raise FloatingPointError(f"invalid dynamic CFL timestep {dynamic_dt}")
                dynamic_level = 0
                density_limiter_requested = False
                while True:
                    rejection_reason: str | None = None
                    rejection_details: dict[str, Any] = {}
                    try:
                        trial = evolve_core_dynamic_one_step(
                            state,
                            a_face,
                            dynamic_dt,
                            params,
                            scales,
                            controls,
                            core_face_end=last_core_face_end,
                            monotone_density_radius_limit_rs=(
                                args.core_radius_limit_rs
                                if density_limiter_requested
                                else None
                            ),
                        )
                        assert_basic_state(trial)
                        trial_density = core_density_monotonicity(
                            trial,
                            args.core_radius_limit_rs,
                            args.density_relative_tolerance,
                        )
                        trial_guards = dynamic_state_guards(
                            state,
                            trial,
                            params.gamma,
                            scales.r0_scale,
                        )
                        if not trial_density["core_density_monotonic"]:
                            rejection_reason = "core_density_nonmonotonic"
                            rejection_details = {"density": trial_density}
                        elif (
                            trial_guards["energy_denominator_min"]
                            <= args.dynamic_energy_denominator_min
                        ):
                            rejection_reason = "dynamic_energy_denominator_below_threshold"
                            rejection_details = {"guards": trial_guards}
                        elif trial_guards["raw_gamma_squared_min"] < 0.0:
                            rejection_reason = "negative_raw_gamma_squared"
                            rejection_details = {"guards": trial_guards}
                        elif trial_guards["max_abs_u_over_c"] >= 1.0:
                            rejection_reason = "superluminal_shell_velocity"
                            rejection_details = {"guards": trial_guards}
                    except Exception as dynamic_error:
                        rejection_reason = (
                            f"{type(dynamic_error).__name__}: {dynamic_error}"
                        )
                    if rejection_reason is None:
                        break
                    if rejection_reason == "core_density_nonmonotonic":
                        density_limiter_requested = True
                    density_half_events += 1
                    next_dynamic_dt = dynamic_dt * args.density_half_factor
                    event = {
                        "event": "dynamic_child_split",
                        "dynamic_anchor_tau": float(dynamic_anchor_tau),
                        "post_hse_dynamic_elapsed_before_t0": float(
                            dynamic_elapsed_t0
                        ),
                        "level": int(dynamic_level),
                        "rejected_dt_t0": float(dynamic_dt),
                        "next_child_dt_t0": float(next_dynamic_dt),
                        "reason": rejection_reason,
                        "details": rejection_details,
                    }
                    append_jsonl(history, event)
                    atomic_write_json(
                        args.output_dir
                        / "transient_checkpoints"
                        / "latest_dynamic_rejection.json",
                        event,
                    )
                    if next_dynamic_dt < args.dt_t0_min:
                        raise FloatingPointError(
                            "dynamic density/state recovery fell below the configured minimum timestep"
                        )
                    dynamic_dt = float(next_dynamic_dt)
                    dynamic_level += 1
                previous_state = state
                state = trial
                dynamic_elapsed_t0 = float(dynamic_elapsed_t0 + dynamic_dt)
                tau = float(dynamic_anchor_tau + dynamic_elapsed_t0)
                branch_steps += 1
                dynamic_steps += 1
                last_dt = float(dynamic_dt)
                last_cfl = float(full_cfl)
                latest_density = trial_density
                latest_smfp = smfp_range_metrics(state, a_face, params, scales)
                if density_limiter_requested:
                    append_jsonl(
                        history,
                        {
                            "event": "accepted_dynamic_monotone_volume_limiter",
                            "dynamic_anchor_tau": float(dynamic_anchor_tau),
                            "post_hse_dynamic_elapsed_t0": float(
                                dynamic_elapsed_t0
                            ),
                            "accepted_dt_t0": float(dynamic_dt),
                            "sublevel": int(dynamic_level),
                            "density": trial_density,
                            "limited_radius_rs": float(
                                args.core_radius_limit_rs
                            ),
                        },
                    )
                if dynamic_level > 0:
                    dynamic_child_payload = make_payload()
                    dynamic_child_payload.update(
                        {
                            "event": "accepted_dynamic_child_checkpoint",
                            "sublevel": int(dynamic_level),
                        }
                    )
                    target = (
                        args.output_dir
                        / "transient_checkpoints"
                        / f"dynamic_half_level_{dynamic_level:02d}_latest"
                    )
                    atomic_save_npy(target / "checkpoint_state.npy", state.as_array())
                    atomic_write_json(
                        target / "checkpoint_meta.json", dynamic_child_payload
                    )

                horizon_face = trial_guards["outermost_future_trapped_face"]
                if (
                    trial_guards["compactness_max"]
                    >= args.horizon_compactness_threshold
                    and horizon_face is not None
                ):
                    horizon_event = make_payload()
                    horizon_event.update(
                        {
                            "event": "first_numerical_apparent_horizon",
                            "horizon_criterion": (
                                "outermost face with theta_plus<=0, theta_minus<0; "
                                "compactness threshold used as a supporting diagnostic"
                            ),
                            "horizon_face": int(horizon_face),
                            "guards": trial_guards,
                            "density": trial_density,
                            "formed_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        }
                    )
                    horizon_path = save_horizon_event(
                        args.output_dir,
                        previous_state,
                        state,
                        horizon_event,
                    )
                    append_jsonl(
                        history,
                        {"horizon_event_dir": str(horizon_path), **horizon_event},
                    )
                    print(json.dumps(horizon_event, sort_keys=True), flush=True)

            formal_output_due = bool(
                mode == "hse" and tau >= checkpoint_tau - 1.0e-12
            )
            if mode == "dynamic":
                formal_output_due = bool(
                    dynamic_next_output_elapsed_t0 is not None
                    and dynamic_elapsed_t0
                    >= dynamic_next_output_elapsed_t0
                    - max(args.dt_t0_min, 32.0 * np.finfo(float).eps * max(1.0, dynamic_next_output_elapsed_t0))
                )
            if formal_output_due:
                if mode == "hse":
                    tau = checkpoint_tau
                else:
                    tau = float(dynamic_anchor_tau + dynamic_next_output_elapsed_t0)
                payload = make_payload()
                payload["event"] = "checkpoint"
                save_formal_checkpoint(args.output_dir, state, payload, args.output_interval_t0)
                append_jsonl(history, payload)
                if mode == "dynamic":
                    dynamic_next_output_elapsed_t0 = float(
                        dynamic_next_output_elapsed_t0 + args.output_interval_t0
                    )

            if branch_steps % args.progress_every == 0:
                progress = make_payload()
                progress.update(
                    {
                        "event": "progress",
                        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
                atomic_write_json(args.output_dir / "latest_progress.json", progress)
                print(json.dumps(progress, sort_keys=True), flush=True)
    except Exception as exc:
        exit_code = 2
        failure = {
            "event": "run_failure",
            "tau": float(tau),
            "evolution_mode": mode,
            "branch_accepted_steps": int(branch_steps),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_jsonl(history, failure)
        atomic_write_json(args.output_dir / "latest_failure.json", failure)

    bounded_completion = bool(
        stop["signal"] is None
        and failure is None
        and (
            tau >= args.max_t0
            or step_limit_reached()
            or horizon_event is not None
        )
    )
    if stop["signal"] is not None or failure is not None or bounded_completion:
        payload = make_payload()
        payload.update(
            {
                "event": "forced_stop_checkpoint",
                "forced_stop_checkpoint": True,
                "stop_signal": stop["signal"],
                "failure": failure,
                "bounded_completion": bounded_completion,
                "horizon_event": horizon_event,
                "stopped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        target = save_stop_checkpoint(args.output_dir, state, payload)
        append_jsonl(history, {"stop_checkpoint_dir": str(target), **payload})
        print(
            json.dumps(
                {"event": "forced_stop_checkpoint", "tau": tau, "mode": mode, "path": str(target)},
                sort_keys=True,
            ),
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
