"""Build a conservative fine-grid seed at the coarse-run HSE switch time.

The source checkpoint is an accepted dynamic, no-baryon state.  The target
Lagrangian mass grid keeps the exact total rest mass, uses a requested first
shell mass, and pins faces at the source compactness maximum and continuous
lambda/H=1 crossing.  Proper volume and face quantities are interpolated in
enclosed rest mass; specific internal energy is remapped conservatively.  All
dependent cell and metric rows are then rebuilt on the target grid.

This script only prepares the switch seed.  The production HSE runner must
recompute the contiguous central lambda/H<=1 region and perform the first HSE
projection from this state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sidm_engine import (  # noqa: E402
    PhysicalParams,
    State,
    StepControls,
    avg_face_to_cell,
    check_state,
    compute_cfl_dt_t0,
    compute_e_a,
    compute_heat_flux,
    compute_pressure_factor,
    compute_scales,
    horizon_metric,
    iterate_mass_gamma,
    mean_free_path_to_scale_height,
    update_gamma,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, np.ascontiguousarray(array))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_faces(name: str, faces: np.ndarray) -> None:
    if faces.ndim != 1 or faces.size < 4:
        raise ValueError(f"{name} must be a one-dimensional face array")
    if not np.all(np.isfinite(faces)):
        raise ValueError(f"{name} contains non-finite values")
    if faces[0] != 0.0 or not np.all(np.diff(faces) > 0.0):
        raise ValueError(f"{name} must start at zero and be strictly increasing")


def geometric_grid(total_mass: float, first_shell: float, cell_count: int) -> tuple[np.ndarray, float]:
    if not total_mass > first_shell > 0.0:
        raise ValueError("require total_mass > first_shell > 0")
    if cell_count < 3:
        raise ValueError("cell_count must be at least three")

    def total_for_ratio(ratio: float) -> float:
        if abs(ratio - 1.0) < 1.0e-14:
            return first_shell * cell_count
        return first_shell * math.expm1(cell_count * math.log(ratio)) / (ratio - 1.0)

    low = 1.0
    high = 1.01
    while total_for_ratio(high) < total_mass:
        high *= 1.05
        if high > 4.0:
            raise RuntimeError("could not bracket the geometric mass ratio")
    for _ in range(160):
        mid = 0.5 * (low + high)
        if total_for_ratio(mid) < total_mass:
            low = mid
        else:
            high = mid
    ratio = 0.5 * (low + high)
    shell_mass = first_shell * ratio ** np.arange(cell_count, dtype=np.float64)
    shell_mass *= total_mass / float(np.sum(shell_mass))
    # Restore the requested first shell exactly, distributing its roundoff to
    # the last shell without changing the total mass.
    correction = first_shell - float(shell_mass[0])
    shell_mass[0] = first_shell
    shell_mass[-1] -= correction
    faces = np.concatenate(([0.0], np.cumsum(shell_mass)))
    faces[-1] = total_mass
    validate_faces("target geometric A", faces)
    return faces, ratio


def pin_faces(faces: np.ndarray, named_values: tuple[tuple[str, float], ...]) -> dict[str, int]:
    pinned: dict[str, int] = {}
    used: set[int] = set()
    for name, value in named_values:
        if not faces[0] < value < faces[-1]:
            raise ValueError(f"pin {name}={value} lies outside target mass range")
        candidates = np.argsort(np.abs(faces - value))
        selected = None
        for raw in candidates:
            index = int(raw)
            if index <= 1 or index >= faces.size - 1 or index in used:
                continue
            if faces[index - 1] < value < faces[index + 1]:
                selected = index
                break
        if selected is None:
            raise RuntimeError(f"could not pin target face for {name}")
        faces[selected] = value
        used.add(selected)
        pinned[name] = selected
    validate_faces("pinned target A", faces)
    return pinned


def _log_segment_integral(
    left: float,
    right: float,
    centers: np.ndarray,
    log_values: np.ndarray,
) -> float:
    log_left = float(np.interp(left, centers, log_values, left=log_values[0], right=log_values[-1]))
    log_right = float(np.interp(right, centers, log_values, left=log_values[0], right=log_values[-1]))
    delta = log_right - log_left
    width = right - left
    if abs(delta) < 1.0e-12:
        return math.exp(0.5 * (log_left + log_right)) * width
    return math.exp(log_left) * math.expm1(delta) / delta * width


def conservative_positive_cell_remap(
    target_faces: np.ndarray,
    source_faces: np.ndarray,
    source_values: np.ndarray,
) -> tuple[np.ndarray, float]:
    if np.any(~np.isfinite(source_values)) or np.any(source_values <= 0.0):
        raise ValueError("source cell quantity must be finite and positive")
    centers = 0.5 * (source_faces[:-1] + source_faces[1:])
    logs = np.log(source_values)
    result = np.empty(target_faces.size - 1, dtype=np.float64)
    for cell, (left, right) in enumerate(zip(target_faces[:-1], target_faces[1:])):
        interior = centers[(centers > left) & (centers < right)]
        knots = np.concatenate(([left], interior, [right]))
        integral = sum(
            _log_segment_integral(float(a), float(b), centers, logs)
            for a, b in zip(knots[:-1], knots[1:])
        )
        result[cell] = integral / (right - left)
    source_integral = float(np.sum(source_values * np.diff(source_faces)))
    target_integral = float(np.sum(result * np.diff(target_faces)))
    normalization = source_integral / target_integral
    result *= normalization
    return result, normalization


def density_from_geometry(a_face: np.ndarray, state: State) -> np.ndarray:
    volume = (4.0 * np.pi / 3.0) * np.diff(state.r**3)
    if np.any(volume <= 0.0):
        raise ValueError("target shell volume is not positive")
    return avg_face_to_cell(state.gamma_lorentz) * np.diff(a_face) / volume


def fill_thermodynamics(
    state: State,
    rho: np.ndarray,
    epsilon: np.ndarray,
    params: PhysicalParams,
    scales,
) -> None:
    p_fac = compute_pressure_factor(
        u_right=state.u[1:],
        shell_width=np.diff(state.r),
        rho_cell=rho,
        epsilon_cell=epsilon,
        params=params,
        scales=scales,
        u_left=state.u[:-1],
        r_left=state.r[:-1],
        r_right=state.r[1:],
    )
    pressure = (params.gamma - 1.0) * rho * epsilon * p_fac
    enthalpy = 1.0 + (epsilon + pressure / np.maximum(rho, 1.0e-300)) / scales.r0_scale
    for full, cell in (
        (state.rho, rho),
        (state.epsilon, epsilon),
        (state.pressure, pressure),
        (state.enthalpy, enthalpy),
    ):
        full[:-1] = cell
        full[-1] = cell[-1]


def rebuild_dependents(
    state: State,
    a_face: np.ndarray,
    epsilon: np.ndarray,
    params: PhysicalParams,
    scales,
    max_iter: int,
    tolerance: float,
) -> dict[str, object]:
    converged = False
    mass_change = math.inf
    rho_change = math.inf
    for iteration in range(1, max_iter + 1):
        old_mass = state.mass.copy()
        old_rho = state.rho[:-1].copy()
        state.gamma_lorentz = update_gamma(state, scales)
        rho = density_from_geometry(a_face, state)
        fill_thermodynamics(state, rho, epsilon, params, scales)
        state.q = compute_heat_flux(state, a_face, params, scales)
        state.e_a = compute_e_a(state.q, state.r, state.rho[:-1])
        iterate_mass_gamma(state, a_face, params, scales, max_iter=32, rel_tol=tolerance)
        mass_scale = np.maximum(np.maximum(np.abs(state.mass), np.abs(old_mass)), 1.0e-30)
        rho_scale = np.maximum(np.maximum(np.abs(state.rho[:-1]), np.abs(old_rho)), 1.0e-30)
        mass_change = float(np.max(np.abs(state.mass - old_mass) / mass_scale))
        rho_change = float(np.max(np.abs(state.rho[:-1] - old_rho) / rho_scale))
        if max(mass_change, rho_change) <= tolerance:
            converged = True
            break
    state.gamma_lorentz = update_gamma(state, scales)
    rho = density_from_geometry(a_face, state)
    fill_thermodynamics(state, rho, epsilon, params, scales)
    state.q = compute_heat_flux(state, a_face, params, scales)
    state.e_a = compute_e_a(state.q, state.r, state.rho[:-1])
    iterate_mass_gamma(state, a_face, params, scales, max_iter=64, rel_tol=tolerance)
    return {
        "converged": converged,
        "iterations": iteration,
        "mass_relative_change": mass_change,
        "density_relative_change": rho_change,
    }


def smfp_metrics(state: State, a_face: np.ndarray, params: PhysicalParams, scales) -> dict[str, object]:
    ratio = mean_free_path_to_scale_height(state, params, scales)
    prefix = 0
    for value in ratio:
        if np.isfinite(value) and value <= 1.0:
            prefix += 1
        else:
            break
    centers = 0.5 * (a_face[:-1] + a_face[1:])
    if prefix == 0:
        boundary = 0.0
    elif prefix >= ratio.size:
        boundary = float(a_face[-1])
    else:
        x0, x1 = float(centers[prefix - 1]), float(centers[prefix])
        y0, y1 = float(ratio[prefix - 1]), float(ratio[prefix])
        # Match display_diagnostics.smfp_range_metrics exactly: the published
        # diagnostic and the HSE runner use a linear K=lambda/H crossing
        # between adjacent cell centers, not a logarithmic interpolation.
        fraction = min(1.0, max(0.0, (1.0 - y0) / (y1 - y0))) if y1 != y0 else 0.5
        boundary = x0 + fraction * (x1 - x0)
    return {
        "face_end": prefix,
        "boundary_a": boundary,
        "central_lambda_over_h": float(ratio[0]),
        "minimum_lambda_over_h": float(np.min(ratio)),
    }


def density_order(rho: np.ndarray) -> dict[str, object]:
    rise = rho[1:] / np.maximum(rho[:-1], 1.0e-300) - 1.0
    bad = np.flatnonzero(rise > 1.0e-12)
    return {
        "nonmonotonic_count": int(bad.size),
        "first_nonmonotonic_cell": int(bad[0]) if bad.size else None,
        "maximum_fractional_rise": float(np.max(rise)) if rise.size else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-grid-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-cells", type=int, default=1000)
    parser.add_argument("--first-shell-msun", type=float, default=10.0)
    parser.add_argument("--halo-mass-msun", type=float, default=6.3e9)
    parser.add_argument("--rs-kpc", type=float, default=2.6)
    parser.add_argument("--sigma0", type=float, default=5.0)
    parser.add_argument("--gamma-eos", type=float, default=5.0 / 3.0)
    parser.add_argument("--fixed-point-max-iter", type=int, default=40)
    parser.add_argument("--fixed-point-tolerance", type=float, default=1.0e-12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_state_path = args.source_checkpoint_dir / "checkpoint_state.npy"
    source_meta_path = args.source_checkpoint_dir / "checkpoint_meta.json"
    source_a_path = args.source_grid_dir / "A.npy"
    source_array = np.asarray(np.load(source_state_path), dtype=np.float64)
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    source_a = np.asarray(np.load(source_a_path), dtype=np.float64)
    validate_faces("source A", source_a)
    if source_array.shape != (11, source_a.size):
        raise ValueError("source state and mass grid shapes do not match")
    source = State.from_initial(source_array)

    params = PhysicalParams(
        rs_kpc=args.rs_kpc,
        halo_mass_msun=args.halo_mass_msun,
        sigma0_cgs=args.sigma0,
        gamma=args.gamma_eos,
        pressure_factor_enabled=True,
        pressure_factor_model="flow-divergence",
        pressure_factor_mach_transition=0.3,
        baryon_profile="none",
        baryon_mass_fraction=0.0,
        baryon_scale_radius_rs=0.0,
    )
    scales = compute_scales(params)
    source_smfp = smfp_metrics(source, source_a, params, scales)
    compactness, compactness_index = horizon_metric(source, scales)
    target_a, geometric_ratio = geometric_grid(
        float(source_a[-1]),
        args.first_shell_msun / args.halo_mass_msun,
        args.target_cells,
    )
    pinned = pin_faces(
        target_a,
        (
            ("source_compactness_peak", float(source_a[compactness_index])),
            ("source_continuous_smfp_boundary", float(source_smfp["boundary_a"])),
        ),
    )

    source_volume = (4.0 * np.pi / 3.0) * source.r**3
    target_volume = np.interp(target_a, source_a, source_volume)
    target_r = np.cbrt(np.maximum(3.0 * target_volume / (4.0 * np.pi), 0.0))
    target_r[0] = 0.0
    if not np.all(np.diff(target_r) > 0.0):
        raise RuntimeError("proper-volume interpolation did not preserve shell order")
    target_u = np.interp(target_a, source_a, source.u)
    target_u[0] = 0.0
    if np.any(source.ephi <= 0.0):
        raise ValueError("source lapse is not positive")
    target_ephi = np.exp(np.interp(target_a, source_a, np.log(source.ephi)))
    target_ephi[-1] = source.ephi[-1]
    target_epsilon, epsilon_normalization = conservative_positive_cell_remap(
        target_a, source_a, source.epsilon[:-1]
    )

    target = State(
        u=target_u,
        r=target_r,
        rho=np.ones_like(target_a),
        epsilon=np.ones_like(target_a),
        pressure=np.ones_like(target_a),
        enthalpy=np.ones_like(target_a),
        ephi=target_ephi,
        mass=np.interp(target_a, source_a, source.mass),
        gamma_lorentz=np.interp(target_a, source_a, source.gamma_lorentz),
        e_a=np.zeros_like(target_a),
        q=np.zeros_like(target_a),
    )
    target.rho[:-1] = density_from_geometry(target_a, target)
    target.rho[-1] = target.rho[-2]
    fixed_point = rebuild_dependents(
        target,
        target_a,
        target_epsilon,
        params,
        scales,
        args.fixed_point_max_iter,
        args.fixed_point_tolerance,
    )

    controls = StepControls(
        cfl_safety=0.25,
        cfl_window_cells=1,
        implicit_heat_enabled=True,
        tridiagonal_backend="numba",
        dt_t0_min=1.0e-24,
        max_retries=0,
        max_fractional_epsilon_change=0.0,
        max_fractional_density_change=0.0,
        min_shell_width_ratio=0.0,
        max_abs_u_code=float("inf"),
        max_density_code=float("inf"),
        max_epsilon_code=float("inf"),
        max_mass_code=float("inf"),
        momentum_update_mode="heun",
        outer_velocity_boundary_enabled=True,
        outer_velocity_boundary_width=20,
        outer_velocity_boundary_passes=2,
    )
    ok, reason = check_state(target, controls, scales)
    if not ok:
        raise RuntimeError(f"target state health check failed: {reason}")
    if not bool(fixed_point["converged"]):
        raise RuntimeError(f"dependent-row fixed point did not converge: {fixed_point}")

    target_smfp = smfp_metrics(target, target_a, params, scales)
    target_compactness, target_compactness_index = horizon_metric(target, scales)
    cfl_dt = compute_cfl_dt_t0(
        target,
        target_a,
        params,
        scales,
        controls,
        cell_start=max(0, int(target_smfp["face_end"]) - 1),
    )
    source_energy = float(np.sum(source.epsilon[:-1] * np.diff(source_a)))
    target_energy = float(np.sum(target.epsilon[:-1] * np.diff(target_a)))
    source_centers = 0.5 * (source_a[:-1] + source_a[1:])
    target_centers = 0.5 * (target_a[:-1] + target_a[1:])
    source_rho_interp = np.exp(np.interp(target_centers, source_centers, np.log(source.rho[:-1])))
    source_eps_interp = np.exp(np.interp(target_centers, source_centers, np.log(source.epsilon[:-1])))
    rho_profile_rel = np.abs(target.rho[:-1] / source_rho_interp - 1.0)
    eps_profile_rel = np.abs(target.epsilon[:-1] / source_eps_interp - 1.0)

    audit: dict[str, object] = {
        "schema": "sidm_gamma1_dynamic_to_fine10_hse_switch_seed_v1",
        "exploratory": True,
        "source_tau_t0": float(source_meta["tau"]),
        "source_mode": source_meta.get("mode"),
        "source_face_count": int(source_a.size),
        "target_face_count": int(target_a.size),
        "target_cell_count": int(target_a.size - 1),
        "source_total_rest_mass_code": float(source_a[-1]),
        "target_total_rest_mass_code": float(target_a[-1]),
        "total_rest_mass_relative_error": float(abs(target_a[-1] - source_a[-1]) / source_a[-1]),
        "target_first_shell_mass_msun": float(np.diff(target_a)[0] * args.halo_mass_msun),
        "target_geometric_shell_ratio": geometric_ratio,
        "pinned_faces": pinned,
        "source_smfp": source_smfp,
        "target_smfp": target_smfp,
        "source_compactness": float(compactness),
        "source_compactness_face": int(compactness_index),
        "target_compactness": float(target_compactness),
        "target_compactness_face": int(target_compactness_index),
        "compactness_relative_change": float((target_compactness - compactness) / compactness),
        "source_outer_dm_mass": float(source.mass[-1]),
        "target_outer_dm_mass": float(target.mass[-1]),
        "outer_dm_mass_relative_change": float((target.mass[-1] - source.mass[-1]) / source.mass[-1]),
        "source_epsilon_integral_dA": source_energy,
        "target_epsilon_integral_dA": target_energy,
        "epsilon_integral_relative_error": float(abs(target_energy - source_energy) / source_energy),
        "epsilon_reconstruction_normalization": epsilon_normalization,
        "rho_profile_relative_difference_median": float(np.median(rho_profile_rel)),
        "rho_profile_relative_difference_max": float(np.max(rho_profile_rel)),
        "epsilon_profile_relative_difference_median": float(np.median(eps_profile_rel)),
        "epsilon_profile_relative_difference_max": float(np.max(eps_profile_rel)),
        "source_density_order": density_order(source.rho[:-1]),
        "target_density_order": density_order(target.rho[:-1]),
        "target_dynamic_exterior_cfl_dt_t0": float(cfl_dt),
        "target_all_finite": bool(np.all(np.isfinite(target.as_array()))),
        "target_radius_strictly_increasing": bool(np.all(np.diff(target.r) > 0.0)),
        "target_density_positive": bool(np.all(target.rho[:-1] > 0.0)),
        "target_epsilon_positive": bool(np.all(target.epsilon[:-1] > 0.0)),
        "target_gamma_positive": bool(np.all(target.gamma_lorentz > 0.0)),
        "state_health_ok": bool(ok),
        "state_health_reason": reason,
        "dependent_fixed_point": fixed_point,
        "source_files": {
            "checkpoint_state": {"path": str(source_state_path), "sha256": sha256_file(source_state_path)},
            "checkpoint_meta": {"path": str(source_meta_path), "sha256": sha256_file(source_meta_path)},
            "A": {"path": str(source_a_path), "sha256": sha256_file(source_a_path)},
        },
    }

    grid_dir = args.output_root / "grids" / "gamma_fine10"
    seed_dir = args.output_root / "seeds" / "gamma_fine10"
    validation_dir = args.output_root / "validation"
    seed_meta = dict(source_meta)
    seed_meta.update(
        {
            "schema": audit["schema"],
            "mode": "dynamic_nonhse_regridded_hse_switch_seed",
            "gamma_a1_power": 1,
            "hybrid_steps": 0,
            "density_half_events": 0,
            "accepted_nonmonotonic_half_steps": 0,
            "last_dt_t0": 0.0,
            "last_cfl_dt_t0": float(cfl_dt),
            "regrid_audit": audit,
            "hse_boundary_policy": "recompute outer face of current contiguous central lambda/H<=1 region every physical step",
        }
    )
    atomic_npy(grid_dir / "A.npy", target_a)
    atomic_npy(grid_dir / "initial.npy", target.as_array())
    atomic_json(grid_dir / "grid_summary.json", audit)
    atomic_npy(seed_dir / "checkpoint_state.npy", target.as_array())
    atomic_json(seed_dir / "checkpoint_meta.json", seed_meta)
    atomic_json(validation_dir / "regrid_audit.json", audit)
    manifest = {
        "schema": audit["schema"],
        "files": {
            str(path.relative_to(args.output_root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(args.output_root.rglob("*"))
            if path.is_file()
        },
    }
    atomic_json(args.output_root / "regrid_manifest.json", manifest)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
