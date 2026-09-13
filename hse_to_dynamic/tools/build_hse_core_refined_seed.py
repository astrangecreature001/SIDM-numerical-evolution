"""Build an isolated tau=480 seed with refinement only inside the HSE mass boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sidm_engine import (  # noqa: E402
    PhysicalParams,
    State,
    StepControls,
    check_state,
    compute_cfl_dt_t0,
    compute_scales,
    horizon_metric,
)
from tools.build_fine_hse_switch_seed import (  # noqa: E402
    atomic_json,
    atomic_npy,
    conservative_positive_cell_remap,
    density_from_geometry,
    geometric_grid,
    rebuild_dependents,
    sha256_file,
    smfp_metrics,
    validate_faces,
)


def relative_max(target: np.ndarray, source: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(target), np.abs(source)), 1.0e-300)
    return float(np.max(np.abs(target - source) / scale))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-grid-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-hse-face-end", type=int, default=30)
    parser.add_argument("--target-core-cells", type=int, default=121)
    parser.add_argument("--first-shell-msun", type=float, default=1.0)
    parser.add_argument("--halo-mass-msun", type=float, default=6.3e9)
    parser.add_argument("--rs-kpc", type=float, default=2.6)
    parser.add_argument("--sigma0", type=float, default=5.0)
    parser.add_argument("--gamma-eos", type=float, default=5.0 / 3.0)
    parser.add_argument("--fixed-point-max-iter", type=int, default=40)
    parser.add_argument("--fixed-point-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()

    source_state_path = args.source_checkpoint_dir / "checkpoint_state.npy"
    source_meta_path = args.source_checkpoint_dir / "checkpoint_meta.json"
    source_a_path = args.source_grid_dir / "A.npy"
    source_array = np.asarray(np.load(source_state_path), dtype=np.float64)
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    source_a = np.asarray(np.load(source_a_path), dtype=np.float64)
    validate_faces("source A", source_a)
    if source_array.shape != (11, source_a.size):
        raise ValueError("source state and mass grid shapes do not match")
    if abs(float(source_meta["tau"]) - 480.0) > 5.0e-10:
        raise ValueError(f"expected tau=480 checkpoint, got {source_meta['tau']}")
    source = State.from_initial(source_array)

    boundary = int(args.source_hse_face_end)
    if not 3 <= boundary < source_a.size - 2:
        raise ValueError("source HSE face is outside the usable mass grid")
    boundary_a = float(source_a[boundary])
    first_shell_code = args.first_shell_msun / args.halo_mass_msun
    core_a, core_ratio = geometric_grid(
        boundary_a,
        first_shell_code,
        args.target_core_cells,
    )
    target_a = np.concatenate((core_a, source_a[boundary + 1 :]))
    validate_faces("HSE-core-refined target A", target_a)
    target_core_end = int(args.target_core_cells)
    if not np.array_equal(target_a[target_core_end:], source_a[boundary:]):
        raise RuntimeError("exterior mass faces were not preserved exactly")

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
    if int(source_smfp["face_end"]) != boundary:
        raise RuntimeError(
            f"source SMFP boundary changed: expected {boundary}, got {source_smfp['face_end']}"
        )

    source_volume = (4.0 * np.pi / 3.0) * source.r**3
    target_volume = np.interp(target_a, source_a, source_volume)
    target_r = np.cbrt(np.maximum(3.0 * target_volume / (4.0 * np.pi), 0.0))
    target_r[0] = 0.0
    if not np.all(np.diff(target_r) > 0.0):
        raise RuntimeError("proper-volume interpolation did not preserve shell order")

    target_u = np.interp(target_a, source_a, source.u)
    target_u[0] = 0.0
    target_ephi = np.exp(np.interp(target_a, source_a, np.log(source.ephi)))
    target_ephi[-1] = source.ephi[-1]
    target_core_epsilon, core_epsilon_normalization = conservative_positive_cell_remap(
        core_a,
        source_a[: boundary + 1],
        source.epsilon[:boundary],
    )
    target_epsilon = np.concatenate(
        (target_core_epsilon, source.epsilon[boundary:-1].copy())
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
        cfl_safety=0.8,
        cfl_window_cells=1,
        implicit_heat_enabled=True,
        tridiagonal_backend="numba",
        dt_t0_min=1.0e-24,
        retry_shrink=0.5,
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
    target_cfl = compute_cfl_dt_t0(
        target,
        target_a,
        params,
        scales,
        controls,
        cell_start=int(target_smfp["face_end"]),
    )
    source_cfl = compute_cfl_dt_t0(
        source,
        source_a,
        params,
        scales,
        controls,
        cell_start=boundary,
    )
    source_compactness, source_compactness_face = horizon_metric(source, scales)
    target_compactness, target_compactness_face = horizon_metric(target, scales)

    source_outer_start = boundary
    target_outer_start = target_core_end
    outer_checks = {
        "A_exact": bool(
            np.array_equal(target_a[target_outer_start:], source_a[source_outer_start:])
        ),
        "R_relative_max": relative_max(
            target.r[target_outer_start:], source.r[source_outer_start:]
        ),
        "u_relative_max": relative_max(
            target.u[target_outer_start:], source.u[source_outer_start:]
        ),
        "epsilon_relative_max": relative_max(
            target.epsilon[target_outer_start:-1], source.epsilon[source_outer_start:-1]
        ),
    }
    audit = {
        "schema": "sidm_tau480_hse_core_only_refined_seed_v1",
        "exploratory": True,
        "source_tau_t0": float(source_meta["tau"]),
        "source_hse_face_end": boundary,
        "target_hse_face_end_after_remap": int(target_smfp["face_end"]),
        "source_face_count": int(source_a.size),
        "target_face_count": int(target_a.size),
        "source_cell_count": int(source_a.size - 1),
        "target_cell_count": int(target_a.size - 1),
        "source_hse_cell_count": boundary,
        "requested_target_core_cells": target_core_end,
        "source_hse_rest_mass_msun": boundary_a * args.halo_mass_msun,
        "target_hse_rest_mass_msun": float(target_a[target_core_end] * args.halo_mass_msun),
        "hse_rest_mass_relative_error": float(
            abs(target_a[target_core_end] - boundary_a) / boundary_a
        ),
        "target_first_shell_mass_msun": float(
            np.diff(target_a)[0] * args.halo_mass_msun
        ),
        "target_core_geometric_shell_ratio": core_ratio,
        "total_rest_mass_relative_error": float(
            abs(target_a[-1] - source_a[-1]) / source_a[-1]
        ),
        "exterior_grid_and_state_checks": outer_checks,
        "source_smfp": source_smfp,
        "target_smfp": target_smfp,
        "source_exterior_only_cfl_dt_t0_at_safety_0p8": float(source_cfl),
        "target_exterior_only_cfl_dt_t0_at_safety_0p8": float(target_cfl),
        "requested_dt_cap_t0": 1.0e-4,
        "dt_cap_over_target_cfl": float(1.0e-4 / target_cfl),
        "source_compactness": float(source_compactness),
        "source_compactness_face": int(source_compactness_face),
        "target_compactness": float(target_compactness),
        "target_compactness_face": int(target_compactness_face),
        "compactness_relative_change": float(
            (target_compactness - source_compactness) / source_compactness
        ),
        "core_epsilon_integral_normalization": core_epsilon_normalization,
        "target_all_finite": bool(np.all(np.isfinite(target.as_array()))),
        "target_radius_strictly_increasing": bool(np.all(np.diff(target.r) > 0.0)),
        "target_density_positive": bool(np.all(target.rho > 0.0)),
        "target_epsilon_positive": bool(np.all(target.epsilon > 0.0)),
        "state_health_ok": bool(ok),
        "state_health_reason": reason,
        "dependent_fixed_point": fixed_point,
        "source_files": {
            "checkpoint_state": {
                "path": str(source_state_path),
                "sha256": sha256_file(source_state_path),
            },
            "checkpoint_meta": {
                "path": str(source_meta_path),
                "sha256": sha256_file(source_meta_path),
            },
            "A": {"path": str(source_a_path), "sha256": sha256_file(source_a_path)},
        },
    }

    grid_dir = args.output_root / "grids" / "gamma_hsefine1"
    seed_dir = args.output_root / "seeds" / "tau480_hsefine1"
    validation_dir = args.output_root / "validation"
    seed_meta = dict(source_meta)
    seed_meta.update(
        {
            "schema": audit["schema"],
            "mode": "tau480_hse_core_only_refined_seed",
            "gamma_a1_power": 1,
            "hybrid_steps": 0,
            "density_half_events": 0,
            "accepted_nonmonotonic_half_steps": 0,
            "last_dt_t0": 0.0,
            "last_cfl_dt_t0": float(target_cfl),
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
    atomic_json(
        args.output_root / "regrid_manifest.json",
        {
            "schema": audit["schema"],
            "files": {
                str(path.relative_to(args.output_root)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(args.output_root.rglob("*"))
                if path.is_file()
            },
        },
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
