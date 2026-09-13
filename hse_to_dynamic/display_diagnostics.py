"""Core structure and connected SMFP-region diagnostics.

The HSE runner uses smfp_range_metrics to locate the core boundary.
collect_display_diagnostics supplies additional snapshot measurements."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sidm_engine import (
    NFW_RHO_S_PER_CODE_DENSITY,
    PhysicalParams,
    Scales,
    State,
    mean_free_path_to_scale_height,
)


DENSITY_CODE_UNIT_RHO_S = NFW_RHO_S_PER_CODE_DENSITY
CORE_DIAGNOSTIC_VERSION = "three_segment_log_density_v1"
SMFP_DIAGNOSTIC_VERSION = "continuous_lambda_over_h_crossing_v1"


def _safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _running_median(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    half = window // 2
    result = np.empty_like(values)
    for index in range(values.size):
        result[index] = np.median(
            values[max(0, index - half) : min(values.size, index + half + 1)]
        )
    return result


def density_core_range_metrics(
    state: State,
    *,
    smooth_dex: float = 0.14,
    fit_drop_dex: float = 6.0,
    shelf_slope_min: float = -1.2,
    shelf_slope_max: float = 0.7,
    envelope_steepening: float = 0.45,
    shelf_min_width_dex: float = 0.18,
) -> dict[str, Any]:
    """Fit the existing spike/shelf/envelope density-core display boundary."""

    radius_face = np.asarray(state.r, dtype=np.float64)
    rho_cell = np.asarray(state.rho[:-1], dtype=np.float64)
    radius_cell = np.sqrt(radius_face[:-1] * radius_face[1:])
    valid = (
        (radius_cell > 0.0)
        & (radius_cell <= 0.3)
        & (rho_cell > 0.0)
        & np.isfinite(radius_cell)
        & np.isfinite(rho_cell)
    )
    x = np.log10(radius_cell[valid])
    y = np.log10(rho_cell[valid])
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x.size < 30:
        raise ValueError("too few positive inner-profile cells")

    preview_x = np.linspace(x[0], x[-1], 700)
    preview_y = np.interp(preview_x, x, y)
    preview_dx = preview_x[1] - preview_x[0]
    preview_y = _running_median(
        preview_y,
        max(5, int(round(0.12 / preview_dx)) | 1),
    )
    inner_count = max(20, preview_y.size // 10)
    inner_reference = float(np.percentile(preview_y[:inner_count], 70))
    below = np.flatnonzero(preview_y < inner_reference - fit_drop_dex)
    fit_x_end = float(preview_x[below[0]]) if below.size else float(x[-1])
    fit_x_end = max(fit_x_end, float(x[0] + 1.0))
    fit_x_end = min(fit_x_end, float(x[-1]))

    fit_x = np.linspace(x[0], fit_x_end, 260)
    fit_y = np.interp(fit_x, x, y)
    fit_dx = fit_x[1] - fit_x[0]
    fit_y = _running_median(
        fit_y,
        max(5, int(round(smooth_dex / fit_dx)) | 1),
    )

    min_spike = max(3, int(round(0.08 / fit_dx)))
    min_shelf = max(6, int(round(shelf_min_width_dex / fit_dx)))
    min_envelope = max(10, int(round(0.35 / fit_dx)))
    best: dict[str, float] | None = None
    for first in range(min_spike, fit_x.size - min_shelf - min_envelope, 2):
        break_one = fit_x[first]
        for second in range(first + min_shelf, fit_x.size - min_envelope, 2):
            break_two = fit_x[second]
            design = np.column_stack(
                (
                    np.ones_like(fit_x),
                    fit_x,
                    np.maximum(0.0, fit_x - break_one),
                    np.maximum(0.0, fit_x - break_two),
                )
            )
            coefficients = np.linalg.lstsq(design, fit_y, rcond=None)[0]
            slope_spike = float(coefficients[1])
            slope_shelf = float(slope_spike + coefficients[2])
            slope_envelope = float(slope_shelf + coefficients[3])
            if not (shelf_slope_min <= slope_shelf <= shelf_slope_max):
                continue
            if not (-5.5 <= slope_envelope <= slope_shelf - envelope_steepening):
                continue
            residual = fit_y - design @ coefficients
            cost = float(
                np.mean(residual[:first] ** 2)
                + np.mean(residual[first:second] ** 2)
                + np.mean(residual[second:] ** 2)
            )
            if best is None or cost < best["fit_cost"]:
                best = {
                    "fit_cost": cost,
                    "break_spike_log10_r": float(break_one),
                    "break_core_log10_r": float(break_two),
                    "slope_spike": slope_spike,
                    "slope_core_shelf": slope_shelf,
                    "slope_outer_envelope": slope_envelope,
                    "fit_x_end_log10_r": fit_x_end,
                }

    if best is None:
        raise RuntimeError("no admissible spike-shelf-envelope fit")

    core_radius = 10.0 ** best["break_core_log10_r"]
    shell_volume = (4.0 * math.pi / 3.0) * (
        radius_face[1:] ** 3 - radius_face[:-1] ** 3
    )
    denominator = radius_face[1:] ** 3 - radius_face[:-1] ** 3
    partial_fraction = np.divide(
        core_radius**3 - radius_face[:-1] ** 3,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    partial_fraction = np.clip(partial_fraction, 0.0, 1.0)
    included_volume = shell_volume * partial_fraction
    core_volume = float(np.sum(included_volume))
    if not core_volume > 0.0:
        raise RuntimeError("fitted core contains zero shell volume")
    core_mean_code = float(np.sum(rho_cell * included_volume) / core_volume)
    core_face_end = int(np.searchsorted(radius_face, core_radius, side="right"))

    return {
        "density_core_metric_version": CORE_DIAGNOSTIC_VERSION,
        "density_core_face_end": core_face_end,
        "density_core_radius_rs": float(core_radius),
        "density_core_mean_density_code": core_mean_code,
        "density_core_mean_density_rho_s": (
            core_mean_code * DENSITY_CODE_UNIT_RHO_S
        ),
        "density_core_included_cells_equivalent": float(np.sum(partial_fraction)),
        **best,
    }


def smfp_range_metrics(
    state: State,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
) -> dict[str, Any]:
    """Measure the contiguous central K=lambda/H <= 1 region for display."""

    ratio = mean_free_path_to_scale_height(state, params, scales)
    a_face = np.asarray(a_face, dtype=np.float64)
    if a_face.shape != state.r.shape:
        raise ValueError(f"A/state shape mismatch: {a_face.shape} != {state.r.shape}")
    if ratio.size != a_face.size - 1:
        raise ValueError(f"K/A cell-count mismatch: {ratio.size} != {a_face.size - 1}")

    cell_end = 0
    for value in ratio:
        if np.isfinite(value) and value <= 1.0:
            cell_end += 1
        else:
            break
    face_end = min(state.r.size - 1, cell_end) if cell_end > 0 else 0
    boundary_a: float | None = None
    boundary_radius: float | None = None
    mass_fraction = 0.0
    interpolation_fraction: float | None = None
    boundary_method = "no_central_smfp_region"

    if cell_end == ratio.size and ratio.size > 0:
        boundary_a = float(a_face[-1])
        boundary_radius = float(state.r[-1])
        mass_fraction = float(state.mass[-1])
        boundary_method = "all_cells_smfp_outer_face"
    elif 0 < cell_end < ratio.size:
        lo = cell_end - 1
        hi = cell_end
        k_lo = float(ratio[lo])
        k_hi = float(ratio[hi])
        denominator = k_hi - k_lo
        if np.isfinite(k_lo) and np.isfinite(k_hi) and denominator > 0.0:
            interpolation_fraction = float(
                np.clip((1.0 - k_lo) / denominator, 0.0, 1.0)
            )
            a_cell = 0.5 * (a_face[:-1] + a_face[1:])
            boundary_a = float(
                a_cell[lo] + interpolation_fraction * (a_cell[hi] - a_cell[lo])
            )
            boundary_radius = float(np.interp(boundary_a, a_face, state.r))
            mass_fraction = float(np.interp(boundary_a, a_face, state.mass))
            boundary_method = "linear_k_cell_center_crossing"
        else:
            boundary_a = float(a_face[face_end])
            boundary_radius = float(state.r[face_end])
            mass_fraction = float(state.mass[face_end])
            boundary_method = "invalid_crossing_fallback_discrete_face"

    return {
        "smfp_metric_version": SMFP_DIAGNOSTIC_VERSION,
        "smfp_cell_end": int(cell_end),
        "smfp_face_end": int(face_end),
        "smfp_cell_count": int(cell_end),
        "smfp_boundary_method": boundary_method,
        "smfp_boundary_a": _safe_float(boundary_a)
        if boundary_a is not None
        else None,
        "smfp_boundary_radius_rs": _safe_float(boundary_radius)
        if boundary_radius is not None
        else None,
        "smfp_boundary_interpolation_fraction": _safe_float(interpolation_fraction)
        if interpolation_fraction is not None
        else None,
        "smfp_mass_fraction": _safe_float(mass_fraction),
        "smfp_lambda_over_h_central": _safe_float(ratio[0])
        if ratio.size
        else None,
        "smfp_lambda_over_h_min": _safe_float(float(np.min(ratio)))
        if ratio.size
        else None,
    }


def collect_display_diagnostics(
    state: State,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
) -> dict[str, Any]:
    """Return diagnostics without ever raising into the numerical evolution."""

    output: dict[str, Any] = {
        "display_diagnostics_affect_evolution": False,
    }
    try:
        output.update(density_core_range_metrics(state))
        output["density_core_diagnostic_ok"] = True
    except Exception as exc:  # display-only by design
        output["density_core_diagnostic_ok"] = False
        output["density_core_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    try:
        output.update(smfp_range_metrics(state, a_face, params, scales))
        output["smfp_diagnostic_ok"] = True
    except Exception as exc:  # display-only by design
        output["smfp_diagnostic_ok"] = False
        output["smfp_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    return output
