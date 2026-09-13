"""Relativistic Lagrangian SIDM evolution and thermodynamic utilities.

Provides state variables, physical scales, heat transport, hydrodynamic
updates, and adaptive step acceptance controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


TINY = 1e-300
NFW_RHO_S_PER_CODE_DENSITY = 4.0 * np.pi * (np.log(11.0) - 10.0 / 11.0)
_NUMBA_THOMAS = None
# This isolated branch uses the corrected Appendix-A1 force coefficient.
A1_GAMMA_POWER = 1


@dataclass
class PhysicalParams:
    rs_kpc: float = 2.6
    halo_mass_msun: float = 6.3e9
    sigma0_cgs: float = 5.0
    # Exact coefficients stated in the paper.  Keep the EOS adiabatic index
    # distinct from the Misner-Sharp/Lorentz factor stored as gamma_lorentz.
    a: float = float(np.sqrt(16.0 / np.pi))
    b: float = float(25.0 * np.sqrt(np.pi) / 32.0)
    gamma: float = 5.0 / 3.0
    ephib: float = 1.0
    conductivity_c: float = 0.75
    pressure_factor_enabled: bool = True
    # ``paper_cell`` reproduces Appendix A13 literally with Delta R/|U|.
    # ``flow_divergence`` replaces the numerical cell-crossing time by the
    # continuum spherical expansion time 1/max(div U, 0), which has a regular
    # grid-refinement limit.  ``off`` is equivalent to f_P = 1.
    pressure_factor_model: str = "paper_cell"
    pressure_factor_mach_transition: float = 0.3
    heat_enabled: bool = True
    heat_in_phi_enabled: bool = True
    mass_evolution_enabled: bool = True
    ephi_evolution_enabled: bool = True
    mass_heat_flux_enabled: bool = True
    pressure_factor_floor: float = 0.0
    # A non-positive limit disables the corresponding numerical clip.  The
    # paper-corrected branch therefore has no hidden lapse-gradient clipping.
    max_ln_ephi_cell_jump: float = 0.0
    max_abs_ln_ephi: float = 0.0
    phi_heat_derivative_floor_t0: float = 0.0
    energy_denominator_floor: float = 1e-12
    epsilon_floor: float = 1e-14
    baryon_profile: str = "none"
    baryon_mass_fraction: float = 0.0
    baryon_scale_radius_rs: float = 0.1
    baryon_powerlaw_index: float = 0.6


@dataclass
class Scales:
    r0_scale: float
    sigma: float
    dt_code_per_t0: float
    t0_gyr: float
    fq: float


@dataclass
class StepControls:
    cfl_safety: float = 0.5
    cfl_window_cells: int = 1
    implicit_heat_enabled: bool = False
    implicit_heat_theta: float = 1.0
    tridiagonal_backend: str = "thomas"
    post_step_smoothing_passes: int = 0
    heat_flux_smoothing_passes: int = 0
    epsilon_filter_passes: int = 0
    epsilon_filter_strength: float = 1.0
    heat_flux_outer_taper_width: int = 0
    smooth_thermodynamic_state: bool = False
    dt_t0_min: float = 1e-12
    retry_shrink: float = 0.5
    max_retries: int = 12
    rollback_shrink: float = 0.5
    max_rollback_events: int = 256
    heat_fraction_safety: float = 0.0
    heat_delta_fraction_limit: float = 0.0
    heat_face_contrast_fraction: float = 0.0
    max_fractional_epsilon_change: float = 0.2
    max_fractional_density_change: float = 0.5
    min_shell_width_ratio: float = 0.2
    outer_velocity_boundary_enabled: bool = False
    outer_velocity_boundary_width: int = 20
    outer_velocity_boundary_passes: int = 2
    max_abs_u_code: float = 1e12
    max_epsilon_code: float = 1e6
    max_density_code: float = 1e12
    max_mass_code: float = 10.0
    radius_repair_enabled: bool = False
    min_shell_width_fraction: float = 1e-6
    shell_width_regularization_ratio: float = 0.0
    shell_width_regularization_strength: float = 0.0
    shell_width_regularization_passes: int = 1
    shell_width_regularization_pad: int = 2
    energy_update_mode: str = "trapezoid"
    energy_update_denominator_switch: float = 1e-8
    momentum_update_mode: str = "euler"
    artificial_viscosity_linear: float = 0.0
    artificial_viscosity_quadratic: float = 0.0


@dataclass
class State:
    u: np.ndarray
    r: np.ndarray
    rho: np.ndarray
    epsilon: np.ndarray
    pressure: np.ndarray
    enthalpy: np.ndarray
    ephi: np.ndarray
    mass: np.ndarray
    gamma_lorentz: np.ndarray
    e_a: np.ndarray
    q: np.ndarray

    @classmethod
    def from_initial(cls, initial: np.ndarray) -> "State":
        if initial.shape[0] != 11:
            raise ValueError(f"expected initial shape (11, N), got {initial.shape}")
        return cls(*(np.ascontiguousarray(initial[i], dtype=np.float64) for i in range(11)))

    def copy(self) -> "State":
        return State(*(np.ascontiguousarray(arr.copy()) for arr in self.as_tuple()))

    def as_tuple(self) -> Tuple[np.ndarray, ...]:
        return (
            self.u,
            self.r,
            self.rho,
            self.epsilon,
            self.pressure,
            self.enthalpy,
            self.ephi,
            self.mass,
            self.gamma_lorentz,
            self.e_a,
            self.q,
        )

    def as_array(self) -> np.ndarray:
        return np.vstack(self.as_tuple())


def compute_scales(params: PhysicalParams) -> Scales:
    term_r = params.rs_kpc / 2.6
    term_m = params.halo_mass_msun / 6.3e9
    r0_scale = (term_r / term_m) * 8.5e6
    sigma = params.sigma0_cgs * (2e33 * params.halo_mass_msun) / (
        (1.48e5 * params.halo_mass_msun) ** 2
    )
    dt_code_per_t0 = 1.35e12 * (params.sigma0_cgs ** -1.0) * (term_m ** -2.5) * (
        term_r**3.5
    )
    t0_gyr = 1.33 * (params.sigma0_cgs ** -1.0) * (term_m ** -1.5) * (term_r**3.5)
    fq = -1.5 * (params.gamma - 1.0) ** 1.5 * params.a
    return Scales(
        r0_scale=r0_scale,
        sigma=sigma,
        dt_code_per_t0=dt_code_per_t0,
        t0_gyr=t0_gyr,
        fq=fq,
    )


def mean_free_path_to_scale_height(
    state: State,
    params: PhysicalParams,
    scales: Scales,
) -> np.ndarray:
    """Return the paper's cell-centered transport ratio lambda/H.

    In code units, rho_phys = rho/R0^3 and
    v^2 = P/rho = (gamma-1) epsilon/R0 when f_P is inactive. With
    lambda = 1/(rho_phys sigma) and H^2 = v^2/(4 pi rho_phys), this gives
    lambda/H = R0^2/sigma * sqrt(4 pi/((gamma-1) rho epsilon)).
    """

    rho_cell = np.maximum(state.rho[:-1], TINY)
    epsilon_cell = np.maximum(state.epsilon[:-1], TINY)
    thermal_factor = max(float(params.gamma) - 1.0, TINY)
    ratio = (
        scales.r0_scale**2
        / max(scales.sigma, TINY)
        * np.sqrt(4.0 * np.pi / (thermal_factor * rho_cell * epsilon_cell))
    )
    return np.nan_to_num(ratio, nan=np.inf, posinf=np.inf, neginf=np.inf)


def baryon_enclosed_mass(r_face: np.ndarray, params: PhysicalParams) -> np.ndarray:
    """Static enclosed baryon mass in the same mass units as ``A`` and ``m``.

    ``r_face`` is measured in units of Rs.  The default profile is disabled.
    For every profile, ``baryon_mass_fraction`` is a mass amplitude in the same
    code units as ``A``.  For the imported grid, one unit is M_NFW(<10 Rs), not
    M0=4*pi*rho_s*Rs**3.  A Feng coefficient stated in M0 units must therefore
    be converted before this function is called.  The power-law option evaluates
    M_b(r)=f_code*r**alpha.  The softened Feng option preserves that outer power
    law but uses a finite-density core, avoiding singular central acceleration
    on grids whose first face is r=0.
    """
    profile = params.baryon_profile.lower().strip()
    f_b = float(params.baryon_mass_fraction)
    r = np.maximum(np.asarray(r_face, dtype=np.float64), 0.0)
    if profile in ("", "none", "off") or f_b <= 0.0:
        return np.zeros_like(r)

    scale = max(float(params.baryon_scale_radius_rs), TINY)
    if profile == "plummer":
        denom = np.maximum(r * r + scale * scale, TINY) ** 1.5
        mass = f_b * r**3 / denom
    elif profile == "hernquist":
        mass = f_b * r**2 / np.maximum((r + scale) ** 2, TINY)
    elif profile in ("powerlaw", "feng2021"):
        alpha = float(params.baryon_powerlaw_index)
        mass = f_b * np.power(r, alpha)
    elif profile in ("powerlaw-softened", "softened-powerlaw", "feng2021-softened", "softened-feng2021"):
        alpha = float(params.baryon_powerlaw_index)
        # Outside the softening radius this tends to f_b r^alpha; inside it
        # tends to r^3, giving a finite central baryon density and force.
        denom_power = 0.5 * max(3.0 - alpha, 0.0)
        mass = f_b * r**3 / np.maximum(r * r + scale * scale, TINY) ** denom_power
    else:
        raise ValueError(f"unknown baryon profile: {params.baryon_profile}")

    mass[0] = 0.0
    return np.maximum(mass, 0.0)


def total_gravity_mass(state: State, params: PhysicalParams) -> np.ndarray:
    """Dark matter Misner-Sharp mass plus static baryon mass for gravity."""
    return state.mass + baryon_enclosed_mass(state.r, params)


def avg_cell_to_face(v_cell: np.ndarray) -> np.ndarray:
    out = np.empty(len(v_cell) + 1, dtype=np.float64)
    out[1:-1] = 0.5 * (v_cell[:-1] + v_cell[1:])
    out[0] = v_cell[0]
    out[-1] = v_cell[-1]
    return out


def avg_face_to_cell(v_face: np.ndarray) -> np.ndarray:
    return 0.5 * (v_face[:-1] + v_face[1:])


def deriv_cell_to_face(v_cell: np.ndarray, a_face: np.ndarray) -> np.ndarray:
    grad = np.zeros(len(v_cell) + 1, dtype=np.float64)
    denom = 0.5 * (a_face[2:] - a_face[:-2])
    grad[1:-1] = (v_cell[1:] - v_cell[:-1]) / np.maximum(denom, TINY)
    grad[0] = grad[1]
    grad[-1] = grad[-2]
    return grad


def deriv_face_to_cell(v_face: np.ndarray, a_face: np.ndarray) -> np.ndarray:
    return (v_face[1:] - v_face[:-1]) / np.maximum(a_face[1:] - a_face[:-1], TINY)


def face_signal_speed(state: State, params: PhysicalParams, scales: Scales) -> np.ndarray:
    rho_face = avg_cell_to_face(state.rho[:-1])
    pressure_face = avg_cell_to_face(state.pressure[:-1])
    enthalpy_face = avg_cell_to_face(state.enthalpy[:-1])
    # Relativistic adiabatic sound speed: c_s^2/c^2 = gamma P/(rho h).
    # Velocities in the state are scaled so U/c = u/R0.
    sound_code = np.sqrt(
        np.maximum(
            params.gamma
            * pressure_face
            / np.maximum(rho_face * enthalpy_face, TINY)
            * scales.r0_scale,
            0.0,
        )
    )
    return np.abs(state.u) + sound_code


def compute_cfl_dt_t0(
    state: State,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls,
    cell_start: int = 0,
    cell_stop: int | None = None,
) -> float:
    del a_face
    dr_full = np.diff(state.r)
    signal = face_signal_speed(state, params, scales)
    signal_cell_full = 0.5 * (signal[:-1] + signal[1:])
    start = min(max(0, int(cell_start)), dr_full.size)
    stop = dr_full.size if cell_stop is None else min(max(start, int(cell_stop)), dr_full.size)
    dr = dr_full[start:stop]
    signal_cell = signal_cell_full[start:stop]
    window = max(1, int(controls.cfl_window_cells))
    if window > 1 and dr.size >= window:
        width_sum = np.convolve(dr, np.ones(window, dtype=np.float64), mode="valid")
        signal_windows = np.lib.stride_tricks.sliding_window_view(signal_cell, window)
        signal_max = np.max(signal_windows, axis=1)
        effective_width = width_sum / window
        dt_code = effective_width * scales.r0_scale**2 / np.maximum(signal_max, TINY)
    else:
        dt_code = dr * scales.r0_scale**2 / np.maximum(signal_cell, TINY)
    finite_positive = dt_code[np.isfinite(dt_code) & (dt_code > 0.0)]
    if finite_positive.size == 0:
        return controls.dt_t0_min
    return controls.cfl_safety * float(np.min(finite_positive)) / scales.dt_code_per_t0


def compute_pressure_factor(
    u_right: np.ndarray,
    shell_width: np.ndarray,
    rho_cell: np.ndarray,
    epsilon_cell: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
    *,
    u_left: np.ndarray | None = None,
    r_left: np.ndarray | None = None,
    r_right: np.ndarray | None = None,
) -> np.ndarray:
    p_fac = np.ones_like(rho_cell)
    model = str(params.pressure_factor_model).strip().lower().replace("_", "-")
    if not params.pressure_factor_enabled or model in ("", "off", "none"):
        return p_fac

    u_abs = np.abs(u_right)
    t_sc = np.zeros_like(rho_cell)
    mach_gate = np.ones_like(rho_cell)

    if model == "paper-cell":
        # Literal Appendix A13 discretization.  This is retained only for an
        # auditable paper comparison; Delta R is a numerical cell width, so
        # this model is not invariant under grid refinement.
        active = u_right > 1e-99
        t_sc[active] = (shell_width[active] * scales.r0_scale) / np.maximum(
            u_abs[active] / scales.r0_scale,
            TINY,
        )
    elif model == "flow-divergence":
        if u_left is None or r_left is None or r_right is None:
            raise ValueError(
                "flow-divergence pressure factor requires u_left, r_left, and r_right"
            )
        u_left_arr = np.asarray(u_left, dtype=np.float64)
        r_left_arr = np.asarray(r_left, dtype=np.float64)
        r_right_arr = np.asarray(r_right, dtype=np.float64)
        volume_denominator = np.maximum(r_right_arr**3 - r_left_arr**3, TINY)
        # Spherical divergence: div U = 3 (R_r^2 U_r - R_l^2 U_l)
        # / (R_r^3 - R_l^3).  U is stored in solver velocity units, hence the
        # physical expansion rate is div(U)/R0^2 and t_flow=R0^2/div(U).
        expansion_rate_code = 3.0 * (
            r_right_arr**2 * np.asarray(u_right, dtype=np.float64)
            - r_left_arr**2 * u_left_arr
        ) / volume_denominator
        active = expansion_rate_code > 1e-99
        t_sc[active] = scales.r0_scale**2 / expansion_rate_code[active]

        # The paper motivates the correction only when bulk expansion becomes
        # comparable to random motion.  A smooth Mach gate avoids an arbitrary
        # discontinuous switch while leaving the quasi-static LMFP phase at
        # f_P ~= 1.  This remains a candidate closure, not a Boltzmann solution.
        random_speed = np.sqrt(
            np.maximum((params.gamma - 1.0) * epsilon_cell * scales.r0_scale, TINY)
        )
        outward_bulk_speed = np.maximum(
            0.5 * (u_left_arr + np.asarray(u_right, dtype=np.float64)), 0.0
        )
        mach = outward_bulk_speed / random_speed
        mach_transition = max(float(params.pressure_factor_mach_transition), TINY)
        mach_gate = mach**2 / (mach**2 + mach_transition**2)
    else:
        raise ValueError(f"unknown pressure_factor_model: {params.pressure_factor_model}")

    if not np.any(active):
        return p_fac
    tr_denom = (
        params.a
        * scales.sigma
        * (rho_cell / scales.r0_scale**3)
        # This is the one-dimensional random velocity dispersion entering the
        # relaxation time, not the adiabatic sound speed used by the CFL test.
        * np.sqrt(
            np.maximum((params.gamma - 1.0) * epsilon_cell / scales.r0_scale, 0.0)
        )
    )
    valid = active & (tr_denom >= 1e-99) & (t_sc > 0.0)
    p_fac[active & ~valid] = params.pressure_factor_floor
    tr = np.zeros_like(rho_cell)
    tr[valid] = 1.0 / tr_denom[valid]
    raw_factor = np.ones_like(rho_cell)
    raw_factor[valid] = 1.0 / (tr[valid] / t_sc[valid] + 1.0)
    p_fac[valid] = 1.0 - mach_gate[valid] * (1.0 - raw_factor[valid])
    return np.clip(p_fac, params.pressure_factor_floor, 1.0)


def artificial_viscosity_pressure(
    state: State,
    scales: Scales,
    controls: StepControls | None,
) -> np.ndarray:
    q_visc = np.zeros_like(state.pressure[:-1])
    if controls is None:
        return q_visc
    linear = max(0.0, float(controls.artificial_viscosity_linear))
    quadratic = max(0.0, float(controls.artificial_viscosity_quadratic))
    if linear == 0.0 and quadratic == 0.0:
        return q_visc

    compression_speed = np.maximum(state.u[:-1] - state.u[1:], 0.0)
    active = compression_speed > 0.0
    if not np.any(active):
        return q_visc

    rho_cell = state.rho[:-1]
    pressure_cell = state.pressure[:-1]
    sound_cell = np.sqrt(
        np.maximum(pressure_cell / np.maximum(rho_cell, TINY), 0.0) * scales.r0_scale
    )
    q_visc[active] = (
        rho_cell[active]
        / scales.r0_scale
        * (
            linear * sound_cell[active] * compression_speed[active]
            + quadratic * compression_speed[active] ** 2
        )
    )
    return np.nan_to_num(q_visc, nan=0.0, posinf=0.0, neginf=0.0)


def effective_momentum_pressure(
    state: State,
    scales: Scales,
    controls: StepControls | None,
) -> np.ndarray:
    return state.pressure[:-1] + artificial_viscosity_pressure(state, scales, controls)


def compute_heat_flux(state: State, a_face: np.ndarray, params: PhysicalParams, scales: Scales) -> np.ndarray:
    """Evaluate the paper's A10/Eq. (24) heat flux.

    The gradient is d(e^phi epsilon)/dA.  No Misner-Sharp Gamma belongs in
    this constitutive relation; Gamma enters the energy equation separately.
    """
    q_new = np.zeros_like(state.q)
    if not params.heat_enabled:
        return q_new

    rho_cell = state.rho[:-1]
    pressure_cell = state.pressure[:-1]
    epsilon_cell = state.epsilon[:-1]
    ephi_cell = avg_face_to_cell(state.ephi)
    ephi_epsilon = epsilon_cell * ephi_cell
    d_ephi_epsilon_da = deriv_cell_to_face(ephi_epsilon, a_face)

    epsilon_face = avg_cell_to_face(epsilon_cell)
    pressure_face = avg_cell_to_face(pressure_cell)
    inv_rho_face = avg_cell_to_face(1.0 / np.maximum(rho_cell, TINY))
    denom = (
        1.0 / params.conductivity_c
        + params.a
        / params.b
        * scales.sigma**2
        / (4.0 * np.pi * scales.r0_scale**4)
        * pressure_face
    )

    q_new[1:] = (
        scales.fq
        * np.sqrt(np.maximum(epsilon_face[1:], 0.0))
        * pressure_face[1:]
        / np.maximum(inv_rho_face[1:], TINY)
        * state.r[1:] ** 2
        / np.maximum(state.ephi[1:], TINY)
        * d_ephi_epsilon_da[1:]
        / np.maximum(denom[1:], TINY)
    )
    q_new[0] = 0.0
    return q_new


def heat_flux_taper_weights(n_face: int, controls: StepControls) -> np.ndarray:
    weights = np.ones(n_face, dtype=np.float64)
    width = max(0, int(controls.heat_flux_outer_taper_width))
    if width <= 0 or n_face < 3:
        return weights
    start = max(1, n_face - width)
    if start >= n_face - 1:
        weights[-1] = 0.0
        return weights
    weights[start:] = np.linspace(1.0, 0.0, n_face - start)
    weights[-1] = 0.0
    return weights


def compute_linear_heat_flux_from_epsilon(
    epsilon_cell: np.ndarray,
    state: State,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls,
) -> np.ndarray:
    q_new = np.zeros_like(state.q)
    if not params.heat_enabled:
        return q_new

    rho_cell = state.rho[:-1]
    pressure_cell = state.pressure[:-1]
    ephi_cell = avg_face_to_cell(state.ephi)
    d_ephi_epsilon_da = deriv_cell_to_face(ephi_cell * epsilon_cell, a_face)

    # Freeze the nonlinear conductivity coefficient at the old state.  This
    # keeps the heat solve tridiagonal while still using the paper heat-flux
    # operator for the new temperature gradient.
    epsilon_face = avg_cell_to_face(state.epsilon[:-1])
    pressure_face = avg_cell_to_face(pressure_cell)
    inv_rho_face = avg_cell_to_face(1.0 / np.maximum(rho_cell, TINY))
    denom = (
        1.0 / params.conductivity_c
        + params.a
        / params.b
        * scales.sigma**2
        / (4.0 * np.pi * scales.r0_scale**4)
        * pressure_face
    )
    weights = heat_flux_taper_weights(q_new.size, controls)

    q_new[1:] = (
        weights[1:]
        * scales.fq
        * np.sqrt(np.maximum(epsilon_face[1:], 0.0))
        * pressure_face[1:]
        / np.maximum(inv_rho_face[1:], TINY)
        * state.r[1:] ** 2
        / np.maximum(state.ephi[1:], TINY)
        * d_ephi_epsilon_da[1:]
        / np.maximum(denom[1:], TINY)
    )
    q_new[0] = 0.0
    # The implicit production path uses an insulating outer boundary q=0.
    q_new[-1] = 0.0
    return q_new


def solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    backend: str = "thomas",
) -> np.ndarray:
    n = diagonal.size
    if lower.size != max(n - 1, 0) or upper.size != max(n - 1, 0) or rhs.size != n:
        raise ValueError("incompatible tridiagonal array sizes")
    if n == 0:
        return rhs.copy()

    selected_backend = str(backend).strip().lower()
    if selected_backend == "numba":
        global _NUMBA_THOMAS
        if _NUMBA_THOMAS is None:
            try:
                from numba import njit
            except ImportError as exc:
                raise RuntimeError(
                    "tridiagonal backend 'numba' was requested but Numba is unavailable"
                ) from exc

            @njit(cache=False, fastmath=False)
            def numba_thomas(
                lower_jit: np.ndarray,
                diagonal_jit: np.ndarray,
                upper_jit: np.ndarray,
                rhs_jit: np.ndarray,
            ) -> np.ndarray:
                n_jit = diagonal_jit.size
                if n_jit == 0:
                    return rhs_jit.copy()
                c_prime_jit = np.zeros(max(n_jit - 1, 0), dtype=np.float64)
                d_prime_jit = np.zeros(n_jit, dtype=np.float64)
                denom_jit = max(diagonal_jit[0], TINY)
                if n_jit > 1:
                    c_prime_jit[0] = upper_jit[0] / denom_jit
                d_prime_jit[0] = rhs_jit[0] / denom_jit
                for i_jit in range(1, n_jit):
                    denom_jit = (
                        diagonal_jit[i_jit]
                        - lower_jit[i_jit - 1] * c_prime_jit[i_jit - 1]
                    )
                    if abs(denom_jit) < TINY:
                        denom_jit = TINY if denom_jit >= 0.0 else -TINY
                    if i_jit < n_jit - 1:
                        c_prime_jit[i_jit] = upper_jit[i_jit] / denom_jit
                    d_prime_jit[i_jit] = (
                        rhs_jit[i_jit]
                        - lower_jit[i_jit - 1] * d_prime_jit[i_jit - 1]
                    ) / denom_jit
                solution_jit = np.zeros(n_jit, dtype=np.float64)
                solution_jit[-1] = d_prime_jit[-1]
                for i_jit in range(n_jit - 2, -1, -1):
                    solution_jit[i_jit] = (
                        d_prime_jit[i_jit]
                        - c_prime_jit[i_jit] * solution_jit[i_jit + 1]
                    )
                return solution_jit

            _NUMBA_THOMAS = numba_thomas
        return _NUMBA_THOMAS(lower, diagonal, upper, rhs)
    if selected_backend != "thomas":
        raise ValueError(f"unsupported tridiagonal backend: {backend}")

    c_prime = np.zeros(max(n - 1, 0), dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)
    denom = max(diagonal[0], TINY)
    if n > 1:
        c_prime[0] = upper[0] / denom
    d_prime[0] = rhs[0] / denom
    for i in range(1, n):
        denom = diagonal[i] - lower[i - 1] * c_prime[i - 1]
        if abs(denom) < TINY:
            denom = TINY if denom >= 0.0 else -TINY
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    solution = np.zeros(n, dtype=np.float64)
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution


def implicit_heat_predictor(
    state: State,
    a_face: np.ndarray,
    dt_code: float,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls,
) -> Tuple[np.ndarray, np.ndarray]:
    eps_old = state.epsilon[:-1]
    n_cell = eps_old.size
    if not params.heat_enabled or n_cell <= 1:
        return np.zeros_like(state.q), np.zeros_like(eps_old)

    rho_cell = state.rho[:-1]
    pressure_cell = state.pressure[:-1]
    ephi_cell = avg_face_to_cell(state.ephi)
    epsilon_face = avg_cell_to_face(eps_old)
    pressure_face = avg_cell_to_face(pressure_cell)
    inv_rho_face = avg_cell_to_face(1.0 / np.maximum(rho_cell, TINY))
    denom = (
        1.0 / params.conductivity_c
        + params.a
        / params.b
        * scales.sigma**2
        / (4.0 * np.pi * scales.r0_scale**4)
        * pressure_face
    )
    weights = heat_flux_taper_weights(state.q.size, controls)
    d_a = np.diff(a_face)

    lower = np.zeros(n_cell - 1, dtype=np.float64)
    diagonal = np.ones(n_cell, dtype=np.float64)
    upper = np.zeros(n_cell - 1, dtype=np.float64)
    rhs = eps_old.copy()
    theta = float(np.clip(controls.implicit_heat_theta, 0.5, 1.0))
    theta_dt = theta * dt_code
    explicit_dt = (1.0 - theta) * dt_code

    interior = slice(1, n_cell)
    grad_width = np.maximum(0.5 * (a_face[2 : n_cell + 1] - a_face[: n_cell - 1]), TINY)
    heat_coeff = (
        weights[interior]
        * scales.fq
        * np.sqrt(np.maximum(epsilon_face[interior], 0.0))
        * pressure_face[interior]
        / np.maximum(inv_rho_face[interior], TINY)
        * state.r[interior] ** 2
        / np.maximum(state.ephi[interior], TINY)
        / np.maximum(denom[interior], TINY)
    )
    flux_coeff = (
        4.0 * np.pi * state.r[interior] ** 2 * state.ephi[interior] ** 2
    )
    face_coeff = flux_coeff * heat_coeff / grad_width

    scale_denominator = scales.r0_scale**3.5
    left_scale = scales.sigma / (
        scale_denominator
        * np.maximum(ephi_cell[:-1], TINY)
        * np.maximum(d_a[:-1], TINY)
    )
    right_scale = scales.sigma / (
        scale_denominator
        * np.maximum(ephi_cell[1:], TINY)
        * np.maximum(d_a[1:], TINY)
    )

    l_ll = left_scale * face_coeff * ephi_cell[:-1]
    l_lr = -left_scale * face_coeff * ephi_cell[1:]
    l_rl = -right_scale * face_coeff * ephi_cell[:-1]
    l_rr = right_scale * face_coeff * ephi_cell[1:]

    # For each interior cell the original face loop adds the right-face
    # contribution first, followed by the left-face contribution.
    diagonal[1:] += -theta_dt * l_rr
    diagonal[:-1] += -theta_dt * l_ll
    upper[:] = -theta_dt * l_lr
    lower[:] = -theta_dt * l_rl

    if explicit_dt > 0.0:
        left_rhs = explicit_dt * (l_ll * eps_old[:-1] + l_lr * eps_old[1:])
        right_rhs = explicit_dt * (l_rl * eps_old[:-1] + l_rr * eps_old[1:])
        rhs[1:] += right_rhs
        rhs[:-1] += left_rhs

    eps_new = solve_tridiagonal(
        lower,
        diagonal,
        upper,
        rhs,
        backend=controls.tridiagonal_backend,
    )
    eps_new = np.maximum(eps_new, params.epsilon_floor)
    q_new = compute_linear_heat_flux_from_epsilon(eps_new, state, a_face, params, scales, controls)
    heat_flux_passes = max(0, int(controls.heat_flux_smoothing_passes))
    if heat_flux_passes > 0:
        q_new = smooth_interior(q_new, heat_flux_passes)
        q_new[0] = 0.0
        q_new[-1] = 0.0
    return q_new, eps_new - eps_old


def compute_predicted_heat_flux(
    state: State,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls,
) -> np.ndarray:
    q_predict = compute_heat_flux(state, a_face, params, scales)
    heat_flux_passes = max(0, int(controls.heat_flux_smoothing_passes))
    if heat_flux_passes > 0:
        q_predict = smooth_interior(q_predict, heat_flux_passes)
        q_predict[0] = 0.0
    return apply_outer_heat_flux_taper(q_predict, controls)


def initialize_heat_flux(
    state: State,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
    controls: Optional[StepControls] = None,
) -> State:
    initialized = state.copy()
    if controls is None:
        controls = StepControls()
    initialized.q = compute_predicted_heat_flux(
        initialized,
        a_face,
        params,
        scales,
        controls,
    )
    initialized.e_a = compute_e_a(initialized.q, initialized.r, initialized.rho[:-1])
    return initialized


def heat_energy_delta(
    q_face: np.ndarray,
    state: State,
    a_face: np.ndarray,
    dt_code: float,
    scales: Scales,
) -> np.ndarray:
    ephi_cell = avg_face_to_cell(state.ephi)
    flux = 4.0 * np.pi * state.r**2 * q_face * state.ephi**2
    div_flux = deriv_face_to_cell(flux, a_face)
    loss_rate = div_flux / scales.r0_scale**3.5 * scales.sigma / np.maximum(ephi_cell, TINY)
    return -loss_rate * dt_code


def apply_heat_delta_limiter(
    q_face: np.ndarray,
    delta_epsilon: np.ndarray,
    state: State,
    controls: StepControls,
) -> Tuple[np.ndarray, np.ndarray]:
    limit = float(controls.heat_delta_fraction_limit)
    if limit <= 0.0 or delta_epsilon.size == 0:
        return q_face, delta_epsilon
    rel_change = np.abs(delta_epsilon) / np.maximum(state.epsilon[:-1], TINY)
    finite = rel_change[np.isfinite(rel_change)]
    if finite.size == 0:
        return q_face, delta_epsilon
    max_rel = float(np.max(finite))
    if max_rel <= limit:
        return q_face, delta_epsilon
    scale = max(limit / max(max_rel, TINY), 0.0)
    return q_face * scale, delta_epsilon * scale


def apply_heat_face_contrast_limiter(
    q_face: np.ndarray,
    state: State,
    a_face: np.ndarray,
    dt_code: float,
    scales: Scales,
    controls: StepControls,
) -> np.ndarray:
    fraction = float(controls.heat_face_contrast_fraction)
    if fraction <= 0.0 or dt_code <= 0.0 or q_face.size <= 2:
        return q_face

    q_limited = q_face.copy()
    n_cell = state.epsilon.size - 1
    ephi_cell = avg_face_to_cell(state.ephi)
    theta = state.epsilon[:-1] * ephi_cell
    d_a = np.diff(a_face)
    flux_factor = 4.0 * np.pi * state.r**2 * state.ephi**2
    prefactor = dt_code * scales.sigma / scales.r0_scale**3.5
    max_fraction = max(fraction, 0.0)

    for face in range(1, n_cell):
        left = face - 1
        right = face
        contrast = theta[right] - theta[left]
        if not np.isfinite(contrast) or contrast == 0.0:
            q_limited[face] = 0.0
            continue

        flux = flux_factor[face] * q_limited[face]
        if not np.isfinite(flux) or flux == 0.0:
            continue

        delta_left = -prefactor * flux / (
            max(ephi_cell[left], TINY) * max(d_a[left], TINY)
        )
        delta_right = prefactor * flux / (
            max(ephi_cell[right], TINY) * max(d_a[right], TINY)
        )
        contrast_change = ephi_cell[right] * delta_right - ephi_cell[left] * delta_left
        if not np.isfinite(contrast_change):
            q_limited[face] = 0.0
            continue

        # Pure heat diffusion should reduce the local temperature contrast, not
        # increase or invert it in one hydrodynamic step.
        if contrast * contrast_change >= 0.0:
            q_limited[face] = 0.0
            continue
        allowed = max_fraction * abs(contrast)
        if abs(contrast_change) > allowed:
            q_limited[face] *= allowed / max(abs(contrast_change), TINY)

    q_limited[0] = 0.0
    q_limited[-1] = 0.0
    return q_limited


def compute_heat_dt_t0(
    state: State,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls,
) -> float:
    if controls.implicit_heat_enabled:
        return np.inf
    if not params.heat_enabled or controls.heat_fraction_safety <= 0.0:
        return np.inf
    q_predict = compute_predicted_heat_flux(state, a_face, params, scales, controls)
    delta_per_t0 = heat_energy_delta(q_predict, state, a_face, scales.dt_code_per_t0, scales)
    frac_per_t0 = np.abs(delta_per_t0) / np.maximum(state.epsilon[:-1], TINY)
    finite = frac_per_t0[np.isfinite(frac_per_t0) & (frac_per_t0 > 0.0)]
    if finite.size == 0:
        return np.inf
    return max(controls.dt_t0_min, controls.heat_fraction_safety / float(np.max(finite)))


def compute_phi_gradient(
    pressure_cell: np.ndarray,
    rho_cell: np.ndarray,
    enthalpy_cell: np.ndarray,
    ephi_face: np.ndarray,
    e_a_old: np.ndarray,
    e_a_new: np.ndarray,
    a_face: np.ndarray,
    dt_code: float,
    params: PhysicalParams,
    scales: Scales,
) -> np.ndarray:
    rho_face = avg_cell_to_face(rho_cell)
    enthalpy_face = avg_cell_to_face(enthalpy_cell)
    d_pressure_da = deriv_cell_to_face(pressure_cell, a_face)
    heat_term = np.zeros_like(ephi_face)
    if params.heat_in_phi_enabled and dt_code > 0.0:
        floor_code = params.phi_heat_derivative_floor_t0 * scales.dt_code_per_t0
        effective_dt = max(dt_code, floor_code) if floor_code > 0.0 else dt_code
        heat_term = (
            scales.sigma
            / scales.r0_scale**1.5
            / np.maximum(ephi_face, TINY)
            * (e_a_new - e_a_old)
            / effective_dt
        )
    bracket = d_pressure_da / np.maximum(rho_face, TINY) + heat_term
    dphi_da = -bracket / np.maximum(enthalpy_face, TINY)
    d_a = np.diff(a_face)
    face_width = np.empty_like(a_face)
    face_width[0] = d_a[0]
    face_width[-1] = d_a[-1]
    face_width[1:-1] = 0.5 * (d_a[:-1] + d_a[1:])
    dphi_da = np.nan_to_num(dphi_da, nan=0.0, posinf=0.0, neginf=0.0)
    if params.max_ln_ephi_cell_jump > 0.0 and np.isfinite(params.max_ln_ephi_cell_jump):
        limit = params.max_ln_ephi_cell_jump * scales.r0_scale / np.maximum(face_width, TINY)
        dphi_da = np.clip(dphi_da, -limit, limit)
    return dphi_da


def integrate_ephi(
    dphi_da: np.ndarray,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
) -> np.ndarray:
    phi_scaled = np.zeros_like(a_face)
    phi_scaled[-1] = np.log(params.ephib) * scales.r0_scale
    da = np.diff(a_face)
    increments = 0.5 * (dphi_da[:-1] + dphi_da[1:]) * da
    phi_scaled[:-1] = phi_scaled[-1] - np.cumsum(increments[::-1])[::-1]
    ln_ephi = phi_scaled / scales.r0_scale
    if params.max_abs_ln_ephi > 0.0 and np.isfinite(params.max_abs_ln_ephi):
        ln_ephi = np.clip(ln_ephi, -params.max_abs_ln_ephi, params.max_abs_ln_ephi)
    return np.exp(ln_ephi)


def compute_e_a(q_face: np.ndarray, r_face: np.ndarray, rho_cell: np.ndarray) -> np.ndarray:
    rho_face = avg_cell_to_face(rho_cell)
    denom = 4.0 * np.pi * np.maximum(r_face**2 * rho_face**2, TINY)
    return q_face / denom


def update_dm_misner_sharp_mass(
    a_face: np.ndarray,
    state: State,
    params: PhysicalParams,
    scales: Scales,
) -> np.ndarray:
    mass = np.zeros_like(state.mass)
    gamma_cell = avg_face_to_cell(state.gamma_lorentz)
    u_pair = state.u[:-1] + state.u[1:]
    q_pair = state.q[:-1] + state.q[1:]
    heat_flux_mass_source = np.zeros_like(gamma_cell)
    if params.mass_heat_flux_enabled:
        heat_flux_mass_source = (
            0.25
            * (u_pair / scales.r0_scale)
            * q_pair
            / np.maximum(state.rho[:-1], TINY)
        )
    material_specific_energy = 1.0 + state.epsilon[:-1] / scales.r0_scale
    # Paper Eq. (17): dm/dA = Gamma(1 + epsilon) + U q/rho.
    # Crucially, Gamma multiplies only the material-energy term, not U q/rho.
    increments = np.diff(a_face) * (
        gamma_cell * material_specific_energy + heat_flux_mass_source
    )
    mass[1:] = np.cumsum(increments)
    mass[0] = 0.0
    return mass


def update_gamma(state: State, scales: Scales) -> np.ndarray:
    gamma_face = np.ones_like(state.gamma_lorentz)
    valid = state.r > 0.0
    radicand = np.ones_like(state.r)
    radicand[valid] = (
        1.0
        + (state.u[valid] / scales.r0_scale) ** 2
        - 2.0 * state.mass[valid] / np.maximum(state.r[valid] * scales.r0_scale, TINY)
    )
    gamma_face[valid] = np.sqrt(np.maximum(radicand[valid], 0.0))
    gamma_face[0] = 1.0
    return gamma_face


def iterate_mass_gamma(
    state: State,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
    max_iter: int = 8,
    rel_tol: float = 1e-12,
) -> State:
    """Iterate the coupled Misner-Sharp mass and Gamma fixed point in place."""
    if not params.mass_evolution_enabled:
        state.gamma_lorentz = update_gamma(state, scales)
        return state
    for _ in range(max(1, int(max_iter))):
        previous_mass = state.mass.copy()
        state.gamma_lorentz = update_gamma(state, scales)
        state.mass = update_dm_misner_sharp_mass(a_face, state, params, scales)
        rel = np.max(np.abs(state.mass - previous_mass) / np.maximum(np.abs(state.mass), TINY))
        if float(rel) <= rel_tol:
            break
    state.gamma_lorentz = update_gamma(state, scales)
    state.mass = update_dm_misner_sharp_mass(a_face, state, params, scales)
    state.gamma_lorentz = update_gamma(state, scales)
    return state


def smooth_interior(values: np.ndarray, passes: int) -> np.ndarray:
    smoothed = np.ascontiguousarray(values.copy())
    if smoothed.size <= 2 or passes <= 0:
        return smoothed
    for _ in range(passes):
        old = smoothed.copy()
        smoothed[1:-1] = 0.25 * old[:-2] + 0.5 * old[1:-1] + 0.25 * old[2:]
    return smoothed


def apply_outer_velocity_boundary(u_face: np.ndarray, controls: StepControls) -> np.ndarray:
    if not controls.outer_velocity_boundary_enabled or u_face.size < 4:
        return u_face
    width = max(1, int(controls.outer_velocity_boundary_width))
    passes = max(0, int(controls.outer_velocity_boundary_passes))
    start = max(1, u_face.size - width)
    for _ in range(passes):
        previous_left = u_face[start - 1]
        for i in range(start, u_face.size - 1):
            old_i = u_face[i]
            u_face[i] = 0.25 * previous_left + 0.5 * u_face[i] + 0.25 * u_face[i + 1]
            previous_left = old_i
    extrapolated = 2.0 * u_face[-2] - u_face[-3]
    u_face[-1] = min(u_face[-2], extrapolated)
    return u_face


def apply_outer_heat_flux_taper(q_face: np.ndarray, controls: StepControls) -> np.ndarray:
    width = max(0, int(controls.heat_flux_outer_taper_width))
    if width <= 0 or q_face.size < 3:
        return q_face
    start = max(1, q_face.size - width)
    if start >= q_face.size - 1:
        q_face[-1] = 0.0
        return q_face
    weights = np.linspace(1.0, 0.0, q_face.size - start)
    q_face[start:] *= weights
    q_face[-1] = 0.0
    return q_face


def apply_epsilon_filter(
    state: State,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls,
) -> State:
    passes = max(0, int(controls.epsilon_filter_passes))
    strength = min(max(float(controls.epsilon_filter_strength), 0.0), 1.0)
    if passes == 0 or strength <= 0.0 or state.epsilon.size <= 3:
        return state

    epsilon_cell_old = np.maximum(state.epsilon[:-1], params.epsilon_floor)
    epsilon_cell_target = np.maximum(
        smooth_interior(epsilon_cell_old, passes), params.epsilon_floor
    )
    epsilon_cell = epsilon_cell_old + strength * (epsilon_cell_target - epsilon_cell_old)
    epsilon_cell = np.maximum(epsilon_cell, params.epsilon_floor)

    shell_weights = np.diff(a_face) * np.maximum(avg_face_to_cell(state.gamma_lorentz), 0.0)
    old_total = float(np.sum(shell_weights * epsilon_cell_old))
    new_total = float(np.sum(shell_weights * epsilon_cell))
    if (
        np.isfinite(old_total)
        and np.isfinite(new_total)
        and old_total > 0.0
        and new_total > 0.0
    ):
        epsilon_cell *= old_total / new_total
    epsilon_cell = np.maximum(epsilon_cell, params.epsilon_floor)

    rho_cell = np.maximum(state.rho[:-1], TINY)
    pressure_factor = np.ones_like(rho_cell)
    if params.pressure_factor_enabled:
        pressure_factor = compute_pressure_factor(
            u_right=state.u[1:],
            shell_width=np.diff(state.r),
            rho_cell=rho_cell,
            epsilon_cell=epsilon_cell,
            params=params,
            scales=scales,
            u_left=state.u[:-1],
            r_left=state.r[:-1],
            r_right=state.r[1:],
        )
    pressure_cell = (params.gamma - 1.0) * rho_cell * epsilon_cell * pressure_factor
    enthalpy_cell = 1.0 + (
        epsilon_cell + pressure_cell / np.maximum(rho_cell, TINY)
    ) / scales.r0_scale

    epsilon = state.epsilon.copy()
    pressure = state.pressure.copy()
    enthalpy = state.enthalpy.copy()
    epsilon[:-1] = epsilon_cell
    pressure[:-1] = pressure_cell
    enthalpy[:-1] = enthalpy_cell
    epsilon[-1] = epsilon[-2]
    pressure[-1] = pressure[-2]
    enthalpy[-1] = enthalpy[-2]

    filtered = State(
        u=state.u.copy(),
        r=state.r.copy(),
        rho=state.rho.copy(),
        epsilon=epsilon,
        pressure=pressure,
        enthalpy=enthalpy,
        ephi=state.ephi.copy(),
        mass=state.mass.copy(),
        gamma_lorentz=state.gamma_lorentz.copy(),
        e_a=np.zeros_like(state.e_a),
        q=state.q.copy(),
    )
    filtered = iterate_mass_gamma(filtered, a_face, params, scales)
    filtered.e_a = compute_e_a(filtered.q, filtered.r, filtered.rho[:-1])
    return filtered


def compute_momentum_acceleration(
    state: State,
    a_face: np.ndarray,
    e_a_old: np.ndarray,
    e_a_new: np.ndarray,
    dt_code: float,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls | None = None,
) -> np.ndarray:
    rho_cell = state.rho[:-1]
    pressure_cell = effective_momentum_pressure(state, scales, controls)
    enthalpy_cell = state.enthalpy[:-1]
    rho_face = avg_cell_to_face(rho_cell)
    pressure_face = avg_cell_to_face(pressure_cell)

    dphi_da = compute_phi_gradient(
        pressure_cell=pressure_cell,
        rho_cell=rho_cell,
        enthalpy_cell=enthalpy_cell,
        ephi_face=state.ephi,
        e_a_old=e_a_old,
        e_a_new=e_a_new,
        a_face=a_face,
        dt_code=dt_code,
        params=params,
        scales=scales,
    )

    du_dt = np.zeros_like(state.u)
    term1 = (
        -(state.gamma_lorentz[1:] ** A1_GAMMA_POWER)
        * dphi_da[1:]
        * 4.0
        * np.pi
        * state.r[1:] ** 2
        * rho_face[1:]
        / np.maximum(state.ephi[1:], TINY)
    )
    gravity_mass = total_gravity_mass(state, params)
    term2 = gravity_mass[1:] / np.maximum(state.r[1:] ** 2, TINY)
    term3 = 4.0 * np.pi * pressure_face[1:] * state.r[1:] / scales.r0_scale
    du_dt[1:] = -state.ephi[1:] * (term1 + term2 + term3) / scales.r0_scale
    return du_dt


def advance_radius_from_velocity(
    state: State,
    u_face: np.ndarray,
    dt_code: float,
    scales: Scales,
    controls: StepControls,
) -> tuple[np.ndarray, np.ndarray]:
    u_new = np.ascontiguousarray(u_face.copy())
    u_new[0] = 0.0
    u_new = apply_outer_velocity_boundary(u_new, controls)

    r_new = state.r + state.ephi * u_new * dt_code / scales.r0_scale**2
    r_new[0] = 0.0
    if controls.radius_repair_enabled:
        old_width = np.diff(state.r)
        min_allowed = r_new[:-1] + controls.min_shell_width_fraction * old_width
        if np.any(r_new[1:] <= min_allowed):
            for i in range(1, len(r_new)):
                min_allowed_i = (
                    r_new[i - 1] + controls.min_shell_width_fraction * old_width[i - 1]
                )
                if r_new[i] <= min_allowed_i:
                    r_new[i] = min_allowed_i
                    u_new[i] = (r_new[i] - state.r[i]) * scales.r0_scale**2 / (
                        max(state.ephi[i], TINY) * dt_code
                    )
    r_new, u_new = regularize_shell_widths(r_new, state, u_new, dt_code, scales, controls)
    return r_new, u_new


def advance_radius_heun(
    state: State,
    predictor_state: State,
    u_predict: np.ndarray,
    u_corrected: np.ndarray,
    dt_code: float,
    scales: Scales,
    controls: StepControls,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance radius with the Heun average of the coordinate velocity."""
    u_new = np.ascontiguousarray(u_corrected.copy())
    u_new[0] = 0.0
    u_new = apply_outer_velocity_boundary(u_new, controls)

    u_predict_bounded = np.ascontiguousarray(u_predict.copy())
    u_predict_bounded[0] = 0.0
    u_predict_bounded = apply_outer_velocity_boundary(u_predict_bounded, controls)
    coordinate_velocity_old = state.ephi * state.u
    coordinate_velocity_predict = predictor_state.ephi * u_predict_bounded
    r_new = state.r + 0.5 * (
        coordinate_velocity_old + coordinate_velocity_predict
    ) * dt_code / scales.r0_scale**2
    r_new[0] = 0.0

    if controls.radius_repair_enabled:
        old_width = np.diff(state.r)
        min_allowed = r_new[:-1] + controls.min_shell_width_fraction * old_width
        if np.any(r_new[1:] <= min_allowed):
            for i in range(1, len(r_new)):
                min_allowed_i = (
                    r_new[i - 1] + controls.min_shell_width_fraction * old_width[i - 1]
                )
                if r_new[i] <= min_allowed_i:
                    r_new[i] = min_allowed_i

    r_new, u_new = regularize_shell_widths(r_new, state, u_new, dt_code, scales, controls)
    return r_new, u_new


def regularize_shell_widths(
    r_new: np.ndarray,
    previous: State,
    u_new: np.ndarray,
    dt_code: float,
    scales: Scales,
    controls: StepControls,
) -> tuple[np.ndarray, np.ndarray]:
    trigger_ratio = max(0.0, float(controls.shell_width_regularization_ratio))
    strength = min(max(float(controls.shell_width_regularization_strength), 0.0), 1.0)
    if trigger_ratio <= 0.0 or strength <= 0.0 or r_new.size < 5:
        return r_new, u_new

    widths = np.diff(r_new)
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        return r_new, u_new

    adjacent_ratio = np.minimum(
        widths[:-1] / np.maximum(widths[1:], TINY),
        widths[1:] / np.maximum(widths[:-1], TINY),
    )
    bad_pairs = np.where(adjacent_ratio < trigger_ratio)[0]
    if bad_pairs.size == 0:
        return r_new, u_new

    target = smooth_interior(widths, max(1, int(controls.shell_width_regularization_passes)))
    if not np.all(np.isfinite(target)) or np.any(target <= 0.0):
        return r_new, u_new

    active = np.zeros_like(widths, dtype=bool)
    pad = max(0, int(controls.shell_width_regularization_pad))
    for idx in bad_pairs:
        lo = max(0, idx - pad)
        hi = min(widths.size, idx + 2 + pad)
        active[lo:hi] = True

    new_widths = widths.copy()
    i = 0
    while i < active.size:
        if not active[i]:
            i += 1
            continue
        start = i
        while i < active.size and active[i]:
            i += 1
        stop = i
        old_sum = float(np.sum(widths[start:stop]))
        target_sum = float(np.sum(target[start:stop]))
        if old_sum <= 0.0 or target_sum <= 0.0:
            continue
        segment_target = target[start:stop] * (old_sum / target_sum)
        new_widths[start:stop] = widths[start:stop] + strength * (
            segment_target - widths[start:stop]
        )

    if not np.all(np.isfinite(new_widths)) or np.any(new_widths <= 0.0):
        return r_new, u_new

    regularized = r_new.copy()
    regularized[0] = 0.0
    regularized[1:] = np.cumsum(new_widths)
    if not np.all(np.diff(regularized) > 0.0):
        return r_new, u_new

    u_regularized = u_new.copy()
    u_regularized[0] = 0.0
    if dt_code > 0.0:
        u_regularized[1:] = (
            (regularized[1:] - previous.r[1:])
            * scales.r0_scale**2
            / (np.maximum(previous.ephi[1:], TINY) * dt_code)
        )
    u_regularized = apply_outer_velocity_boundary(u_regularized, controls)
    return regularized, u_regularized


def apply_post_step_smoothing(
    state: State,
    a_face: np.ndarray,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls,
) -> State:
    passes = max(0, int(controls.post_step_smoothing_passes))
    if passes == 0:
        return apply_epsilon_filter(state, a_face, params, scales, controls)

    u = smooth_interior(state.u, passes)
    u[0] = 0.0
    u = apply_outer_velocity_boundary(u, controls)

    q = smooth_interior(state.q, passes)

    if not controls.smooth_thermodynamic_state:
        rho_cell = state.rho[:-1].copy()
        epsilon_cell = state.epsilon[:-1].copy()
        pressure_factor = np.ones_like(rho_cell)
        if params.pressure_factor_enabled:
            pressure_factor = compute_pressure_factor(
                u_right=u[1:],
                shell_width=np.diff(state.r),
                rho_cell=rho_cell,
                epsilon_cell=epsilon_cell,
                params=params,
                scales=scales,
                u_left=u[:-1],
                r_left=state.r[:-1],
                r_right=state.r[1:],
            )
        pressure_cell = (params.gamma - 1.0) * rho_cell * epsilon_cell * pressure_factor
        enthalpy_cell = 1.0 + (
            epsilon_cell + pressure_cell / np.maximum(rho_cell, TINY)
        ) / scales.r0_scale

        pressure = state.pressure.copy()
        enthalpy = state.enthalpy.copy()
        pressure[:-1] = pressure_cell
        enthalpy[:-1] = enthalpy_cell
        pressure[-1] = pressure[-2]
        enthalpy[-1] = enthalpy[-2]

        smoothed = State(
            u=u,
            r=state.r.copy(),
            rho=state.rho.copy(),
            epsilon=state.epsilon.copy(),
            pressure=pressure,
            enthalpy=enthalpy,
            ephi=state.ephi.copy(),
            mass=state.mass.copy(),
            gamma_lorentz=state.gamma_lorentz.copy(),
            e_a=np.zeros_like(state.e_a),
            q=q,
        )
        smoothed = iterate_mass_gamma(smoothed, a_face, params, scales)
        smoothed.e_a = compute_e_a(smoothed.q, smoothed.r, smoothed.rho[:-1])
        return apply_epsilon_filter(smoothed, a_face, params, scales, controls)

    r = smooth_interior(state.r, passes)
    r[0] = 0.0
    for i in range(1, len(r)):
        if r[i] <= r[i - 1]:
            r[i] = r[i - 1] + max(TINY, 1e-12 * max(1.0, abs(r[i - 1])))

    rho_cell = np.maximum(smooth_interior(state.rho[:-1], passes), 1e-14)
    epsilon_cell = np.maximum(smooth_interior(state.epsilon[:-1], passes), 1e-14)
    ephi = np.maximum(smooth_interior(state.ephi, passes), TINY)
    ephi[-1] = params.ephib

    pressure_factor = np.ones_like(rho_cell)
    if params.pressure_factor_enabled:
        pressure_factor = compute_pressure_factor(
            u_right=u[1:],
            shell_width=np.diff(r),
            rho_cell=rho_cell,
            epsilon_cell=epsilon_cell,
            params=params,
            scales=scales,
            u_left=u[:-1],
            r_left=r[:-1],
            r_right=r[1:],
        )
    pressure_cell = (params.gamma - 1.0) * rho_cell * epsilon_cell * pressure_factor
    enthalpy_cell = 1.0 + (
        epsilon_cell + pressure_cell / np.maximum(rho_cell, TINY)
    ) / scales.r0_scale

    rho_full = np.empty_like(state.rho)
    epsilon_full = np.empty_like(state.epsilon)
    pressure_full = np.empty_like(state.pressure)
    enthalpy_full = np.empty_like(state.enthalpy)
    rho_full[:-1] = rho_cell
    epsilon_full[:-1] = epsilon_cell
    pressure_full[:-1] = pressure_cell
    enthalpy_full[:-1] = enthalpy_cell
    rho_full[-1] = rho_full[-2]
    epsilon_full[-1] = epsilon_full[-2]
    pressure_full[-1] = pressure_full[-2]
    enthalpy_full[-1] = enthalpy_full[-2]

    smoothed = State(
        u=u,
        r=r,
        rho=rho_full,
        epsilon=epsilon_full,
        pressure=pressure_full,
        enthalpy=enthalpy_full,
        ephi=ephi,
        mass=state.mass.copy(),
        gamma_lorentz=state.gamma_lorentz.copy(),
        e_a=np.zeros_like(state.e_a),
        q=q,
    )
    smoothed = iterate_mass_gamma(smoothed, a_face, params, scales)
    smoothed.e_a = compute_e_a(smoothed.q, smoothed.r, smoothed.rho[:-1])
    return apply_epsilon_filter(smoothed, a_face, params, scales, controls)


def predict_step_heat_update(
    state: State,
    a_face: np.ndarray,
    dt_t0: float,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls,
) -> tuple[np.ndarray, np.ndarray]:
    dt_code = dt_t0 * scales.dt_code_per_t0
    if controls.implicit_heat_enabled:
        q_predict, delta_epsilon_heat = implicit_heat_predictor(
            state,
            a_face,
            dt_code,
            params,
            scales,
            controls,
        )
    else:
        q_predict = compute_predicted_heat_flux(state, a_face, params, scales, controls)
        delta_epsilon_heat = heat_energy_delta(q_predict, state, a_face, dt_code, scales)
    if controls.heat_face_contrast_fraction > 0.0:
        q_predict = apply_heat_face_contrast_limiter(
            q_predict,
            state,
            a_face,
            dt_code,
            scales,
            controls,
        )
        delta_epsilon_heat = heat_energy_delta(q_predict, state, a_face, dt_code, scales)
    return apply_heat_delta_limiter(
        q_predict,
        delta_epsilon_heat,
        state,
        controls,
    )


def evolve_one_step(
    state: State,
    a_face: np.ndarray,
    dt_t0: float,
    params: PhysicalParams,
    scales: Scales,
    controls: StepControls,
    dynamic_face_start: int = 1,
    dynamic_face_end: int | None = None,
    q_new_override: np.ndarray | None = None,
    delta_eps_heat_override: np.ndarray | None = None,
) -> State:
    dynamic_start = min(max(1, int(dynamic_face_start)), state.r.size)
    dynamic_end = (
        state.r.size - 1
        if dynamic_face_end is None
        else min(max(dynamic_start, int(dynamic_face_end)), state.r.size - 1)
    )
    motion_state = state.copy()
    motion_state.u[:dynamic_start] = 0.0
    motion_state.u[dynamic_end + 1 :] = 0.0
    dt_code = dt_t0 * scales.dt_code_per_t0
    if (q_new_override is None) != (delta_eps_heat_override is None):
        raise ValueError(
            "q_new_override and delta_eps_heat_override must be supplied together"
        )
    if q_new_override is None:
        q_predict, delta_epsilon_heat = predict_step_heat_update(
            state,
            a_face,
            dt_t0,
            params,
            scales,
            controls,
        )
    else:
        q_predict = np.ascontiguousarray(q_new_override, dtype=np.float64).copy()
        delta_epsilon_heat = np.ascontiguousarray(
            delta_eps_heat_override,
            dtype=np.float64,
        ).copy()
        if q_predict.shape != state.q.shape:
            raise ValueError("q_new_override must match the state heat-flux shape")
        if delta_epsilon_heat.shape != state.epsilon[:-1].shape:
            raise ValueError(
                "delta_eps_heat_override must match the physical-cell shape"
            )
    e_a_predict = compute_e_a(q_predict, state.r, state.rho[:-1])

    def assemble_advanced_state(u_new: np.ndarray, r_new: np.ndarray) -> State:
        state_for_gamma = state.copy()
        state_for_gamma.u = u_new
        state_for_gamma.r = r_new
        gamma_new = update_gamma(state_for_gamma, scales)
        gamma_cell = avg_face_to_cell(gamma_new)

        shell_volume = (4.0 * np.pi / 3.0) * (r_new[1:] ** 3 - r_new[:-1] ** 3)
        shell_mass = gamma_cell * np.diff(a_face)
        rho_new_cell = shell_mass / np.maximum(shell_volume, TINY)

        vol_old = 1.0 / np.maximum(state.rho[:-1], TINY)
        vol_new = 1.0 / np.maximum(rho_new_cell, TINY)
        d_vol = vol_new - vol_old
        denom = 1.0 + 0.5 * (params.gamma - 1.0) * rho_new_cell * d_vol
        numerator = state.epsilon[:-1] + delta_epsilon_heat - 0.5 * state.pressure[:-1] * d_vol
        energy_mode = str(controls.energy_update_mode).lower().strip()
        if energy_mode in ("exact-adiabatic", "exact_adiabatic", "exact_adibatic", "exact"):
            epsilon_after_heat = np.maximum(state.epsilon[:-1] + delta_epsilon_heat, params.epsilon_floor)
            compression = np.maximum(rho_new_cell / np.maximum(state.rho[:-1], TINY), TINY)
            epsilon_new_cell = epsilon_after_heat * compression ** (params.gamma - 1.0)
        elif energy_mode in ("hybrid-exact", "hybrid_exact", "hybrid"):
            trapezoid = numerator / np.maximum(denom, params.energy_denominator_floor)
            epsilon_after_heat = np.maximum(state.epsilon[:-1] + delta_epsilon_heat, params.epsilon_floor)
            compression = np.maximum(rho_new_cell / np.maximum(state.rho[:-1], TINY), TINY)
            exact = epsilon_after_heat * compression ** (params.gamma - 1.0)
            switch = (
                (~np.isfinite(trapezoid))
                | (denom <= max(float(controls.energy_update_denominator_switch), params.energy_denominator_floor))
                | (trapezoid <= 0.0)
            )
            epsilon_new_cell = np.where(switch, exact, trapezoid)
        elif energy_mode == "trapezoid":
            epsilon_new_cell = numerator / np.maximum(denom, params.energy_denominator_floor)
        else:
            raise ValueError(f"unknown energy update mode: {controls.energy_update_mode}")
        epsilon_new_cell = np.maximum(epsilon_new_cell, params.epsilon_floor)

        p_fac = np.ones_like(rho_new_cell)
        if params.pressure_factor_enabled:
            p_fac = compute_pressure_factor(
                u_right=u_new[1:],
                shell_width=np.diff(r_new),
                rho_cell=rho_new_cell,
                epsilon_cell=epsilon_new_cell,
                params=params,
                scales=scales,
                u_left=u_new[:-1],
                r_left=r_new[:-1],
                r_right=r_new[1:],
            )
        pressure_new_cell = (params.gamma - 1.0) * rho_new_cell * epsilon_new_cell * p_fac
        enthalpy_new_cell = 1.0 + (
            epsilon_new_cell + pressure_new_cell / np.maximum(rho_new_cell, TINY)
        ) / scales.r0_scale

        if params.ephi_evolution_enabled:
            dphi_da_new = compute_phi_gradient(
                pressure_cell=pressure_new_cell,
                rho_cell=rho_new_cell,
                enthalpy_cell=enthalpy_new_cell,
                ephi_face=state.ephi,
                e_a_old=state.e_a,
                e_a_new=e_a_predict,
                a_face=a_face,
                dt_code=dt_code,
                params=params,
                scales=scales,
            )
            ephi_new = integrate_ephi(dphi_da_new, a_face, params, scales)
        else:
            ephi_new = state.ephi.copy()

        q_new = q_predict
        # Store e_A on the same geometry as the advanced state.  The original
        # kernel finalizes eA after updating R and rho; keeping the predictor
        # geometry here makes the next (e_A^n - e_A^{n-1}) / dt term spuriously
        # include a geometric mismatch.
        e_a_new = compute_e_a(q_new, r_new, rho_new_cell)

        rho_full = np.empty_like(state.rho)
        epsilon_full = np.empty_like(state.epsilon)
        pressure_full = np.empty_like(state.pressure)
        enthalpy_full = np.empty_like(state.enthalpy)
        rho_full[:-1] = rho_new_cell
        epsilon_full[:-1] = epsilon_new_cell
        pressure_full[:-1] = pressure_new_cell
        enthalpy_full[:-1] = enthalpy_new_cell
        rho_full[-1] = rho_full[-2]
        epsilon_full[-1] = epsilon_full[-2]
        pressure_full[-1] = pressure_full[-2]
        enthalpy_full[-1] = enthalpy_full[-2]

        advanced = State(
            u=u_new,
            r=r_new,
            rho=rho_full,
            epsilon=epsilon_full,
            pressure=pressure_full,
            enthalpy=enthalpy_full,
            ephi=ephi_new,
            mass=state.mass.copy(),
            gamma_lorentz=gamma_new,
            e_a=e_a_new,
            q=q_new,
        )
        advanced = iterate_mass_gamma(advanced, a_face, params, scales)
        return advanced

    du_dt_old = compute_momentum_acceleration(
        state,
        a_face,
        e_a_old=state.e_a,
        e_a_new=e_a_predict,
        dt_code=dt_code,
        params=params,
        scales=scales,
        controls=controls,
    )
    du_dt_old[:dynamic_start] = 0.0
    du_dt_old[dynamic_end + 1 :] = 0.0
    momentum_mode = str(controls.momentum_update_mode).lower().strip()
    if momentum_mode == "euler":
        u_predict = motion_state.u + du_dt_old * dt_code
        u_predict[dynamic_end + 1 :] = 0.0
        r_predict, u_predict = advance_radius_from_velocity(
            motion_state, u_predict, dt_code, scales, controls
        )
        next_state = assemble_advanced_state(u_predict, r_predict)
    elif momentum_mode in ("heun", "predictor-corrector", "predictor_corrector"):
        u_predict = motion_state.u + du_dt_old * dt_code
        u_predict[0] = 0.0
        u_predict[dynamic_end + 1 :] = 0.0
        u_predict = apply_outer_velocity_boundary(u_predict, controls)
        r_predict, _ = advance_radius_from_velocity(
            motion_state, motion_state.u, dt_code, scales, controls
        )
        predictor_state = assemble_advanced_state(u_predict, r_predict)
        du_dt_predict = compute_momentum_acceleration(
            predictor_state,
            a_face,
            e_a_old=state.e_a,
            e_a_new=predictor_state.e_a,
            dt_code=dt_code,
            params=params,
            scales=scales,
            controls=controls,
        )
        du_dt_predict[:dynamic_start] = 0.0
        du_dt_predict[dynamic_end + 1 :] = 0.0
        u_corrected = motion_state.u + 0.5 * (du_dt_old + du_dt_predict) * dt_code
        u_corrected[dynamic_end + 1 :] = 0.0
        r_corrected, u_corrected = advance_radius_heun(
            motion_state,
            predictor_state,
            u_predict,
            u_corrected,
            dt_code,
            scales,
            controls,
        )
        next_state = assemble_advanced_state(u_corrected, r_corrected)
    else:
        raise ValueError(f"unknown momentum update mode: {controls.momentum_update_mode}")

    next_state = apply_post_step_smoothing(next_state, a_face, params, scales, controls)
    return next_state


def horizon_metric(state: State, scales: Scales) -> Tuple[float, int]:
    compactness = np.zeros_like(state.r)
    valid = state.r > 0.0
    compactness[valid] = 2.0 * state.mass[valid] / np.maximum(state.r[valid] * scales.r0_scale, TINY)
    idx = int(np.argmax(compactness))
    return float(compactness[idx]), idx


def total_compactness_metric(state: State, params: PhysicalParams, scales: Scales) -> Tuple[float, int]:
    compactness = np.zeros_like(state.r)
    valid = state.r > 0.0
    mass_total = total_gravity_mass(state, params)
    compactness[valid] = 2.0 * mass_total[valid] / np.maximum(state.r[valid] * scales.r0_scale, TINY)
    idx = int(np.argmax(compactness))
    return float(compactness[idx]), idx


def check_state(state: State, controls: StepControls, scales: Scales) -> Tuple[bool, str]:
    del scales
    arrays: Dict[str, np.ndarray] = {
        "u": state.u,
        "r": state.r,
        "rho": state.rho,
        "epsilon": state.epsilon,
        "pressure": state.pressure,
        "enthalpy": state.enthalpy,
        "ephi": state.ephi,
        "mass": state.mass,
        "gamma": state.gamma_lorentz,
        "e_a": state.e_a,
        "q": state.q,
    }
    for name, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            return False, f"{name} contains non-finite values"
    if state.r[0] != 0.0:
        return False, "central radius is not zero"
    if not np.all(np.diff(state.r) > 0.0):
        return False, "radius grid is not strictly increasing"
    if not np.all(state.rho > 0.0):
        return False, "rho is not positive"
    if not np.all(state.epsilon > 0.0):
        return False, "epsilon is not positive"
    if not np.all(state.pressure >= 0.0):
        return False, "pressure is negative"
    if not np.all(state.ephi > 0.0):
        return False, "ephi is not positive"
    if np.max(np.abs(state.u)) > controls.max_abs_u_code:
        idx = int(np.argmax(np.abs(state.u)))
        return False, f"velocity exceeded configured bound at i={idx}, value={state.u[idx]:.6e}"
    if np.max(state.epsilon) > controls.max_epsilon_code:
        idx = int(np.argmax(state.epsilon))
        return False, f"epsilon exceeded configured bound at i={idx}, value={state.epsilon[idx]:.6e}"
    if np.max(state.rho) > controls.max_density_code:
        idx = int(np.argmax(state.rho))
        return False, f"density exceeded configured bound at i={idx}, value={state.rho[idx]:.6e}"
    if np.max(state.mass) > controls.max_mass_code:
        idx = int(np.argmax(state.mass))
        return False, f"mass exceeded configured bound at i={idx}, value={state.mass[idx]:.6e}"
    if abs(state.mass[0]) > 1e-14:
        return False, "central mass boundary is not zero"
    if np.any(np.diff(state.mass) < -1e-12):
        return False, "mass is not monotonic"
    return True, "ok"


def check_step_transition(previous: State, trial: State, controls: StepControls) -> Tuple[bool, str]:
    old_width = np.diff(previous.r)
    new_width = np.diff(trial.r)
    width_ratio = new_width / np.maximum(old_width, TINY)
    min_width_ratio = float(np.min(width_ratio))
    if controls.min_shell_width_ratio > 0.0 and min_width_ratio < controls.min_shell_width_ratio:
        idx = int(np.argmin(width_ratio))
        return (
            False,
            "shell width ratio too small: "
            f"{min_width_ratio:.3e} at i={idx}, "
            f"old_width={old_width[idx]:.6e}, new_width={new_width[idx]:.6e}",
        )

    if controls.max_fractional_epsilon_change > 0.0:
        eps_rel = np.abs(trial.epsilon[:-1] - previous.epsilon[:-1]) / np.maximum(
            previous.epsilon[:-1],
            TINY,
        )
        max_eps_rel = float(np.max(eps_rel))
        if max_eps_rel > controls.max_fractional_epsilon_change:
            idx = int(np.argmax(eps_rel))
            return (
                False,
                "epsilon changed too much in one step: "
                f"{max_eps_rel:.3e} at i={idx}, "
                f"old={previous.epsilon[idx]:.6e}, new={trial.epsilon[idx]:.6e}",
            )

    if controls.max_fractional_density_change > 0.0:
        rho_rel = np.abs(trial.rho[:-1] - previous.rho[:-1]) / np.maximum(previous.rho[:-1], TINY)
        max_rho_rel = float(np.max(rho_rel))
        if max_rho_rel > controls.max_fractional_density_change:
            idx = int(np.argmax(rho_rel))
            return (
                False,
                "density changed too much in one step: "
                f"{max_rho_rel:.3e} at i={idx}, "
                f"old={previous.rho[idx]:.6e}, new={trial.rho[idx]:.6e}",
            )

    return True, "ok"
