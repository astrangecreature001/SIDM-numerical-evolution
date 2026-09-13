"""One-way dynamical continuation of the active Lagrangian core.

The first frozen exterior cell supplies the pressure boundary.  Only faces
1..core_face_end and cells 0..core_face_end-1 evolve; the exterior remains the
same reservoir used by the frozen-exterior HSE branch.
"""

from __future__ import annotations

from dataclasses import replace

from sidm_engine import (
    PhysicalParams,
    State,
    StepControls,
    compute_e_a,
    evolve_one_step,
    update_gamma,
)


def evolve_core_dynamic_one_step(
    state: State,
    a_face,
    dt_t0: float,
    params: PhysicalParams,
    scales,
    controls: StepControls,
    *,
    core_face_end: int,
    monotone_density_radius_limit_rs: float | None = None,
) -> State:
    """Advance the active core dynamically while retaining one exterior ghost."""

    n = int(core_face_end)
    if not 3 <= n < state.r.size - 1:
        raise ValueError("core dynamic continuation requires one exterior ghost cell")

    stop = n + 2
    sub = State(
        **{
            name: getattr(state, name)[:stop].copy()
            for name in state.__dataclass_fields__
        }
    )
    for name in ("rho", "epsilon", "pressure", "enthalpy"):
        values = getattr(sub, name)
        values[-1] = values[-2]
    sub.u[n + 1] = 0.0
    sub.q[n + 1] = 0.0

    boundary_params = replace(params, ephib=float(state.ephi[n + 1]))
    advanced = evolve_one_step(
        sub,
        a_face[:stop],
        dt_t0,
        boundary_params,
        scales,
        controls,
        dynamic_face_start=1,
        dynamic_face_end=n,
        monotone_density_radius_limit_rs=monotone_density_radius_limit_rs,
    )

    merged = state.copy()
    active_faces = slice(0, n + 1)
    active_cells = slice(0, n)
    for name in ("u", "r", "ephi", "mass", "gamma_lorentz", "e_a", "q"):
        getattr(merged, name)[active_faces] = getattr(advanced, name)[active_faces]
    for name in ("rho", "epsilon", "pressure", "enthalpy"):
        getattr(merged, name)[active_cells] = getattr(advanced, name)[active_cells]

    radius_shift = float(advanced.r[n] - state.r[n])
    mass_shift = float(advanced.mass[n] - state.mass[n])
    merged.r[n + 1 :] = state.r[n + 1 :] + radius_shift
    merged.mass[n + 1 :] = state.mass[n + 1 :] + mass_shift
    merged.u[n + 1 :] = 0.0
    merged.gamma_lorentz = update_gamma(merged, scales)
    merged.e_a = compute_e_a(merged.q, merged.r, merged.rho[:-1])
    return merged
