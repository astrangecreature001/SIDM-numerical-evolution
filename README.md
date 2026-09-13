# SIDM numerical evolution

Lagrangian evolution of self-interacting dark matter, with static baryon profiles, explicit RKL2 heat transport, and quasistatic HSE core evolution.

## Environment

Use Linux and Python 3.12. The case runner uses the Unix `fcntl` module.

```bash
python -m pip install -r requirements.txt
```

## 1. Prepare non-HSE initial conditions

`non_hse_prd_rkl2/parameter_anchors.json` is the input for 61 halo/baryon configurations. It records physical parameters, numerical settings, and the meaning and calibration range of the halo quantities.

Supply these input arrays under `non_hse_prd_rkl2/reference/`:

- `A.npy`: the reference cumulative rest-mass grid.
- `initial.npy`: the compatible reference state, with rows ordered as `U, R, rho, epsilon, P, w, exp(phi), m, Gamma, eA, q`.

These arrays are external inputs and are not included in this repository. The preparation step conservatively remaps the reference mass-volume relation and solves a discrete equilibrium for each configuration.

```bash
cd non_hse_prd_rkl2
python prepare.py
```

Outputs include `prepared/<case>/A.npy`, `initial.npy`, `run_config.json`, and an initial-equilibrium check. The preparation step also generates `run_manifest.json` and `source_hashes.json`. Use a fresh working directory; existing prepared cases are protected from overwrite.

## 2. Run non-HSE evolution

```bash
python run_case.py --case PRD_Z00_FENG_P01
```

The runner checks input/source hashes, advances with the acoustic CFL timestep and RKL2 heat transport, and saves results under `runs/<case>/`. The `run_case_peak01.py` entry invokes the same driver.

The sampled SMFP mass peak is confirmed after at least eight lower post-peak samples, at least `0.1` in native time beyond the peak, native time at least `10`, and central density above `1.1` times its recorded minimum. Cases with `post_peak_continue=true` continue after confirmation. This is a finite-sampling stopping condition, not a guarantee of the global future maximum.

To continue a compatible saved run:

```bash
python run_case.py --case PRD_Z00_FENG_P01 --resume
```

`main.py` also provides state validation, timestep utilities, and a standalone single-case entry. Its `config.example.txt` and `parameter.example.txt` templates are separate from the batch parameter input.

## 3. Prepare an HSE checkpoint and grid

`hse_to_dynamic/` uses its own state format and physical scales. Supply a compatible grid directory containing `A.npy` and a checkpoint directory containing `checkpoint_state.npy` and `checkpoint_meta.json`. Non-HSE batch snapshots are not direct inputs to this runner; matching the physical parameters alone does not establish checkpoint compatibility.

For a compatible no-baryon checkpoint, `tools/build_fine_hse_switch_seed.py` builds a conservatively remapped fine grid. `tools/build_hse_core_refined_seed.py` refines only the HSE core and specifically requires a checkpoint at `tau=480`. Use each tool's `--help` for its required inputs. Grid and checkpoint arrays are external inputs.

## 4. Run HSE evolution and the dynamical transition

From the repository root:

```bash
cd hse_to_dynamic
python run_hse_sublevel_then_dynamic.py \
  --grid-dir /path/to/compatible_grid \
  --resume-from /path/to/compatible_checkpoint \
  --output-dir /path/to/output \
  --gamma-a1-power 1
```

The default HSE path evolves the core with a frozen exterior, applies convergence and density checks, and subdivides rejected intervals. When the required interval falls below the configured switch threshold, it reconstructs velocities from the last two accepted HSE states, rebuilds metric variables, and starts dynamical continuation.

`hybrid_hse.py`, `sidm_engine.py`, `run_hse_late.py`, `tools/diagnose_hse_direct_solver.py`, `display_diagnostics.py`, and `core_dynamic.py` supply imported computational functions. Keep them with the entry point. Additional command-line controls are listed by `--help`.

## Units and verification

- The non-HSE `tau_native` uses the approximation defined in `prepare.ctx`. Exact physical-reference scales are recorded separately by `physics.Units`; account for this distinction when converting times or combining results.
- Grid, state ordering, units, physical parameters, and source hashes must match when resuming a run. Source changes require a fresh preparation or an explicit compatibility review.
- `SOURCE_MANIFEST.sha256` lists the current tracked files, excluding the manifest itself. On Linux, verify with `sha256sum -c SOURCE_MANIFEST.sha256`.
- Syntax and structural checks do not establish numerical convergence or physical validity for a new run.

## Attribution

Derived from [Hua-Peng-G/SIDM](https://github.com/Hua-Peng-G/SIDM). The upstream MIT license and copyright are retained in `LICENSE`. Cite the upstream repository and associated research when using the code.
