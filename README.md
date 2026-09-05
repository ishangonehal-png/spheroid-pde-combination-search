# Spheroid Reaction-Diffusion Evaluation of Combination Bayesian Optimization

Code and data for **"PDE Reaction-Diffusion Spheroid Modeling for Probabilistic
Chemotherapy Drug Search"** (Simbiochem @ NeurIPS 2026).

The question: combination drug-search methods are scored by a single number
measured in a well-mixed monolayer assay. Does that number still rank
combinations correctly once spatial structure (drug transport, oxygen gradients,
hypoxic resistance) is modeled? Answer: no. The published well-mixed ODE bridge
is exactly rank-preserving (Spearman rho = -1.0000), while the spheroid PDE
re-orders 36.5% of within-cell-line pairs.

## Quick start

```bash
conda env create -f environment.yml && conda activate savitar-pde
# or: pip install -r requirements.txt
python tests/test_reproduce.py      # ~30 s, 11 checks
```

`tests/test_reproduce.py` verifies the solver against analytic references and
re-runs a slice of the simulation, checking it reproduces the shipped results
**bit-exactly** (max|diff| = 0.000e+00). If that passes, the environment is
correct.

## Layout

```
src/spheroid_rd.py           the PDE solver (fields, Laplacian, integrator)
src/spheroid_readouts.py     outcome-anchored readouts from a solved trajectory
src/volume_render.py         CPU volume ray-caster for the 3-D figure
scripts/33_*                 recover BO-selected combinations (needs upstream repo)
scripts/34_simulate_*        simulate every combination            -> results/pde_readouts*.csv
scripts/35_*_sensitivity     13-setting parameter sweep            -> results/pde_sensitivity*.csv
scripts/36_analyze_*         re-ranking / monotonicity statistics  -> results/cohort_*.csv
scripts/37_render_*          3-D spheroid renders
data/                        inputs sufficient to re-run the simulation
results/                     published output tables
figures/                     paper figures
paper/                       manuscript source and PDF
docs/                        literature survey linking readouts to patient outcome
```

## The model

Viable cell density `n(r,t)`, oxygen `c_O(r,t)`, and one concentration `c_j(r,t)`
per drug on `r` in `[0, R]`:

    dn/dt = D_n * lap(n) + rho * phi(c_O) * n * (1 - n) - sum_j gamma_j * psi_j(c_j) * n

with `lap` the spherically symmetric Laplacian `n_rr + (2/r) n_r`, using the
l'Hopital limit `3 n_rr` at the origin. Method of lines with LSODA; state is
interleaved **by node** rather than by field, making the Jacobian banded with
half-bandwidth `2*n_f - 1`. That single change is a 100x speedup with bitwise
identical output (298 s -> 2.99 s on a 2-drug run).

Defaults: `R = 400 um`, oxygen consumption `q_O = 15 /day` (calibrated so the
viable rim lands at ~170 um, inside the 150-200 um oxygen-diffusion limit),
21-day horizon, 5 daily doses.

### Solver verification

`spheroid_rd.verify_solver()` checks three things, all run by the test suite:

| check | result |
|---|---|
| oxygen profile vs analytic steady state | rel L2 = 5.5e-3 |
| Fisher-KPP front speed vs `2*sqrt(D*rho)` | 169.2 vs 178.9, 5.4% error |
| mesh convergence (80/160/320 nodes) | ratio 3.30 |

## Readouts

Readouts are chosen because they have **published correlations with patient
outcome**, not because they are convenient. See `docs/pde_outcome_metrics.md`
(58 primary papers, 55 DOI-verified references).

The load-bearing finding from that survey: the tumor **growth** rate constant
`K_G` predicts overall survival (r = -0.72), while the **kill** rate constant
`K_S` does not (r = -0.218). A monolayer assay measures something like `K_S`.

Two readouts are reported as **not established**: `core_sparing` inverts under
4 of 12 perturbations, and `log_kill_nadir` is the least stable (median rho =
0.684). They are mechanistic diagnostics, not ranking quantities.

## What is and is not reproducible here

**Reproducible from this repo alone** (verified, bit-exact):
- solver verification suite
- the spheroid simulation, from `data/almanac_hill_fits.parquet` +
  `data/bo_selected_combinations.csv`
- all downstream statistics and figures

**Requires the upstream Savitar repo** (not included here):
- `scripts/33_run_almanac_selection_recovery.py`, which re-runs Bayesian
  optimization to recover which combination each method selected. It needs the
  BO implementation, `ComboCompoundSet.sdf`, and the 615 MB raw
  `ComboDrugGrowth_Nov2017.csv`. Its **output is shipped** as
  `data/bo_selected_combinations.csv`, so the simulation does not need it.

**Known upstream gaps** (documented, not fixed here): every published NCI
download URL for the raw ALMANAC file now returns 403/404, and the upstream BO
traces record regret without recording the selected combination, which is why
the recovery script exists at all.

## Determinism

The simulation is deterministic given the same scipy version. `scipy==1.17.1` is
pinned deliberately: the LSODA banded path and `curve_fit` basin selection both
shift across major versions (a prior audit found 13 of 6,293 Hill fits flip
basin under a version change). Parallelism uses `multiprocessing` with `fork`
and does not affect results.

## Citation

Predecessor paper (Savitar kernel, CoLoRAI @ ICML 2026):
<https://openreview.net/forum?id=DqTURDPF5T>
