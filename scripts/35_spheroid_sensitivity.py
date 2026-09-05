#!/usr/bin/env python
"""Parameter sensitivity and robustness sweeps for the spheroid RD study.

The spheroid model carries parameters that are only known to within an order of
magnitude (oxygen consumption, drug diffusivity, proliferation rate) and design
choices that are ours rather than the data's (spheroid radius, dosing schedule).
A conclusion that survives only at the default settings is not a conclusion.

This script re-runs the whole combination study under perturbed settings and
reports, for each perturbation, whether the *ranking of BO methods by PDE
readout* is preserved.  The headline robustness statistic is the Spearman
correlation between the per-method mean readout under the perturbation and under
the default, computed across methods.

Sweeps
------
``oxygen_consumption``  q_O in {7.5, 15, 30}   - hypoxic-core size
``drug_diffusivity``    global scale {0.3, 1, 3} - penetration barrier strength
``rho``                 proliferation {0.3, 0.6, 1.2}
``radius``              spheroid size {250, 400, 600} um
``dose_schedule``       {5 daily, 3 q2d, 1 bolus} at matched total exposure
``cycle_dependence``    phi scaled {0.5, 1.0, 1.5} - hypoxic-resistance strength

Usage
-----
    python scripts/35_spheroid_sensitivity.py \
        --combos results/pde/bo_selected_combinations.csv \
        --fits   results/hill_fits.parquet \
        --out    results/pde/pde_sensitivity.csv \
        --workers 6 --max-combos 120
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module
_sim = import_module("34_simulate_combinations")
build_jobs, run_jobs = _sim.build_jobs, _sim.run_jobs


# name -> (spheroid_params_override, drug_property_override)
def sweep_grid(t_end: float) -> dict[str, tuple[dict, dict]]:
    base_sp = {"t_end": t_end, "n_t": int(t_end * 5) + 1}
    g: dict[str, tuple[dict, dict]] = {
        "default": (dict(base_sp), {}),

        # --- microenvironment ---
        "oxygen_low":   (dict(base_sp, oxygen_consumption=7.5), {}),
        "oxygen_high":  (dict(base_sp, oxygen_consumption=30.0), {}),

        # --- transport ---
        "diffusivity_low":  (dict(base_sp), {"diffusivity_scale": 0.3}),
        "diffusivity_high": (dict(base_sp), {"diffusivity_scale": 3.0}),

        # --- growth ---
        "rho_low":  (dict(base_sp, rho=0.30), {}),
        "rho_high": (dict(base_sp, rho=1.20), {}),

        # --- geometry ---
        "radius_small": (dict(base_sp, radius=250.0), {}),
        "radius_large": (dict(base_sp, radius=600.0, n_r=220), {}),

        # --- schedule, at matched cumulative surface exposure ---
        # default is 5 daily doses of 0.25 d; these two redistribute the same
        # total dose-time so differences are schedule shape, not total dose.
        "schedule_q2d":   (dict(base_sp, dose_times=(0.0, 2.0, 4.0),
                                dose_duration=0.4167), {}),
        "schedule_bolus": (dict(base_sp, dose_times=(0.0,),
                                dose_duration=1.25), {}),

        # --- hypoxic resistance strength ---
        "phi_weak":   (dict(base_sp), {"phi_scale": 0.5}),
        "phi_strong": (dict(base_sp), {"phi_scale": 1.5}),
    }
    return g


KEY_READOUTS = [
    "log_kill_nadir", "log_kill_auc", "nadir_burden", "depth_of_response",
    "kg_regrowth", "core_sparing", "hypoxic_fraction", "residual_core_burden",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combos", type=Path,
                    default=ROOT / "results" / "pde" / "bo_selected_combinations.csv")
    ap.add_argument("--fits", type=Path, default=ROOT / "results" / "hill_fits.parquet")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "pde" / "pde_sensitivity.csv")
    ap.add_argument("--summary-out", type=Path,
                    default=ROOT / "results" / "pde" / "pde_sensitivity_summary.csv")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--t-end", type=float, default=21.0)
    ap.add_argument("--max-combos", type=int, default=None,
                    help="subsample combinations to keep the sweep affordable")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    combos = pd.read_csv(args.combos)
    fits = pd.read_parquet(args.fits)

    # Subsample by (cell_line, method, seed) so every method stays represented.
    if args.max_combos and len(combos) > args.max_combos:
        # Stratify by method so every BO method keeps representation, and take
        # the sample by positional index to avoid groupby-apply folding the
        # grouping key into the index (which would drop `method` as a column).
        frac = args.max_combos / len(combos)
        idx: list[int] = []
        for _, g in combos.groupby("method", sort=False):
            n_take = max(1, int(round(len(g) * frac)))
            idx.extend(g.sample(n_take, random_state=args.seed).index.tolist())
        combos = combos.loc[sorted(idx)].reset_index(drop=True)
    print(f"sweeping {len(combos)} combinations x {len(sweep_grid(args.t_end))} settings",
          flush=True)

    all_rows = []
    for name, (sp, over) in sweep_grid(args.t_end).items():
        jobs = build_jobs(combos, fits, spheroid_params=sp, params_override=over)
        df = run_jobs(jobs, workers=args.workers, label=name)
        df["sweep"] = name
        all_rows.append(df)
        print(f"[{name}] {len(df)} rows, ok={int(df['sim_ok'].sum())}", flush=True)

    out = pd.concat(all_rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} shape={out.shape}")

    # ---- robustness summary: does each sweep preserve the method ranking? ---
    treated = out[out["method"] != "no_treatment"]
    ref = (treated[treated["sweep"] == "default"]
           .groupby("method")[KEY_READOUTS].mean())

    rows = []
    for name, grp in treated.groupby("sweep"):
        m = grp.groupby("method")[KEY_READOUTS].mean()
        common = ref.index.intersection(m.index)
        for ro in KEY_READOUTS:
            a, b = ref.loc[common, ro], m.loc[common, ro]
            ok = a.notna() & b.notna()
            if ok.sum() >= 3:
                rho, p = spearmanr(a[ok], b[ok])
            else:
                rho, p = np.nan, np.nan
            rows.append({"sweep": name, "readout": ro, "n_methods": int(ok.sum()),
                         "spearman_vs_default": rho, "p_value": p})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.summary_out, index=False)
    print(f"wrote {args.summary_out}")

    piv = summary.pivot(index="sweep", columns="readout",
                        values="spearman_vs_default")
    print("\nRank preservation vs default (Spearman across methods):")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
