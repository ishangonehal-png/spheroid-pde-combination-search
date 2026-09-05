"""Reproducibility checks for the spheroid RD pipeline.

Run:  python tests/test_reproduce.py
Exits non-zero on any failure. No pytest dependency.
"""
import subprocess, sys, tempfile, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
FAIL = []

def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)

# 1. Solver verification against analytic references.
import spheroid_rd as S
v = S.verify_solver()
check("oxygen profile vs analytic (rel L2 < 0.01)",
      v["oxygen_rel_l2"] < 0.01, f"{v['oxygen_rel_l2']:.2e}")
check("Fisher-KPP wave speed within 8% of 2*sqrt(D*rho)",
      v["kpp_rel_err"] < 0.08, f"{v['kpp_rel_err']*100:.1f}%")
check("mesh convergence ratio > 2 (>=1st order)",
      v["mesh_convergence_ratio"] > 2.0, f"{v['mesh_convergence_ratio']:.2f}")

# 2. Pipeline reproduces published readouts bit-exactly.
out = Path(tempfile.mkdtemp()) / "repro.csv"
cmd = [sys.executable, str(ROOT / "scripts" / "34_simulate_combinations.py"),
       "--combos", str(ROOT / "data" / "bo_selected_combinations.csv"),
       "--fits",   str(ROOT / "data" / "almanac_hill_fits.parquet"),
       "--out", str(out), "--workers", "2", "--limit", "6"]
r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
check("simulation script exits 0", r.returncode == 0, r.stderr.strip()[-160:])

if out.exists():
    new = pd.read_csv(out)
    ref = pd.read_csv(ROOT / "results" / "pde_readouts.csv")
    check("all simulations succeeded", bool(new.sim_ok.all()))
    keys = ["cell_line", "method", "seed", "nsc1", "nsc2"]
    m = new[new.method != "no_treatment"].merge(ref, on=keys, suffixes=("_n", "_o"))
    check("rows matched against published results", len(m) > 0, f"n={len(m)}")
    for c in ["log_kill_auc", "hypoxic_fraction", "nadir_burden", "depth_of_response"]:
        if f"{c}_n" in m and len(m):
            d = float((m[f"{c}_n"] - m[f"{c}_o"]).abs().max())
            check(f"{c} reproduces exactly", d == 0.0, f"max|diff|={d:.3e}")

# 3. Published result tables are internally consistent with the paper's claims.
mono = pd.read_csv(ROOT / "results" / "cohort_monotonicity.csv")
col = "spearman" if "spearman" in mono.columns else mono.columns[1]
lka = mono.loc[mono.iloc[:, 0].astype(str).str.contains("log_kill_auc"), col]
check("cohort log-kill AUC monotonicity approx -0.55",
      len(lka) and abs(float(lka.iloc[0]) + 0.548) < 0.01,
      f"{float(lka.iloc[0]):.4f}" if len(lka) else "column not found")

print()
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED: {FAIL}")
    sys.exit(1)
print("ALL CHECKS PASSED")
