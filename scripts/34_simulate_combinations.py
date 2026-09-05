#!/usr/bin/env python
"""Simulate every BO-selected drug combination on the spheroid RD model.

Reads the combination table recovered by the BO-selection patch
(`bo_selected_combinations.csv`) plus the ALMANAC Hill fits, converts each
(drug, cell line) pair into a :class:`~src.spheroid_rd.DrugSpec`, runs the
spheroid PDE, and writes one row of outcome-linked readouts per combination.

For every cell line we also run two controls:
  * ``no_treatment``  - the untreated spheroid, which supplies the reference
    burden for log-kill and the reference radial profile for regional kill;
  * ``pool_optimum``  - the best combination in that cell line's BO pool,
    i.e. what an oracle with a full in vitro screen would have picked.

Usage
-----
    python scripts/34_simulate_combinations.py \
        --combos results/pde/bo_selected_combinations.csv \
        --fits   results/hill_fits.parquet \
        --out    results/pde/pde_readouts.csv \
        --workers 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Keep BLAS single-threaded: we parallelise across simulations, and nested
# threading oversubscribes the machine and makes everything slower.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
# ProcessPoolExecutor probes POSIX semaphores at construction, which this
# sandbox forbids; run_jobs uses a forked multiprocessing.Pool instead.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spheroid_rd import (            # noqa: E402
    DrugSpec, SpheroidParams, simulate_spheroid, hill_params_to_drugspec,
)
from src.spheroid_readouts import extract_readouts   # noqa: E402


# --------------------------------------------------------------------------- #
# Drug property assignment
# --------------------------------------------------------------------------- #
#
# The ALMANAC screen gives us potency (from the Hill fit) but says nothing about
# transport or cell-cycle dependence, because a 2D well has neither a diffusion
# barrier nor a hypoxic compartment.  Those two properties are what the spatial
# model needs and what makes it more than a re-parameterised ODE.
#
# We assign them from the drug's molecular identity via its NSC number, using
# published mechanism-of-action classes.  Assignments are deliberately coarse -
# three transport tiers and three cycle-dependence tiers - because the point is
# to test whether *any* physically reasonable spatial structure re-ranks the BO
# methods, not to claim a calibrated per-drug pharmacokinetic model.  The
# sensitivity analysis re-runs the whole study with these values perturbed.

# NSC -> (name, MoA class).  Covers the agents that appear most often in the
# ALMANAC combination pool.
NSC_ANNOTATION = {
    752:    ("thioguanine",       "antimetabolite"),
    755:    ("mercaptopurine",    "antimetabolite"),
    762:    ("mechlorethamine",   "alkylator"),
    3053:   ("dactinomycin",      "dna_intercalator"),
    3088:   ("chlorambucil",      "alkylator"),
    6396:   ("thiotepa",          "alkylator"),
    8806:   ("melphalan",         "alkylator"),
    9706:   ("triethylenemelamine", "alkylator"),
    13875:  ("altretamine",       "alkylator"),
    19893:  ("fluorouracil",      "antimetabolite"),
    26271:  ("cyclophosphamide",  "alkylator"),
    26980:  ("mitomycin",         "alkylator"),
    32065:  ("hydroxyurea",       "antimetabolite"),
    34462:  ("uracil mustard",    "alkylator"),
    38721:  ("mitotane",          "other"),
    45388:  ("dacarbazine",       "alkylator"),
    45923:  ("teniposide",        "topo2"),
    49842:  ("vinblastine",       "antimitotic"),
    63878:  ("cytarabine",        "antimetabolite"),
    66847:  ("thalidomide",       "other"),
    67574:  ("vincristine",       "antimitotic"),
    71423:  ("megestrol",         "hormonal"),
    77213:  ("procarbazine",      "alkylator"),
    79037:  ("lomustine",         "alkylator"),
    82151:  ("daunorubicin",      "dna_intercalator"),
    85998:  ("streptozocin",      "alkylator"),
    92859:  ("dromostanolone",    "hormonal"),
    102816: ("azacitidine",       "antimetabolite"),
    105014: ("cladribine",        "antimetabolite"),
    109724: ("ifosfamide",        "alkylator"),
    118218: ("mithramycin",       "dna_intercalator"),
    122758: ("tretinoin",         "other"),
    123127: ("doxorubicin",       "dna_intercalator"),
    125066: ("bleomycin",         "dna_cleaver"),
    125973: ("paclitaxel",        "antimitotic"),
    127716: ("decitabine",        "antimetabolite"),
    138783: ("amifostine",        "other"),
    141540: ("etoposide",         "topo2"),
    169780: ("dexrazoxane",       "other"),
    180973: ("tamoxifen",         "hormonal"),
    218321: ("pentostatin",       "antimetabolite"),
    241240: ("carboplatin",       "platinum"),
    246131: ("mitoxantrone",      "dna_intercalator"),
    256439: ("idarubicin",        "dna_intercalator"),
    256942: ("epirubicin",        "dna_intercalator"),
    266046: ("oxaliplatin",       "platinum"),
    271674: ("floxuridine",       "antimetabolite"),
    296961: ("interferon",        "other"),
    312887: ("fludarabine",       "antimetabolite"),
    329680: ("vinorelbine",       "antimitotic"),
    362856: ("temozolomide",      "alkylator"),
    369100: ("aldesleukin",       "other"),
    409962: ("carmustine",        "alkylator"),
    606869: ("clofarabine",       "antimetabolite"),
    608210: ("bortezomib",        "proteasome"),
    609699: ("topotecan",         "topo1"),
    613327: ("gemcitabine",       "antimetabolite"),
    673089: ("docetaxel",         "antimitotic"),
    681239: ("bortezomib",        "proteasome"),
    683864: ("temsirolimus",      "kinase_inhibitor"),
    698037: ("pemetrexed",        "antimetabolite"),
    701852: ("irinotecan",        "topo1"),
    712807: ("capecitabine",      "antimetabolite"),
    713563: ("arsenic trioxide",  "other"),
    715055: ("gefitinib",         "kinase_inhibitor"),
    718781: ("erlotinib",         "kinase_inhibitor"),
    719276: ("zoledronic acid",   "other"),
    719344: ("anastrozole",       "hormonal"),
    719345: ("exemestane",        "hormonal"),
    719627: ("letrozole",         "hormonal"),
    720568: ("fulvestrant",       "hormonal"),
    721517: ("imatinib",          "kinase_inhibitor"),
    724998: ("bevacizumab",       "biologic"),
    725776: ("azacitidine",       "antimetabolite"),
    727989: ("nelarabine",        "antimetabolite"),
    732517: ("dasatinib",         "kinase_inhibitor"),
    733504: ("lenalidomide",      "other"),
    737754: ("ixabepilone",       "antimitotic"),
    743414: ("sunitinib",         "kinase_inhibitor"),
    747599: ("nilotinib",         "kinase_inhibitor"),
    747971: ("sorafenib",         "kinase_inhibitor"),
    747972: ("lapatinib",         "kinase_inhibitor"),
    747973: ("vorinostat",        "hdac_inhibitor"),
    750690: ("everolimus",        "kinase_inhibitor"),
    753082: ("pazopanib",         "kinase_inhibitor"),
    754230: ("romidepsin",        "hdac_inhibitor"),
    754143: ("bendamustine",      "alkylator"),
    755986: ("cabazitaxel",       "antimitotic"),
    757441: ("eribulin",          "antimitotic"),
    758774: ("vandetanib",        "kinase_inhibitor"),
    761431: ("vemurafenib",       "kinase_inhibitor"),
    763371: ("crizotinib",        "kinase_inhibitor"),
    119875: ("cisplatin",         "platinum"),
}

# MoA -> (tissue diffusivity [um^2/day], saturable uptake [1/day],
#         cycle dependence phi in [0,1])
#
# Transport tiers reflect molecular size and binding avidity: small unbound
# agents (platinums, antimetabolites, temozolomide) move fastest; DNA-binding
# intercalators and large natural products are strongly retained by the first
# cells they meet, which is the classic cause of poor spheroid penetration;
# taxanes and biologics are slowest.
# Cycle dependence reflects whether the drug needs a cycling cell: S-phase
# antimetabolites and M-phase antimitotics are near 1, DNA-damaging alkylators
# and platinums act on quiescent cells too and sit near 0.3.
MOA_PROPERTIES = {
    "antimetabolite":   (9.0e4, 0.15, 0.95),
    "alkylator":        (8.0e4, 0.20, 0.35),
    "platinum":         (9.5e4, 0.18, 0.40),
    "topo1":            (4.0e4, 0.45, 0.85),
    "topo2":            (3.5e4, 0.50, 0.80),
    "dna_intercalator": (1.2e4, 0.85, 0.60),
    "dna_cleaver":      (2.0e4, 0.70, 0.45),
    "antimitotic":      (1.5e4, 0.75, 0.98),
    "kinase_inhibitor": (5.0e4, 0.35, 0.55),
    "proteasome":       (3.0e4, 0.55, 0.50),
    "hdac_inhibitor":   (5.5e4, 0.30, 0.60),
    "hormonal":         (7.0e4, 0.25, 0.70),
    "biologic":         (2.0e3, 0.95, 0.50),
    "other":            (5.0e4, 0.35, 0.55),
}
DEFAULT_MOA = "other"


def drug_properties(nsc: int) -> tuple[str, str, float, float, float]:
    """Return (name, moa, diffusivity, uptake, cycle_dependence) for an NSC id."""
    name, moa = NSC_ANNOTATION.get(int(nsc), (f"NSC{int(nsc)}", DEFAULT_MOA))
    D, u, phi = MOA_PROPERTIES[moa]
    return name, moa, D, u, phi


# --------------------------------------------------------------------------- #
# One simulation
# --------------------------------------------------------------------------- #

def build_drugspec(nsc: int, conc_M: float, fit_row, 
                   params_override: dict | None = None) -> DrugSpec | None:
    """Build a DrugSpec from an NSC id, its applied concentration and Hill fit.

    ``fit_row`` is an itertuples namedtuple (attribute access), not a Series.
    """
    if fit_row is None:
        return None
    name, moa, D, u, phi = drug_properties(nsc)
    over = params_override or {}
    return hill_params_to_drugspec(
        name=f"{name}({int(nsc)})",
        e0=float(fit_row.e0),
        emax_pct=float(fit_row.emax),
        log_ec50=float(fit_row.log_ec50),
        slope=float(fit_row.slope),
        applied_conc_M=float(conc_M),
        diffusivity=D * over.get("diffusivity_scale", 1.0),
        cycle_dependence=float(np.clip(phi * over.get("phi_scale", 1.0), 0.0, 1.0)),
        uptake=u * over.get("uptake_scale", 1.0),
    )


def simulate_one(job: dict) -> dict:
    """Worker: simulate one combination and return its readouts."""
    t0 = time.time()
    p = SpheroidParams(**job.get("spheroid_params", {}))

    specs = [s for s in job["drug_specs"] if s is not None]
    try:
        ctrl = simulate_spheroid([], p)
        res = simulate_spheroid(specs, p)
        ro = extract_readouts(res, ctrl)
        ok, msg = res.success, res.message
    except Exception as exc:                              # noqa: BLE001
        ro = {}
        ok, msg = False, f"{type(exc).__name__}: {exc}"

    out = dict(job["key"])
    out.update(ro)
    out["n_drugs_simulated"] = len(specs)
    out["sim_ok"] = bool(ok)
    out["sim_message"] = str(msg)[:200]
    out["wall_s"] = round(time.time() - t0, 3)
    return out


# --------------------------------------------------------------------------- #
# Job construction
# --------------------------------------------------------------------------- #

def build_jobs(combos: pd.DataFrame, fits: pd.DataFrame,
               spheroid_params: dict | None = None,
               params_override: dict | None = None) -> list[dict]:
    """Turn the recovered-combination table into a list of simulation jobs."""
    fits_ok = fits[fits["success"]] if "success" in fits.columns else fits
    fit_index = {(str(r.cell_line), int(r.nsc)): r
                 for r in fits_ok.itertuples(index=False)}

    jobs: list[dict] = []
    seen_controls: set[str] = set()

    for row in combos.itertuples(index=False):
        cell = str(row.cell_line)

        # one no-treatment control per cell line
        if cell not in seen_controls:
            seen_controls.add(cell)
            jobs.append({
                "key": {"cell_line": cell, "method": "no_treatment", "seed": -1,
                        "nsc1": -1, "nsc2": -1, "conc1_M": 0.0, "conc2_M": 0.0,
                        "percent_growth": np.nan, "regret": np.nan,
                        "drug1_name": "", "drug2_name": "",
                        "drug1_moa": "", "drug2_moa": ""},
                "drug_specs": [],
                "spheroid_params": spheroid_params or {},
            })

        f1 = fit_index.get((cell, int(row.nsc1)))
        f2 = fit_index.get((cell, int(row.nsc2)))
        s1 = build_drugspec(row.nsc1, row.conc1_M, f1, params_override)
        s2 = build_drugspec(row.nsc2, row.conc2_M, f2, params_override)

        n1, m1, *_ = drug_properties(int(row.nsc1))
        n2, m2, *_ = drug_properties(int(row.nsc2))

        jobs.append({
            "key": {
                "cell_line": cell,
                "method": str(row.method),
                "seed": int(row.seed),
                "nsc1": int(row.nsc1), "nsc2": int(row.nsc2),
                "conc1_M": float(row.conc1_M), "conc2_M": float(row.conc2_M),
                "percent_growth": float(getattr(row, "percent_growth", np.nan)),
                "regret": float(getattr(row, "regret", np.nan)),
                "drug1_name": n1, "drug2_name": n2,
                "drug1_moa": m1, "drug2_moa": m2,
                "has_fit_drug1": f1 is not None, "has_fit_drug2": f2 is not None,
            },
            "drug_specs": [s1, s2],
            "spheroid_params": spheroid_params or {},
        })
    return jobs


def run_jobs(jobs: list[dict], workers: int = 6, label: str = "") -> pd.DataFrame:
    """Run jobs in parallel.

    Uses ``multiprocessing.Pool`` with the *fork* start method rather than
    ``ProcessPoolExecutor``: the latter calls ``os.sysconf("SC_SEM_NSEMS_MAX")``
    at construction, which raises ``PermissionError`` in this sandbox.  A forked
    Pool needs no POSIX semaphore probe.  Falls back to serial execution if even
    that is unavailable, so the script always runs.
    """
    t0, n = time.time(), len(jobs)
    rows: list[dict] = []

    def _tick(i: int) -> None:
        if i % 50 == 0 or i == n:
            el = time.time() - t0
            print(f"  [{label}] {i}/{n} done, {el:.0f}s elapsed, "
                  f"{el/i:.2f}s/sim, eta {(n-i)*el/i:.0f}s", flush=True)

    # Append each result to a shard file as it lands, so a killed run (this
    # machine is memory-constrained and the OS reaps long jobs) resumes instead
    # of restarting.  maxtasksperchild recycles workers to bound RSS growth.
    shard = Path(f"results/pde/_shard_{label or 'run'}.csv")
    shard.parent.mkdir(parents=True, exist_ok=True)
    done_keys: set = set()
    if shard.exists():
        prev = pd.read_csv(shard)
        rows = prev.to_dict("records")
        done_keys = {(r.get("cell_line"), r.get("nsc1"), r.get("nsc2"),
                      r.get("conc1_M"), r.get("conc2_M"), r.get("method"))
                     for r in rows}
        print(f"  [{label}] resuming: {len(rows)} already done", flush=True)
        jobs = [j for j in jobs
                if (j.get("cell_line"), j.get("nsc1"), j.get("nsc2"),
                    j.get("conc1_M"), j.get("conc2_M"),
                    j.get("method")) not in done_keys]
        n = len(jobs)
        if not jobs:
            return pd.DataFrame(rows)

    # The header must be the UNION of every record's keys, fixed before the
    # first write.  Control rows carry no ``has_fit_drug*`` fields, so taking
    # the header from whichever record happens to land first writes a short
    # header that later treated rows silently overflow -- the table then parses
    # with two columns' worth of shift and every downstream number is wrong.
    _shard_cols: list = []

    def _append(rec: dict) -> None:
        # Always write the full column set in a fixed order and quote every
        # field.  The solver's status string contains commas; an unquoted append
        # shifts every subsequent column and silently corrupts the table.
        import csv as _csv
        nonlocal _shard_cols
        if not _shard_cols:
            if shard.exists():
                _shard_cols = list(pd.read_csv(shard, nrows=0).columns)
            else:
                _shard_cols = list(rec.keys())
        new = [k for k in rec if k not in _shard_cols]
        if new:
            # Never silently widen an existing file: rewrite it under the wider
            # header so old and new rows stay column-aligned.
            _shard_cols = _shard_cols + new
            if shard.exists():
                old = pd.read_csv(shard)
                old.reindex(columns=_shard_cols).to_csv(
                    shard, index=False, quoting=_csv.QUOTE_ALL)
        df1 = pd.DataFrame([rec]).reindex(columns=_shard_cols)
        df1.to_csv(shard, mode="a", header=not shard.exists(), index=False,
                   quoting=_csv.QUOTE_ALL)

    if workers and workers > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=workers, maxtasksperchild=25) as pool:
                for i, r in enumerate(pool.imap_unordered(simulate_one, jobs,
                                                          chunksize=1), 1):
                    rows.append(r)
                    _append(r)
                    _tick(i)
            return pd.DataFrame(rows)
        except (PermissionError, OSError, ValueError) as exc:
            print(f"  [{label}] parallel unavailable ({type(exc).__name__}), "
                  f"falling back to serial", flush=True)
            rows = []

    for i, j in enumerate(jobs, 1):
        r = simulate_one(j)
        rows.append(r); _append(r)
        _tick(i)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combos", type=Path,
                    default=ROOT / "results" / "pde" / "bo_selected_combinations.csv")
    ap.add_argument("--fits", type=Path, default=ROOT / "results" / "hill_fits.parquet")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "pde" / "pde_readouts.csv")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--t-end", type=float, default=21.0)
    ap.add_argument("--radius", type=float, default=400.0)
    ap.add_argument("--n-r", type=int, default=160)
    ap.add_argument("--limit", type=int, default=None,
                    help="simulate only the first N combinations (smoke test)")
    args = ap.parse_args()

    combos = pd.read_csv(args.combos)
    fits = pd.read_parquet(args.fits)
    if args.limit:
        combos = combos.head(args.limit)

    sp = {"t_end": args.t_end, "radius": args.radius, "n_r": args.n_r,
          "n_t": int(args.t_end * 5) + 1}

    # Deduplicate before simulating.  The PDE depends only on the cell line and
    # the two (drug, dose) pairs; different methods and seeds frequently select
    # the SAME combination, and the shared no-treatment control is identical
    # across every row of a cell line.  On the recovered ALMANAC table this
    # collapses 3,660 rows to ~1,450 distinct simulations.
    SIM_KEY = ["cell_line", "nsc1", "nsc2", "conc1_M", "conc2_M"]
    uniq = combos.drop_duplicates(subset=SIM_KEY).copy()
    print(f"{len(combos)} combination rows -> {len(uniq)} distinct simulations "
          f"({combos['cell_line'].nunique()} cell lines)", flush=True)

    jobs = build_jobs(uniq, fits, spheroid_params=sp)
    print(f"built {len(jobs)} jobs (incl. one no-treatment control per cell line)",
          flush=True)

    sims = run_jobs(jobs, workers=args.workers, label="pde")

    # Split controls out, then broadcast the deduplicated readouts back onto
    # every original (cell_line, method, seed) row.
    ctrl = sims[sims["method"] == "no_treatment"]
    sim_only = sims[sims["method"] != "no_treatment"]
    readout_cols = [c for c in sim_only.columns
                    if c not in set(combos.columns) | {"method", "seed"}]
    merged = combos.merge(sim_only[SIM_KEY + readout_cols], on=SIM_KEY, how="left")

    df = pd.concat([merged, ctrl], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_ok = int(df["sim_ok"].fillna(False).sum())
    print(f"wrote {args.out}  shape={df.shape}  ok={n_ok}/{len(df)}  "
          f"(unique sims run: {len(sims)})")


if __name__ == "__main__":
    main()
