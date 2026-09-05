"""Re-run the ALMANAC BO benchmark under the paper protocol, RECORDING which
combination each method actually selected at every step.

The published artifacts record regret only; `BOTrace.queries` was never
persisted, so the chemical identity of each method's chosen combination was
lost. This driver reuses the (additively patched) machinery in
`11_run_benchmarks.py` -- same pool construction, same seed-13 subsample, same
EI loop -- and joins `BOTrace.queries` back onto the row-aligned pool identity
table.

Protocol (paper App. N): n_init=10, n_iter=20, q=8, m_max=2, ZeroMean,
fit_iters=30 @ fit_lr=0.05, EI, pool subsampled to 4000 per context, seeds 0..9.

Parallelised across (cell_line, seed) with one torch thread per worker.

Usage:
    python scripts/33_run_almanac_selection_recovery.py --n-seeds 10 --workers 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Single-threaded per worker: set BEFORE importing numpy/torch.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import torch


def _install_memory_watchdog(limit_gb: float = 8.0, poll_s: float = 0.5) -> None:
    """Self-terminate if this process's RSS exceeds ``limit_gb``.

    SAASBO's fully Bayesian posterior is the one component here that can
    allocate without bound (an (S, N, N) joint covariance over the candidate
    pool).  That is fixed properly in ``surrogates_saasbo`` by chunking, but a
    runaway must never again be able to take the whole workstation down with
    it: macOS ignores RLIMIT_AS, so we poll and exit instead.  A dead worker
    loses one (cell, seed); a swapped machine loses everything.
    """
    import resource
    import threading

    def _watch() -> None:
        while True:
            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3
            if rss_gb > limit_gb:
                print(f"[watchdog] RSS {rss_gb:.1f} GB > {limit_gb} GB -- aborting worker",
                      flush=True)
                os._exit(9)
            time.sleep(poll_s)

    threading.Thread(target=_watch, daemon=True).start()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("bench_runner", ROOT / "scripts" / "11_run_benchmarks.py")
bench = _ilu.module_from_spec(_spec)
# Register before exec: @dataclass resolves annotations via sys.modules.
sys.modules["bench_runner"] = bench
_spec.loader.exec_module(bench)

from src.almanac_bo import make_combo_pool
from src.combo_kernel import MonoFingerprintKernel

METHODS = ["ours+zero", "tanimoto+zero", "hamming+zero",
           "act_hamming+zero", "linear_curve", "random"]

# Protocol constants (paper App. N).
N_INIT, N_ITER = 10, 20
EMB_DIM, INTER_ORDER = 8, 2
FIT_ITERS, FIT_LR = 30, 0.05
POOL_SUBSAMPLE = 4000


def _worker(task):
    """Run all methods x one seed for one cell line. Returns (sel, summ, opt)."""
    cell_line, seed, methods = task
    torch.set_num_threads(1)
    _install_memory_watchdog(float(os.environ.get("SAVITAR_MEM_LIMIT_GB", "8")))

    combo = _G["combo"]
    fits = _G["fits"]

    kernel = MonoFingerprintKernel(
        fits, cell_line=cell_line, n_drugs=2,
        embedding_dim=EMB_DIM, interaction_order=INTER_ORDER,
    )
    x_full, y_full, tbl_full = make_combo_pool(combo, cell_line, kernel)
    x_pool, y_pool, sub_idx = bench.subsample_pool_with_index(
        x_full, y_full, pool_subsample=POOL_SUBSAMPLE
    )
    pool_table = tbl_full.iloc[sub_idx].reset_index(drop=True)

    # Integrity guard: the identity table must be row-aligned to y_pool.
    assert np.array_equal(
        y_pool.numpy(), pool_table["y_mean"].to_numpy(np.float32)
    ), f"pool/table misalignment for {cell_line}"

    ctx = bench.ContextPool(
        dataset="almanac", context=cell_line, fits=fits,
        x_pool=x_pool, y_pool=y_pool,
        n_drugs_in_pool=kernel.n_drugs_in_pool,
        legacy_factory_kwargs={},
        pool_table=pool_table, pool_source_index=sub_idx,
    )

    global_min = float(y_pool.min().item())
    opt_rows = []
    if seed == 0:  # pool optimum is seed-independent; record once per cell
        oi = int(torch.argmin(y_pool).item())
        o = pool_table.iloc[oi]
        opt_rows.append({
            "dataset": "almanac", "context": cell_line, "row_kind": "pool_optimum",
            "method": "pool_optimum", "seed": -1,
            "pool_index": oi, "source_pool_index": int(sub_idx[oi]),
            "nsc1": int(o["nsc1"]), "nsc2": int(o["nsc2"]),
            "conc1_M": float(o["conc1_M"]), "conc2_M": float(o["conc2_M"]),
            "percent_growth": global_min, "n_replicates": int(o["n_replicates"]),
            "final_regret": 0.0, "pool_size": int(len(y_pool)),
            "pool_size_full": int(len(y_full)),
        })

    sel_rows, summ_rows = [], []
    for method in methods:
        t0 = time.time()
        trace, wall = bench.run_method(
            ctx, method, seed=seed, n_init=N_INIT, n_iter=N_ITER,
            fit_iters=FIT_ITERS, fit_lr=FIT_LR,
            embedding_dim=EMB_DIM, interaction_order=INTER_ORDER,
            saas_fit_kwargs={},
        )
        q = np.asarray(trace.queries, dtype=np.int64)
        y_q = y_pool.numpy()[q]
        best_pos = int(np.argmin(y_q))

        # Consistency guard: cummin of queried y must equal the trace's
        # best_so_far, i.e. the query sequence really is this run's trajectory.
        assert np.allclose(np.minimum.accumulate(y_q), trace.best_so_far, atol=0), \
            f"query/best_so_far mismatch {cell_line} {method} seed{seed}"

        for step, pidx in enumerate(q):
            r = pool_table.iloc[int(pidx)]
            sel_rows.append({
                "dataset": "almanac", "context": cell_line, "method": method,
                "seed": seed, "step": step,
                "phase": "init" if step < N_INIT else "bo",
                "pool_index": int(pidx),
                "source_pool_index": int(sub_idx[int(pidx)]),
                "nsc1": int(r["nsc1"]), "nsc2": int(r["nsc2"]),
                "conc1_M": float(r["conc1_M"]), "conc2_M": float(r["conc2_M"]),
                "percent_growth": float(y_q[step]),
                "n_replicates": int(r["n_replicates"]),
                "regret": float(y_q[step] - global_min),
                "best_so_far": float(trace.best_so_far[step]),
                "is_best_selection": bool(step == best_pos),
                "is_final_step": bool(step == len(q) - 1),
            })

        b = pool_table.iloc[int(q[best_pos])]
        summ_rows.append({
            "dataset": "almanac", "context": cell_line, "row_kind": "bo_selection",
            "method": method, "seed": seed,
            "pool_index": int(q[best_pos]),
            "source_pool_index": int(sub_idx[int(q[best_pos])]),
            "nsc1": int(b["nsc1"]), "nsc2": int(b["nsc2"]),
            "conc1_M": float(b["conc1_M"]), "conc2_M": float(b["conc2_M"]),
            "percent_growth": float(y_q[best_pos]),
            "n_replicates": int(b["n_replicates"]),
            "final_regret": float(trace.regret[-1]),
            "pool_optimum_percent_growth": global_min,
            "chose_pool_optimum": bool(float(trace.regret[-1]) == 0.0),
            "best_step": best_pos,
            "wall_time_s": float(wall),
            "pool_size": int(len(y_pool)),
            "pool_size_full": int(len(y_full)),
            "n_init": N_INIT, "n_iter": N_ITER,
        })
        del t0
    return sel_rows, summ_rows, opt_rows


_G: dict = {}


def _init_worker(combo_path: str, fits_path: str):
    torch.set_num_threads(1)
    _G["combo"] = pd.read_parquet(combo_path)
    _G["fits"] = pd.read_parquet(fits_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--out-prefix", type=str, default="almanac_selrecovery")
    args = ap.parse_args()

    results_dir = ROOT / "results"
    combo_path = results_dir / "almanac_combo.parquet"
    fits_path = results_dir / "hill_fits.parquet"

    combo = pd.read_parquet(combo_path, columns=["cell_line"])
    cells = args.cells or sorted(combo["cell_line"].unique())
    del combo

    tasks = [(c, s, args.methods) for c in cells for s in range(args.n_seeds)]
    print(f"[driver] {len(cells)} cells x {args.n_seeds} seeds x "
          f"{len(args.methods)} methods = {len(tasks)} tasks, "
          f"{args.workers} workers", flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    sel_all, summ_all, opt_all = [], [], []
    t0 = time.time()
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(str(combo_path), str(fits_path))) as pool:
        for i, (s, m, o) in enumerate(
                pool.imap_unordered(_worker, tasks, chunksize=1), start=1):
            sel_all.extend(s); summ_all.extend(m); opt_all.extend(o)
            if i % 25 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"[driver] {i}/{len(tasks)} tasks  {el:.0f}s elapsed  "
                      f"eta {el/i*(len(tasks)-i):.0f}s", flush=True)

    sel = pd.DataFrame(sel_all).sort_values(
        ["context", "method", "seed", "step"]).reset_index(drop=True)
    summ = pd.DataFrame(summ_all)
    opt = pd.DataFrame(opt_all)

    p1 = results_dir / f"{args.out_prefix}_selections.parquet"
    p2 = results_dir / f"{args.out_prefix}_chosen.parquet"
    p3 = results_dir / f"{args.out_prefix}_pool_optimum.parquet"
    sel.to_parquet(p1, index=False)
    summ.to_parquet(p2, index=False)
    opt.to_parquet(p3, index=False)
    print(f"[driver] wrote {p1.name} ({len(sel)}), {p2.name} ({len(summ)}), "
          f"{p3.name} ({len(opt)}) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
