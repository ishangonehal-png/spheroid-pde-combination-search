#!/usr/bin/env python
"""Does spatial simulation re-rank the BO methods relative to in vitro regret?

This is the central analysis of the PDE extension.  Appendix V of the published
manuscript maps each method's chosen combination through a well-mixed Gompertz
ODE and finds that the map "preserves rank order across methods" - it is
monotone in the in vitro readout, so it cannot change any conclusion.

The spheroid model is not monotone, because a combination's spatial fate depends
on drug transport and on the oxygen field, neither of which the in vitro
percent-growth number contains.  Here we quantify that:

1. **Monotonicity test.**  Spearman correlation between in vitro percent growth
   and each PDE readout, pooled and within cell line.  A perfectly monotone
   bridge gives |rho| = 1.  We also fit the *published* Appendix-V ODE to the
   same combinations and confirm it is (near-)monotone, which is the control
   that shows the difference comes from spatial structure and not from our
   choice of readout.

2. **Method-ranking test.**  Rank the BO methods by mean in vitro regret and by
   mean PDE readout, per cell line, and count discordant pairs (Kendall tau).

3. **Pairwise re-ranking (the headline).**  For every pair of combinations
   within a cell line, ask whether the PDE ordering disagrees with the in vitro
   ordering, and report the discordance rate with a bootstrap CI.  Discordant
   pairs are then characterised: what distinguishes a combination that the
   spheroid promotes from one it demotes?

4. **Paired method comparison.**  Wilcoxon signed-rank on Savitar vs each
   baseline, matching the manuscript's existing statistical methodology, with
   Holm correction across baselines.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations as iter_pairs
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAVITAR = "ours+zero"

# Readouts the literature survey supports as outcome-linked, plus the
# mechanism diagnostics that explain re-ranking when it happens.
OUTCOME_READOUTS = [
    "kg_biexp",          # Tier A: growth constant, predicts OS (Stein 2008)
    "ttg_biexp",         # Tier A: time to growth (Claret 2013)
    "depth_of_response", # Tier A: DpR (Cremolini 2015)
    "ets_early",         # Tier A: early tumour shrinkage (Piessevaux 2013)
    "log_kill_nadir",
    "log_kill_auc",
    "nadir_burden",
    "residual_core_burden",
    "hypoxic_fraction",
]
MECHANISM_READOUTS = [
    "core_sparing", "penetration_depth_drug1", "penetration_depth_drug2",
    "core_to_surface_ratio_drug1", "core_to_surface_ratio_drug2",
    "kill_fraction_core", "kill_fraction_rim", "viable_rim_thickness",
]
# Sign convention: +1 if a LARGER value means a BETTER outcome.
READOUT_SIGN = {
    "kg_biexp": -1, "ttg_biexp": +1, "depth_of_response": +1, "ets_early": +1,
    "log_kill_nadir": +1, "log_kill_auc": +1, "nadir_burden": -1,
    "residual_core_burden": -1, "hypoxic_fraction": -1,
}


# --------------------------------------------------------------------------- #
# The published Appendix-V ODE, as the monotone control
# --------------------------------------------------------------------------- #

def appendix_v_ode(percent_growth: np.ndarray, k_drugs: int = 2,
                   a: float = 0.10, K: float = 1e12, N0: float = 1e9,
                   t_invitro: float = 2.0, t_total: float = 30.0,
                   t_start: float = 2.0, period: float = 1.0,
                   t_half: float = 1.0, viability_floor: float = 1e-3,
                   return_components: bool = False):
    """Reproduce the manuscript's Gompertz + linear-PK bridge (Appendix V).

    The paper (p. 20) maps ALMANAC percent growth ``g`` to a kill rate via

        v(g)        = (1/2)(g + 100)                 [% viability]
        gamma_combo = -log(v/100) / T_vitro
        gamma_i     = gamma_combo / k                (equal log-split)

    Note the ``(g + 100)/2`` transform: ALMANAC percent growth runs from +100
    (no effect) down to -100 (complete kill relative to the time-zero density),
    so viability is the midpoint rescaling, NOT ``g/100``.  The two coincide
    only at ``g = 100``.  Table 21's ``gamma`` column reports the PER-DRUG
    ``gamma_i``, which is how the values there (0.85 for the Savitar combination
    at y = -93.2, 0.42 at y = -62.4, 0.25 at y = -27.0, 0.13 at y = +17.0)
    are reproduced.

    Because all k drugs share the same dosing schedule, the total kill term is
    ``sum_i gamma_i * PK(t) = gamma_combo * PK(t)``, so k affects the reported
    per-drug gamma but not the trajectory.

    Growth follows the Gompertz form in log space,

        d log N / dt = a (log K - log N) - gamma(t),
        gamma(t)     = gamma_combo * sum_{t_d <= t} exp(-(t - t_d) ln2 / T_half)

    with daily dosing on days {2, ..., 30}.

    Returns Table 21's ``log10_red``, which is referenced to the INITIAL burden,

        log10_red = log10( N0 / N(t_total) ),

    i.e. how many logs the tumour was driven below where it started.  It is NOT
    referenced to the untreated day-30 population; a regimen that merely holds
    the tumour at ``N0`` scores 0, and the no-treatment arm scores negative
    (-2.85 in Table 21, since the untreated tumour grows from 1e9 to 7.1e11).
    The untreated-control log-kill is also returned, as ``log10_red_vs_control``,
    for comparison with the spheroid readouts, which use that convention.
    """
    from scipy.integrate import solve_ivp

    g = np.asarray(percent_growth, float)
    # Paper's transform: percent growth -> percent viability -> kill rate.
    v_pct = 0.5 * (g + 100.0)
    v = np.clip(v_pct / 100.0, viability_floor, None)
    gamma_combo = -np.log(v) / t_invitro
    gamma_per_drug = gamma_combo / max(int(k_drugs), 1)

    doses = np.arange(t_start, t_total + 1e-9, period)
    ke = np.log(2.0) / t_half

    def pk(t: float) -> float:
        d = t - doses
        d = d[d >= 0]
        return float(np.sum(np.exp(-ke * d))) if d.size else 0.0

    # untreated Gompertz control, integrated once
    sol0 = solve_ivp(lambda t, y: [a * (np.log(K) - y[0])],
                     (0.0, t_total), [np.log(N0)], rtol=1e-10, atol=1e-10)
    log_n_ctrl = float(sol0.y[0, -1])

    log_n_end = np.empty(len(v))
    for i, gc in enumerate(gamma_combo):
        sol = solve_ivp(lambda t, y: [a * (np.log(K) - y[0]) - gc * pk(t)],
                        (0.0, t_total), [np.log(N0)], rtol=1e-10, atol=1e-10,
                        max_step=0.25)
        log_n_end[i] = float(sol.y[0, -1])

    # Table 21 convention: reduction relative to the INITIAL burden N0.
    log10_red = (np.log(N0) - log_n_end) / np.log(10.0)
    # Alternative convention, matching the spheroid log-kill readouts.
    log10_red_vs_control = (log_n_ctrl - log_n_end) / np.log(10.0)

    if return_components:
        return pd.DataFrame({
            "percent_growth": g,
            "viability_pct": v_pct,
            "gamma_combo": gamma_combo,
            "gamma_per_drug": gamma_per_drug,
            "N30": np.exp(log_n_end),
            "log10_red": log10_red,
            "log10_red_vs_control": log10_red_vs_control,
            "N30_control": np.exp(log_n_ctrl),
        })
    return log10_red


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

def monotonicity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman of each readout against in vitro percent growth."""
    rows = []
    for ro in OUTCOME_READOUTS + MECHANISM_READOUTS:
        if ro not in df.columns:
            continue
        sub = df[["percent_growth", ro]].dropna()
        if len(sub) < 8:
            continue
        rho, p = spearmanr(sub["percent_growth"], sub[ro])
        # within-cell-line, then averaged (removes cell-line as a confounder)
        withins = []
        for cl, g in df.groupby("cell_line"):
            s = g[["percent_growth", ro]].dropna()
            if len(s) >= 6 and s["percent_growth"].nunique() > 2:
                r, _ = spearmanr(s["percent_growth"], s[ro])
                if np.isfinite(r):
                    withins.append(r)
        rows.append({
            "readout": ro,
            "spearman_vs_invitro_pooled": rho,
            "p_pooled": p,
            "spearman_within_cellline_mean": float(np.mean(withins)) if withins else np.nan,
            "spearman_within_cellline_sd": float(np.std(withins)) if withins else np.nan,
            "n_celllines": len(withins),
            "n_obs": len(sub),
            "abs_rho": abs(rho),
        })
    return pd.DataFrame(rows).sort_values("abs_rho")


def _cellline_pair_counts(pg: np.ndarray, ro: np.ndarray,
                          sign: int) -> tuple[int, int]:
    """(discordant, comparable) pair counts within one cell line, vectorised.

    In vitro "better" is LOWER percent growth; ``sign`` orients the readout so
    larger is better.  A pair is comparable when neither ordering is a tie.
    """
    a = -pg                      # larger = better, matching the readout's sense
    b = sign * ro
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(len(a), k=1)
    da, db = da[iu], db[iu]
    ok = (da != 0) & (db != 0)
    return int(np.sum(ok & (da != db))), int(np.sum(ok))


def pairwise_reranking(df: pd.DataFrame, readout: str,
                       n_boot: int = 2000, seed: int = 0,
                       max_records: int = 5000) -> dict:
    """Fraction of within-cell-line combination pairs the PDE orders differently.

    All O(n^2) pair comparisons are done with numpy broadcasting per cell line,
    and the cluster bootstrap resamples *precomputed per-cell-line counts*
    rather than recomputing pairs, which is what makes 2000 replicates over
    3,600 combinations tractable.
    """
    sign = READOUT_SIGN.get(readout, +1)
    per_cell: dict[str, tuple[int, int]] = {}
    records: list[dict] = []

    for cl, g in df.groupby("cell_line"):
        sub = g[["method", "seed", "percent_growth", readout,
                 "core_sparing"]].dropna(subset=["percent_growth", readout])
        if len(sub) < 2:
            continue
        pg = sub["percent_growth"].to_numpy(float)
        ro = sub[readout].to_numpy(float)
        per_cell[cl] = _cellline_pair_counts(pg, ro, sign)

        if len(records) < max_records:      # sample discordant pairs for QC
            a, b = -pg, sign * ro
            da = np.sign(a[:, None] - a[None, :])
            db = np.sign(b[:, None] - b[None, :])
            iu = np.triu_indices(len(a), k=1)
            m = (da[iu] != 0) & (db[iu] != 0) & (da[iu] != db[iu])
            idx_i, idx_j = iu[0][m], iu[1][m]
            meth = sub["method"].to_numpy()
            cs = sub["core_sparing"].to_numpy(float)
            for i, j in list(zip(idx_i, idx_j))[:max_records - len(records)]:
                records.append({
                    "cell_line": cl,
                    "method_a": meth[i], "method_b": meth[j],
                    "pg_a": pg[i], "pg_b": pg[j],
                    "ro_a": ro[i], "ro_b": ro[j],
                    "d_core_sparing": cs[i] - cs[j],
                })

    disc = sum(d for d, _ in per_cell.values())
    total = sum(t for _, t in per_cell.values())
    rate = disc / total if total else np.nan

    # cluster bootstrap over cell lines, resampling the cached counts
    cls = np.array(list(per_cell.keys()))
    boots = []
    if len(cls) > 1:
        D = np.array([per_cell[c][0] for c in cls], float)
        T = np.array([per_cell[c][1] for c in cls], float)
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, len(cls), size=(n_boot, len(cls)))
        num, den = D[pick].sum(axis=1), T[pick].sum(axis=1)
        good = den > 0
        boots = (num[good] / den[good]).tolist()
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))

    return {"readout": readout, "discordance_rate": rate, "ci_lo": lo, "ci_hi": hi,
            "n_pairs": total, "n_discordant": disc,
            "n_celllines": len(per_cell),
            "discordant_records": pd.DataFrame(records)}


def method_ranking(df: pd.DataFrame, readout: str) -> pd.DataFrame:
    """Per-cell-line Kendall tau between method ranking by regret and by readout."""
    sign = READOUT_SIGN.get(readout, +1)
    rows = []
    for cl, g in df.groupby("cell_line"):
        m = g.groupby("method").agg(regret=("regret", "mean"),
                                    ro=(readout, "mean")).dropna()
        if len(m) < 3:
            continue
        tau, p = kendalltau(m["regret"].rank(), (-sign * m["ro"]).rank())
        rows.append({"cell_line": cl, "readout": readout, "kendall_tau": tau,
                     "p_value": p, "n_methods": len(m)})
    return pd.DataFrame(rows)


def savitar_vs_baselines(df: pd.DataFrame, readout: str) -> pd.DataFrame:
    """Paired Wilcoxon of Savitar against each baseline, Holm-corrected."""
    sign = READOUT_SIGN.get(readout, +1)
    piv = (df.pivot_table(index=["cell_line", "seed"], columns="method",
                          values=readout, aggfunc="mean"))
    if SAVITAR not in piv.columns:
        return pd.DataFrame()
    rows = []
    for b in [c for c in piv.columns if c != SAVITAR]:
        sub = piv[[SAVITAR, b]].dropna()
        if len(sub) < 6:
            continue
        a, c = sign * sub[SAVITAR].values, sign * sub[b].values
        if np.allclose(a, c):
            continue
        try:
            stat, p = wilcoxon(a, c)
        except ValueError:
            continue
        rows.append({"readout": readout, "baseline": b, "n_pairs": len(sub),
                     "savitar_mean": float(sub[SAVITAR].mean()),
                     "baseline_mean": float(sub[b].mean()),
                     "median_diff_oriented": float(np.median(a - c)),
                     "wilcoxon_stat": float(stat), "p_raw": float(p)})
    out = pd.DataFrame(rows)
    if len(out):   # Holm correction across baselines
        out = out.sort_values("p_raw").reset_index(drop=True)
        n = len(out)
        out["p_holm"] = [min(1.0, (n - i) * p) for i, p in enumerate(out["p_raw"])]
        out["p_holm"] = out["p_holm"].cummax()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readouts", type=Path,
                    default=ROOT / "results" / "pde" / "pde_readouts.csv")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "pde")
    ap.add_argument("--primary", type=str, default="log_kill_auc")
    ap.add_argument("--with-ode-control", action="store_true",
                    help="also run the Appendix-V ODE on the same combinations")
    args = ap.parse_args()

    df = pd.read_csv(args.readouts)
    df = df[(df["method"] != "no_treatment") & df.get("sim_ok", True)].copy()
    # The recovered combination table names the BO objective `final_regret`;
    # the simulation table carries an all-NaN `regret` placeholder.  Prefer the
    # real column so the method-ranking comparison has something to rank by.
    if df.get("regret") is None or df["regret"].notna().sum() == 0:
        if "final_regret" in df.columns:
            df["regret"] = df["final_regret"]
    print(f"{len(df)} simulated combinations, {df['cell_line'].nunique()} cell lines, "
          f"{df['method'].nunique()} methods")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1. monotonicity
    mono = monotonicity_table(df)
    mono.to_csv(args.outdir / "monotonicity_vs_invitro.csv", index=False)
    print("\n=== Monotonicity vs in vitro percent growth (|rho| ascending) ===")
    print(mono[["readout", "spearman_vs_invitro_pooled",
                "spearman_within_cellline_mean", "n_obs"]].round(4).to_string(index=False))

    # 1b. the published ODE, as the monotone control
    if args.with_ode_control:
        u = df["percent_growth"].dropna().unique()
        ode = pd.DataFrame({"percent_growth": u,
                            "ode_log10_red": appendix_v_ode(u)})
        ode.to_csv(args.outdir / "appendix_v_ode_control.csv", index=False)
        r, p = spearmanr(ode["percent_growth"], ode["ode_log10_red"])
        print(f"\nAppendix-V ODE control: Spearman(percent_growth, log10_red) "
              f"= {r:.4f} (p={p:.2e}) over {len(ode)} unique combinations")

    # 2/3/4 for each outcome readout
    rr_rows, rank_frames, wil_frames = [], [], []
    for ro in OUTCOME_READOUTS:
        if ro not in df.columns or df[ro].notna().sum() < 20:
            continue
        rr = pairwise_reranking(df, ro)
        rr.pop("discordant_records").to_csv(
            args.outdir / f"discordant_pairs_{ro}.csv", index=False)
        rr_rows.append(rr)
        rank_frames.append(method_ranking(df, ro))
        wil_frames.append(savitar_vs_baselines(df, ro))

    rr_df = pd.DataFrame(rr_rows)
    rr_df.to_csv(args.outdir / "reranking_rates.csv", index=False)
    print("\n=== Pairwise re-ranking vs in vitro ordering ===")
    print(rr_df.round(4).to_string(index=False))

    ranks = pd.concat([f for f in rank_frames if len(f)], ignore_index=True)
    ranks.to_csv(args.outdir / "method_rank_agreement.csv", index=False)
    print("\n=== Method-ranking agreement (Kendall tau, mean over cell lines) ===")
    print(ranks.groupby("readout")["kendall_tau"].agg(["mean", "std", "count"])
          .round(4).to_string())

    wil = pd.concat([f for f in wil_frames if len(f)], ignore_index=True)
    wil.to_csv(args.outdir / "savitar_vs_baselines.csv", index=False)
    print(f"\n=== Savitar vs baselines ({args.primary}) ===")
    sel = wil[wil["readout"] == args.primary]
    if len(sel):
        print(sel.round(5).to_string(index=False))
    print(f"\nwrote analysis tables to {args.outdir}")


if __name__ == "__main__":
    main()
