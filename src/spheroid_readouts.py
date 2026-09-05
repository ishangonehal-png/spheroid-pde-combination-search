"""Outcome-linked readouts extracted from a spheroid RD simulation.

Each function here computes one quantity from a :class:`~src.spheroid_rd.SimulationResult`.
The quantities are chosen because they are (a) computable from an avascular
spheroid model calibrated on in vitro Hill curves and (b) analogues of measures
with reported associations to patient outcome.  The mapping to the clinical
literature - and, importantly, which of these are outcome-*validated* versus
merely mechanistically motivated - is documented in the accompanying survey
artifact; nothing here should be read as a validated clinical prediction.

Readout groups
--------------
Burden / TGI-style
    ``log_kill``          log10 reduction in viable burden vs no-treatment control
    ``nadir_burden``      minimum burden reached, relative to t=0
    ``time_to_nadir``     day at which the minimum occurs
    ``depth_of_response`` fractional shrinkage at nadir
    ``kg_regrowth``       post-nadir exponential regrowth rate [1/day]
    ``time_to_regrowth``  days from nadir until burden returns to its t=0 value

Spatial / microenvironmental
    ``hypoxic_fraction``      volume fraction below the hypoxic O2 threshold
    ``necrotic_fraction``     volume fraction below the necrotic threshold
    ``viable_rim_thickness``  thickness of the oxygenated proliferating rim [um]
    ``penetration_depth``     depth at which a drug falls to 1/e of its surface value
    ``core_sparing``          ratio of core to rim kill; the direct measure of the
                              spatial effect the well-mixed ODE cannot represent
    ``residual_core_burden``  viable burden remaining inside the hypoxic core
"""

from __future__ import annotations

import numpy as np

from src.spheroid_rd import SimulationResult

__all__ = ["extract_readouts", "READOUT_COLUMNS"]


def _shell_weights(r: np.ndarray) -> np.ndarray:
    """4 pi r^2 weights for integrating a radial profile over the sphere."""
    return 4.0 * np.pi * r ** 2


def _integrate(profile: np.ndarray, r: np.ndarray) -> float:
    return float(np.trapezoid(profile * _shell_weights(r), r))


def extract_readouts(res: SimulationResult,
                     control: SimulationResult | None = None) -> dict:
    """Compute all readouts for one simulation.

    Parameters
    ----------
    res : SimulationResult
        The treated simulation.
    control : SimulationResult, optional
        Matched no-treatment run, used for the log-kill and growth-delay
        readouts.  If omitted those fields are NaN.
    """
    out: dict[str, float] = {}
    r, t = res.r, res.t
    burden = res.burden()
    b0 = burden[0]

    # ---------------- burden / TGI-style ---------------------------------- #
    out["burden_initial"] = float(b0)
    out["burden_final"] = float(burden[-1])
    out["burden_ratio_final"] = float(burden[-1] / b0) if b0 > 0 else np.nan

    i_nadir = int(np.argmin(burden))
    out["nadir_burden"] = float(burden[i_nadir] / b0) if b0 > 0 else np.nan
    out["time_to_nadir"] = float(t[i_nadir])
    out["depth_of_response"] = float(1.0 - burden[i_nadir] / b0) if b0 > 0 else np.nan

    # post-nadir regrowth rate: slope of log burden after the nadir
    if i_nadir < len(t) - 3:
        seg_t = t[i_nadir:]
        seg_b = np.maximum(burden[i_nadir:], 1e-30)
        out["kg_regrowth"] = float(np.polyfit(seg_t, np.log(seg_b), 1)[0])
    else:
        out["kg_regrowth"] = np.nan

    # time from nadir back up to the starting burden ("time to regrowth")
    post = burden[i_nadir:]
    back = np.where(post >= b0)[0]
    out["time_to_regrowth"] = float(t[i_nadir + back[0]] - t[i_nadir]) if back.size else np.nan

    if control is not None:
        cb = control.burden()
        # End-of-horizon log-kill saturates: once dosing stops, both arms relax
        # to the same carrying-capacity attractor, so burden[-1] carries almost
        # no information about the treatment.  We therefore report three
        # complementary kill measures and use the non-saturating ones downstream.
        out["log_kill"] = float(np.log10(max(cb[-1], 1e-30) / max(burden[-1], 1e-30)))
        # (a) at the treated arm's nadir, against the control at that same time
        out["log_kill_nadir"] = float(
            np.log10(max(cb[i_nadir], 1e-30) / max(burden[i_nadir], 1e-30)))
        # (b) trajectory-integrated: the ratio of areas under the burden curves.
        #     This is the spheroid analogue of a TGI AUC ratio and does not
        #     saturate, because it weights the whole response, not the endpoint.
        auc_t = float(np.trapezoid(burden, t))
        auc_c = float(np.trapezoid(cb, t))
        out["burden_auc_ratio"] = float(auc_t / auc_c) if auc_c > 0 else np.nan
        out["log_kill_auc"] = float(np.log10(max(auc_c, 1e-30) / max(auc_t, 1e-30)))
        out["growth_delay"] = float(_growth_delay(t, burden, cb))
    else:
        for k in ("log_kill", "log_kill_nadir", "burden_auc_ratio",
                  "log_kill_auc", "growth_delay"):
            out[k] = np.nan

    # ---------------- clinical TGI-model metrics --------------------------- #
    # The literature survey identifies the Stein/Claret two-phase (biexponential)
    # decomposition as the form whose growth constant KG is validated against
    # overall survival across several tumour types, while the kill constant KS
    # is explicitly NOT (Stein 2008: r = -0.72 for log g vs -0.218 for log d).
    # We therefore fit that exact functional form to the simulated burden and
    # report KG, KS, TTG and ETS, treating KG/TTG as the outcome-linked readouts
    # and KS as a diagnostic that should track in vitro potency.
    tgi = _fit_biexponential(t, burden)
    out.update(tgi)

    # Early tumour shrinkage: relative burden reduction at a fixed early time,
    # the clinical analogue being the week 6-8 assessment.  We take one quarter
    # of the horizon, which is the same position in the treatment course.
    t_early = t[0] + 0.25 * (t[-1] - t[0])
    b_early = float(np.interp(t_early, t, burden))
    out["ets_early"] = float(1.0 - b_early / b0) if b0 > 0 else np.nan
    out["t_early"] = float(t_early)

    # ---------------- spatial / microenvironmental ------------------------- #
    p = res.params
    w = _shell_weights(r)
    total_vol = float(np.trapezoid(w, r))

    ox_final = res.oxygen[-1]
    n_final = res.n[-1]

    hyp_mask = ox_final < p.hypoxia_threshold
    nec_mask = ox_final < p.necrosis_threshold
    out["hypoxic_fraction"] = float(np.trapezoid(hyp_mask * w, r) / total_vol)
    out["necrotic_fraction"] = float(np.trapezoid(nec_mask * w, r) / total_vol)

    # viable rim: outermost contiguous region above the hypoxic threshold
    viable = ~hyp_mask
    if viable.any():
        idx = np.where(viable)[0]
        # walk inward from the surface while still oxygenated
        k = len(r) - 1
        while k >= 0 and viable[k]:
            k -= 1
        out["viable_rim_thickness"] = float(r[-1] - r[max(k, 0)])
    else:
        out["viable_rim_thickness"] = 0.0

    # --- drug penetration -------------------------------------------------- #
    # Penetration must be judged on TIME-INTEGRATED exposure (AUC), not on a
    # single snapshot.  A snapshot at peak surface concentration is taken at the
    # instant the bolus is applied, when nothing has diffused inward yet, and
    # would report ~zero penetration for every drug regardless of its
    # diffusivity.  AUC(r) = int c(r,t) dt is also the pharmacologically
    # meaningful quantity: it is what the local cells actually experience.
    for j in range(res.drugs.shape[0]):
        prof = res.drugs[j]                     # (n_t, n_r)
        auc = np.trapezoid(prof, t, axis=0)     # (n_r,) exposure per radius
        auc_surf = auc[-1]
        if auc_surf > 0:
            out[f"auc_surface_drug{j+1}"] = float(auc_surf)
            out[f"auc_core_drug{j+1}"] = float(auc[0])
            out[f"core_to_surface_ratio_drug{j+1}"] = float(auc[0] / auc_surf)
            # depth inward from the surface at which exposure falls to 1/e
            thresh = auc_surf / np.e
            below = np.where(auc < thresh)[0]
            if below.size:
                # deepest radius still above threshold, interpolated
                i = below[-1]
                if i + 1 < len(r):
                    a1, a2 = auc[i], auc[i + 1]
                    frac = (thresh - a1) / (a2 - a1) if a2 != a1 else 0.0
                    r_cross = r[i] + frac * (r[i + 1] - r[i])
                else:
                    r_cross = r[i]
                out[f"penetration_depth_drug{j+1}"] = float(r[-1] - r_cross)
            else:
                # never falls below 1/e => fully penetrated
                out[f"penetration_depth_drug{j+1}"] = float(r[-1])
        else:
            for key in ("auc_surface", "auc_core", "core_to_surface_ratio",
                        "penetration_depth"):
                out[f"{key}_drug{j+1}"] = np.nan
    for j in range(res.drugs.shape[0], 2):      # pad so the schema is stable
        for key in ("auc_surface", "auc_core", "core_to_surface_ratio",
                    "penetration_depth"):
            out[f"{key}_drug{j+1}"] = np.nan

    # Core sparing: kill achieved in the inner third vs the outer third,
    # each measured AGAINST THE MATCHED CONTROL at the same time rather than
    # against t = 0.  Referencing to the control is essential: the core loses
    # cells to hypoxic death whether or not it is treated, so a t=0 reference
    # would score that spontaneous necrosis as drug effect and report large
    # "kill" in the core of an untreated spheroid.
    # This regional contrast is the quantity a well-mixed ODE structurally
    # cannot produce, since it has no radius.
    inner = r <= r[-1] / 3.0
    outer = r >= 2.0 * r[-1] / 3.0
    shell = _shell_weights(r)

    def _regional_kill(mask):
        if control is None:
            return np.nan
        ref = np.trapezoid(control.n[-1][mask] * shell[mask], r[mask])
        got = np.trapezoid(n_final[mask] * shell[mask], r[mask])
        if ref <= 0:
            return np.nan
        return float(1.0 - got / ref)

    k_in, k_out = _regional_kill(inner), _regional_kill(outer)
    out["kill_fraction_core"] = k_in
    out["kill_fraction_rim"] = k_out
    out["core_sparing"] = float(k_out - k_in) if np.isfinite(k_in) and np.isfinite(k_out) else np.nan

    # viable burden left inside the hypoxic core at the end
    if hyp_mask.any():
        out["residual_core_burden"] = float(
            np.trapezoid(n_final[hyp_mask] * w[hyp_mask], r[hyp_mask]) / max(b0, 1e-30))
    else:
        out["residual_core_burden"] = 0.0

    out["solver_success"] = float(res.success)
    return out


def _fit_biexponential(t: np.ndarray, burden: np.ndarray) -> dict:
    """Fit the Stein/Claret two-phase TGI model to a simulated burden curve.

        N(t) / N(0) = exp(-KS t) + exp(KG t) - 1

    KS is the exponential kill (regression) constant and KG the regrowth
    constant.  The published clinical result is that KG predicts overall
    survival while KS does not, which matters here because KS is the constant a
    well-mixed in vitro potency measure most directly determines.

    Returns KG, KS, their fit quality, and the derived time-to-growth (TTG), the
    time at which the fitted trajectory turns from net shrinkage to net growth:

        d/dt = 0  =>  TTG = ln(KS/KG) / (KS + KG)
    """
    from scipy.optimize import curve_fit

    out: dict[str, float] = {
        "kg_biexp": np.nan, "ks_biexp": np.nan,
        "ttg_biexp": np.nan, "biexp_r2": np.nan,
    }
    b0 = burden[0]
    if b0 <= 0 or len(t) < 5:
        return out
    y = burden / b0

    def model(tt, ks, kg):
        return np.exp(-ks * tt) + np.exp(kg * tt) - 1.0

    try:
        popt, _ = curve_fit(
            model, t, y, p0=(0.3, 0.1),
            bounds=([0.0, 0.0], [20.0, 20.0]), maxfev=20000,
        )
        ks, kg = float(popt[0]), float(popt[1])
        pred = model(t, ks, kg)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out["ks_biexp"] = ks
        out["kg_biexp"] = kg
        out["biexp_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if ks > 0 and kg > 0:
            ttg = np.log(ks / kg) / (ks + kg)
            out["ttg_biexp"] = float(ttg) if np.isfinite(ttg) and ttg > 0 else 0.0
    except Exception:                                   # noqa: BLE001
        pass
    return out


def _growth_delay(t: np.ndarray, treated: np.ndarray, control: np.ndarray,
                  factor: float = 2.0) -> float:
    """Extra days the treated arm takes to reach ``factor`` x its starting burden."""
    def _cross(b):
        target = factor * b[0]
        idx = np.where(b >= target)[0]
        if idx.size == 0:
            return np.nan
        i = idx[0]
        if i == 0:
            return float(t[0])
        b1, b2 = b[i - 1], b[i]
        w = (target - b1) / (b2 - b1) if b2 != b1 else 0.0
        return float(t[i - 1] + w * (t[i] - t[i - 1]))

    tt, tc = _cross(treated), _cross(control)
    if np.isnan(tc):
        return np.nan
    if np.isnan(tt):
        return float(t[-1] - tc)          # never reached within the horizon
    return float(tt - tc)


READOUT_COLUMNS = [
    "burden_initial", "burden_final", "burden_ratio_final",
    "nadir_burden", "time_to_nadir", "depth_of_response",
    "kg_regrowth", "time_to_regrowth", "log_kill", "log_kill_nadir",
    "burden_auc_ratio", "log_kill_auc", "growth_delay",
    "kg_biexp", "ks_biexp", "ttg_biexp", "biexp_r2", "ets_early", "t_early",
    "hypoxic_fraction", "necrotic_fraction", "viable_rim_thickness",
    "penetration_depth_drug1", "core_to_surface_ratio_drug1",
    "auc_surface_drug1", "auc_core_drug1",
    "penetration_depth_drug2", "core_to_surface_ratio_drug2",
    "auc_surface_drug2", "auc_core_drug2",
    "kill_fraction_core", "kill_fraction_rim", "core_sparing",
    "residual_core_burden", "solver_success",
]
