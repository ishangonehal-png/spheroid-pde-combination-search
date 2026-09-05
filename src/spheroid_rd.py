"""Radially-symmetric avascular tumour-spheroid reaction-diffusion solver.

Purpose
-------
The Appendix-V bridge in the Savitar manuscript maps an in vitro percent-growth
readout to a 30-day tumour trajectory through a well-mixed Gompertz ODE.  That
map is *monotone*: a combination with lower in vitro viability always yields a
lower final burden, so the ODE cannot re-rank the BO methods and adds no
information beyond in vitro regret.

This module replaces the well-mixed compartment with a spatially resolved
spheroid.  Three mechanisms present here are absent from the ODE and can each
break monotonicity:

1. **Drug transport.**  Each drug diffuses inward from the spheroid surface and
   is consumed/degraded en route, so a potent-but-poorly-penetrating drug
   reaches the core at a fraction of its nominal concentration.
2. **Oxygen-gated proliferation.**  Oxygen is consumed by viable cells, giving
   the classic ~150-200 um viable rim.  Cells below a hypoxic threshold cycle
   slowly, and S-phase-specific kill scales with proliferation, so hypoxic cells
   are intrinsically refractory.
3. **Geometry.**  Two drugs with different diffusivities overlap only over part
   of the radius, so a combination that is synergistic in a well plate can be
   effectively sequential in space.

Model
-----
State variables on r in [0, R], all functions of (r, t):

    n(r,t)  viable tumour-cell density, normalised to carrying capacity  [-]
    c_O(r,t) oxygen concentration, normalised to the boundary value      [-]
    c_j(r,t) concentration of drug j = 1, 2, in units of its own EC50    [-]

Governing equations (radial Laplacian in spherical coordinates):

    dn/dt   = D_n Lap n + rho * g(c_O) * n * (1 - n) - kill(n, c, c_O) * n
    dc_O/dt = D_O Lap c_O - q_O * n * c_O / (c_O + K_O)
    dc_j/dt = D_j Lap c_j - lambda_j * c_j - u_j * n * c_j / (c_j + K_u)

with Lap f = f_rr + (2/r) f_r, no-flux at r = 0 by symmetry, and Dirichlet
boundary conditions at r = R (well-stirred medium / perfused surface).

Proliferation gate and hypoxia-modulated kill:

    g(c_O)      = c_O / (c_O + K_g)                       Monod-type
    E_j(c_j)    = E_max,j * c_j^h_j / (c_j^h_j + 1)       Hill, c in EC50 units
    kill_total  = (sum_j E_j) * [ (1 - phi) + phi * g(c_O) ]

phi in [0, 1] interpolates between a cycle-independent cytotoxic (phi = 0) and a
fully proliferation-dependent agent (phi = 1); the bracket is the hypoxia
resistance factor.  Drug effects combine additively at the kill-rate level,
which is Bliss independence in the same sense used by the manuscript's
Appendix-V equal-log-split.

Numerics
--------
Method of lines on a uniform radial grid with second-order central differences,
integrated with `scipy.integrate.solve_ivp` (LSODA: stiff/non-stiff switching).
The r = 0 node uses the l'Hopital limit Lap f -> 3 f_rr, which preserves
second-order accuracy at the origin.

Verification (see `verify_solver` and `scripts/33_verify_spheroid_solver.py`):
  * Fisher-KPP planar travelling-wave speed  v -> 2 sqrt(D rho)
  * steady-state oxygen penetration depth against the analytic linear-consumption
    solution c_O(r) = (R/r) sinh(r/L) / sinh(R/L),  L = sqrt(D_O/(q_O/K_O))
  * spatial and temporal mesh-refinement convergence

Units.  Length um, time days, concentrations dimensionless (drugs in multiples
of their own EC50, oxygen relative to the boundary value, cells relative to
carrying capacity).  Diffusivities are therefore um^2/day.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "SpheroidParams",
    "DrugSpec",
    "SimulationResult",
    "simulate_spheroid",
    "hill_params_to_drugspec",
    "fisher_kpp_wave_speed",
    "oxygen_analytic_profile",
    "verify_solver",
]


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

@dataclass
class DrugSpec:
    """One drug's transport and pharmacodynamic description.

    Concentrations are expressed in multiples of the drug's own EC50, so the
    Hill term is ``c^h / (c^h + 1)``.  ``surface_conc`` is therefore literally
    "how many EC50 multiples are applied at the spheroid surface", which is the
    natural way to carry an in vitro dose into the spatial model.

    Attributes
    ----------
    name : str
        Label (e.g. the NSC id).
    emax : float
        Maximum fractional kill rate [1/day] at saturating concentration.
    hill : float
        Hill coefficient of the concentration-effect relation.
    diffusivity : float
        Drug diffusivity within tumour tissue [um^2/day].
    surface_conc : float
        Applied concentration at r = R, in EC50 multiples.
    decay : float
        First-order loss (metabolism/degradation) [1/day].
    uptake : float
        Maximum cell-mediated uptake/binding rate [1/day], saturating with
        half-constant ``uptake_km``.  This is what makes a drug "consumed"
        as it penetrates, producing a genuine penetration front.
    uptake_km : float
        Half-saturation constant for uptake, in EC50 multiples.
    cycle_dependence : float
        phi in [0, 1].  0 = kills regardless of proliferative state,
        1 = kill scales with the oxygen-dependent proliferation gate.
    """

    name: str
    emax: float
    hill: float = 1.0
    diffusivity: float = 5.0e4
    surface_conc: float = 1.0
    decay: float = 0.0
    uptake: float = 0.30
    uptake_km: float = 1.0
    cycle_dependence: float = 0.5

    def effect(self, conc: np.ndarray) -> np.ndarray:
        """Hill kill rate [1/day] at concentration ``conc`` (EC50 multiples)."""
        c = np.clip(conc, 0.0, None)
        ch = c ** self.hill
        return self.emax * ch / (ch + 1.0)


@dataclass
class SpheroidParams:
    """Tumour, oxygen and numerical parameters.

    Defaults are order-of-magnitude values for an avascular spheroid drawn from
    the standard 3D-culture literature; every one of them is swept in the
    sensitivity analysis rather than being treated as known.
    """

    # geometry / discretisation
    radius: float = 400.0          # spheroid radius [um]
    n_r: int = 160                 # radial grid points
    t_end: float = 14.0            # horizon [day]
    n_t: int = 141                 # output time points

    # tumour cells
    rho: float = 0.60              # max proliferation rate [1/day]
    cell_diffusivity: float = 2.0e2  # random cell motility [um^2/day]
    n0_core: float = 0.90          # initial density (uniform) [-]

    # oxygen
    # q_O is CALIBRATED, not assumed: with D_O = 1.7e5 um^2/day and K_O = 0.05,
    # q_O = 15 /day puts the viable (non-hypoxic) rim of a 400 um spheroid at
    # ~170 um, inside the 150-200 um oxygen-diffusion limit that is the standard
    # observation for multicellular tumour spheroids.  q_O is swept in the
    # sensitivity analysis; see scripts/35_spheroid_sensitivity.py.
    oxygen_diffusivity: float = 1.7e5   # [um^2/day]
    oxygen_consumption: float = 15.0    # max uptake q_O [1/day], calibrated
    oxygen_km: float = 0.05             # Michaelis constant K_O [-]
    prolif_km: float = 0.10             # K_g in the proliferation gate [-]
    hypoxia_threshold: float = 0.10     # c_O below this counts as hypoxic [-]
    necrosis_threshold: float = 0.02    # c_O below this counts as necrotic [-]
    # Severely hypoxic cells do not merely stop cycling, they die.  Without this
    # term the model has no necrotic core: core cells would sit at their initial
    # density indefinitely, which contradicts the defining structure of a
    # spheroid (proliferating rim / quiescent shell / necrotic centre) and
    # removes the reservoir of drug-refractory cells that drives regrowth.
    # The rate is a smooth switch on oxygen, active below `necrosis_threshold`.
    hypoxic_death: float = 0.25         # max hypoxia-induced death rate [1/day]
    hypoxic_death_km: float = 0.01      # sharpness of the switch [-]

    # dosing
    dose_times: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)
    dose_duration: float = 0.25    # each dose held at the surface this long [day]

    # numerics
    rtol: float = 1e-6
    atol: float = 1e-9
    method: str = "LSODA"

    def grid(self) -> np.ndarray:
        """Radial grid, excluding r=0 singularity handling done in the RHS."""
        return np.linspace(0.0, self.radius, self.n_r)


# --------------------------------------------------------------------------- #
# Curve conversion: in vitro Hill fit -> spatial drug spec
# --------------------------------------------------------------------------- #

def hill_params_to_drugspec(
    *,
    name: str,
    e0: float,
    emax_pct: float,
    log_ec50: float,
    slope: float,
    applied_conc_M: float,
    t_invitro_days: float = 2.0,
    diffusivity: float = 5.0e4,
    cycle_dependence: float = 0.5,
    max_kill_rate: float = 5.0,
    **kwargs,
) -> DrugSpec:
    """Convert a repo Hill fit + an applied dose into a :class:`DrugSpec`.

    The repo stores, per (drug, cell line), a Hill fit in *percent-growth* space
    with parameters ``e0`` (upper asymptote, ~100), ``emax`` (lower asymptote, the
    maximal achievable percent growth), ``log_ec50`` (log10 M) and ``slope``.

    We convert the *asymptotic* effect to a kill rate with the same
    exponential-kill assumption the manuscript uses in Appendix V,

        viability v = emax_pct / 100 ,   gamma = -ln(v) / t_invitro ,

    so a drug that drives percent-growth to 0 has a large kill rate and one that
    plateaus at 80% growth has a small one.  The *spatial* concentration is then
    carried in EC50 multiples, ``surface_conc = applied_conc_M / EC50``, which is
    exactly the quantity the Hill term needs and is dimensionless, so it can be
    transported by the PDE without further unit bookkeeping.

    Negative percent-growth (net cell loss below the time-zero seeding density)
    is clipped to a small positive viability, matching ``src/ode_pkpd.py``.
    """
    v = float(emax_pct) / 100.0
    v = min(max(v, 1e-3), 1.0)          # clip as in ode_pkpd.gamma_from_in_vitro_viability
    gamma = -np.log(v) / float(t_invitro_days)
    gamma = float(np.clip(gamma, 0.0, max_kill_rate))

    ec50_M = 10.0 ** float(log_ec50)
    surface_conc = float(applied_conc_M) / ec50_M if ec50_M > 0 else 0.0

    # Hill slopes from noisy 3-point fits can be pathological; keep them sane.
    h = float(np.clip(abs(slope), 0.3, 4.0))

    return DrugSpec(
        name=name,
        emax=gamma,
        hill=h,
        diffusivity=diffusivity,
        surface_conc=float(np.clip(surface_conc, 0.0, 1e4)),
        cycle_dependence=cycle_dependence,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Spatial operators
# --------------------------------------------------------------------------- #

def _radial_laplacian(f: np.ndarray, dr: float, *, outer: str = "dirichlet",
                      f_boundary: float = 0.0) -> np.ndarray:
    """Spherical Laplacian f_rr + (2/r) f_r on a grid whose last node is r = R.

    Symmetry at the origin gives the ghost node f[-1] = f[1] (no flux), and the
    singular (2/r) f_r term is replaced by its l'Hopital limit Lap f -> 3 f_rr.

    The last grid node lies exactly on r = R, so the two outer conditions are:

    ``outer="dirichlet"``
        f(R) is *prescribed*.  The node is pinned algebraically by the caller
        (its time derivative is set to zero), and the Laplacian returned for it
        is irrelevant, so we return 0 there.  Pinning the node - rather than
        evolving it against a ghost - is what preserves second-order accuracy.
    ``outer="noflux"``
        f_r(R) = 0, imposed with the mirror ghost f[n] = f[n-2].

    Both interior stencils are second-order accurate.
    """
    n = f.size
    lap = np.zeros(n)

    fm = np.empty(n + 2)
    fm[0] = f[1]                 # symmetry ghost at r = -dr
    fm[1:-1] = f
    if outer == "noflux":
        fm[-1] = f[n - 2]        # mirror ghost => f_r(R) = 0
        last = n                 # evolve the boundary node too
    else:
        fm[-1] = f_boundary      # value beyond R is irrelevant; node is pinned
        last = n - 1             # do NOT evolve the pinned Dirichlet node

    d2 = (fm[2:] - 2.0 * fm[1:-1] + fm[:-2]) / (dr * dr)
    d1 = (fm[2:] - fm[:-2]) / (2.0 * dr)

    r = np.arange(n) * dr
    lap[0] = 3.0 * d2[0]                       # l'Hopital limit at the origin
    lap[1:last] = d2[1:last] + 2.0 * d1[1:last] / r[1:last]
    return lap


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class SimulationResult:
    t: np.ndarray                    # (n_t,) [day]
    r: np.ndarray                    # (n_r,) [um]
    n: np.ndarray                    # (n_t, n_r) viable density [-]
    oxygen: np.ndarray               # (n_t, n_r) [-]
    drugs: np.ndarray                # (n_drug, n_t, n_r) [EC50 multiples]
    params: SpheroidParams
    drug_specs: list[DrugSpec]
    success: bool = True
    message: str = ""
    metadata: dict = field(default_factory=dict)

    # -- integral quantities ------------------------------------------------ #
    def burden(self) -> np.ndarray:
        """Total viable cell number (4*pi*int n r^2 dr) vs time, in um^3 units."""
        w = 4.0 * np.pi * self.r ** 2
        return np.trapezoid(self.n * w, self.r, axis=1)

    def volume(self) -> float:
        return 4.0 / 3.0 * np.pi * self.params.radius ** 3


# --------------------------------------------------------------------------- #
# Core solver
# --------------------------------------------------------------------------- #

def simulate_spheroid(
    drugs: Sequence[DrugSpec],
    params: SpheroidParams | None = None,
    *,
    n_init: np.ndarray | None = None,
) -> SimulationResult:
    """Integrate the coupled spheroid RD system.

    Parameters
    ----------
    drugs : sequence of DrugSpec
        Zero, one or two (or more) drugs.  An empty sequence gives the
        no-treatment control.
    params : SpheroidParams, optional
    n_init : array, optional
        Initial radial cell-density profile; defaults to uniform ``n0_core``.
    """
    p = params or SpheroidParams()
    r = p.grid()
    dr = r[1] - r[0]
    nr = p.n_r
    nd = len(drugs)

    # --- initial conditions ---
    n_0 = np.full(nr, p.n0_core) if n_init is None else np.asarray(n_init, float).copy()
    # oxygen starts at its drug-free steady state to avoid a startup transient
    ox_0 = _oxygen_steady_state(n_0, p, dr)
    dr_0 = np.zeros((nd, nr))

    # The r = R nodes of the oxygen and drug fields are PRESCRIBED, not
    # integrated: their time derivative is held at zero and they keep whatever
    # value they are initialised with.  Oxygen is pinned at 1 for all time.
    # Drug surface values switch with the dosing schedule, so they are written
    # back onto the stored solution after integration (see below); the interior
    # physics is unaffected because the RHS always applies the correct
    # instantaneous boundary value when building the Laplacian.
    ox_0[-1] = 1.0

    # Interleave the state by node - [n, O2, c1, c2] at r0, then at r1, ... -
    # so that the Jacobian is BANDED (each node couples only to its two
    # neighbours).  With nf fields the half-bandwidth is 2*nf - 1, which lets
    # LSODA do a banded LU instead of a dense 640x640 one.  This is worth
    # roughly two orders of magnitude in wall time for the grids we run.
    nf = 2 + nd
    y0 = np.empty(nr * nf)
    y0[0::nf] = n_0
    y0[1::nf] = ox_0
    for j in range(nd):
        y0[2 + j::nf] = dr_0[j]

    d_diff = np.array([d.diffusivity for d in drugs]) if nd else np.zeros(0)
    d_decay = np.array([d.decay for d in drugs]) if nd else np.zeros(0)
    d_uptake = np.array([d.uptake for d in drugs]) if nd else np.zeros(0)
    d_km = np.array([d.uptake_km for d in drugs]) if nd else np.zeros(0)
    d_surf = np.array([d.surface_conc for d in drugs]) if nd else np.zeros(0)
    d_phi = np.array([d.cycle_dependence for d in drugs]) if nd else np.zeros(0)

    dose_times = np.asarray(p.dose_times, float)

    def _surface_on(t: float) -> np.ndarray:
        """Surface drug concentration at time t (bolus held for dose_duration)."""
        if nd == 0:
            return np.zeros(0)
        on = np.any((t >= dose_times) & (t < dose_times + p.dose_duration))
        return d_surf if on else np.zeros(nd)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        n = np.clip(y[0::nf], 0.0, None)
        ox = np.clip(y[1::nf], 0.0, None)
        cd = np.empty((nd, nr))
        for j in range(nd):
            cd[j] = np.clip(y[2 + j::nf], 0.0, None)

        surf = _surface_on(t)

        # --- oxygen: Dirichlet c_O(R) = 1, Michaelis-Menten consumption -------
        ox[-1] = 1.0                                     # enforce the BC on the state
        lap_ox = _radial_laplacian(ox, dr, outer="dirichlet", f_boundary=1.0)
        consume = p.oxygen_consumption * n * ox / (ox + p.oxygen_km)
        d_ox = p.oxygen_diffusivity * lap_ox - consume
        d_ox[-1] = 0.0                                   # pinned node does not evolve

        # proliferation gate
        gate = ox / (ox + p.prolif_km)

        # --- drugs: Dirichlet at the surface, decay + saturable uptake --------
        d_cd = np.zeros((nd, nr))
        kill = np.zeros(nr)
        for j in range(nd):
            cd[j, -1] = surf[j]                          # enforce surface BC
            lap_j = _radial_laplacian(cd[j], dr, outer="dirichlet",
                                      f_boundary=surf[j])
            uptake = d_uptake[j] * n * cd[j] / (cd[j] + d_km[j])
            d_cd[j] = d_diff[j] * lap_j - d_decay[j] * cd[j] - uptake
            d_cd[j, -1] = 0.0                            # pinned

            e_j = drugs[j].effect(cd[j])
            # hypoxia resistance: cycle-dependent fraction scales with the gate
            kill += e_j * ((1.0 - d_phi[j]) + d_phi[j] * gate)

        # --- cells: no flux at r = R (cells do not leave the spheroid) --------
        # Hypoxia-induced death: a smooth switch that turns on as oxygen falls
        # below the necrosis threshold, producing the necrotic core.
        necro = p.hypoxic_death * p.hypoxic_death_km / (ox + p.hypoxic_death_km)
        lap_n = _radial_laplacian(n, dr, outer="noflux")
        d_n = (
            p.cell_diffusivity * lap_n
            + p.rho * gate * n * (1.0 - n)
            - kill * n
            - necro * n
        )

        dy = np.empty_like(y)
        dy[0::nf] = d_n
        dy[1::nf] = d_ox
        for j in range(nd):
            dy[2 + j::nf] = d_cd[j]
        return dy

    t_eval = np.linspace(0.0, p.t_end, p.n_t)
    # Banded Jacobian: with the node-interleaved layout each unknown couples
    # only to fields at the same node and the two adjacent nodes.
    band = 2 * nf - 1
    sol = solve_ivp(
        rhs, (0.0, p.t_end), y0,
        t_eval=t_eval, method=p.method, rtol=p.rtol, atol=p.atol,
        lband=band, uband=band,
    )

    Y = sol.y.T                              # (n_t, n_state)
    n_hist = np.clip(Y[:, 0::nf], 0.0, None)
    ox_hist = np.clip(Y[:, 1::nf], 0.0, None)
    ox_hist[:, -1] = 1.0                     # prescribed surface value
    if nd:
        dr_hist = np.empty((nd, len(sol.t), nr))
        for j in range(nd):
            dr_hist[j] = np.clip(Y[:, 2 + j::nf], 0.0, None)
            # restore the prescribed surface trace (the node is not integrated)
            dr_hist[j][:, -1] = [
                d_surf[j] if np.any((tt >= dose_times) & (tt < dose_times + p.dose_duration))
                else 0.0
                for tt in sol.t
            ]
    else:
        dr_hist = np.zeros((0, len(sol.t), nr))

    return SimulationResult(
        t=sol.t, r=r, n=n_hist, oxygen=ox_hist, drugs=dr_hist,
        params=p, drug_specs=list(drugs),
        success=bool(sol.success), message=str(sol.message),
    )


def _oxygen_steady_state(n: np.ndarray, p: SpheroidParams, dr: float,
                         n_iter: int = 4000) -> np.ndarray:
    """Relax the oxygen field to steady state for a frozen cell profile."""
    ox = np.ones_like(n)
    dt = 0.2 * dr * dr / p.oxygen_diffusivity
    for _ in range(n_iter):
        ox[-1] = 1.0
        lap = _radial_laplacian(ox, dr, outer="dirichlet", f_boundary=1.0)
        upd = dt * (p.oxygen_diffusivity * lap
                    - p.oxygen_consumption * n * ox / (ox + p.oxygen_km))
        upd[-1] = 0.0
        ox = ox + upd
        np.clip(ox, 0.0, 1.0, out=ox)
    ox[-1] = 1.0
    return ox


# --------------------------------------------------------------------------- #
# Analytic references for verification
# --------------------------------------------------------------------------- #

def fisher_kpp_wave_speed(D: float, rho: float) -> float:
    """Asymptotic planar Fisher-KPP front speed 2 sqrt(D rho) [um/day]."""
    return 2.0 * np.sqrt(D * rho)


def oxygen_analytic_profile(r: np.ndarray, R: float, D: float,
                            q: float, km: float) -> np.ndarray:
    """Steady oxygen profile for *linear* consumption q*n/km with n = 1.

    In the low-concentration limit (c << km) Michaelis-Menten consumption
    linearises to (q/km) c, and the spherical steady state has the closed form

        c(r) = (R/r) * sinh(r/L) / sinh(R/L),   L = sqrt(D km / q).

    This is the reference the solver is checked against.
    """
    L = np.sqrt(D * km / q)
    out = np.empty_like(r)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (R / np.maximum(r, 1e-12)) * np.sinh(r / L) / np.sinh(R / L)
    out[r < 1e-12] = (R / L) / np.sinh(R / L)     # l'Hopital limit at r -> 0
    return out


def verify_solver(verbose: bool = True) -> dict:
    """Run the analytic and convergence checks; returns a dict of errors."""
    out: dict[str, float] = {}

    # -- 1. oxygen steady state vs analytic (linear-consumption regime) -------
    # Use km >> c so MM consumption is effectively linear.
    p = SpheroidParams(radius=300.0, n_r=400, oxygen_km=50.0,
                       oxygen_consumption=8.0, oxygen_diffusivity=1.7e5)
    r = p.grid()
    dr = r[1] - r[0]
    num = _oxygen_steady_state(np.ones_like(r), p, dr, n_iter=60000)
    ana = oxygen_analytic_profile(r, p.radius, p.oxygen_diffusivity,
                                  p.oxygen_consumption, p.oxygen_km)
    out["oxygen_rel_l2"] = float(np.linalg.norm(num - ana) / np.linalg.norm(ana))

    # -- 2. Fisher-KPP travelling-wave speed ---------------------------------
    # Planar limit.  Two things matter for hitting 2 sqrt(D rho):
    #   (i)  the front must be far from the origin so the 2/r curvature term is
    #        negligible (curvature slows a radially expanding front by ~2D/r);
    #   (ii) the speed must be measured in the ASYMPTOTIC regime, after the
    #        initial condition has relaxed onto the travelling-wave profile.
    # We therefore seed a large plug, run long, and fit only the late window,
    # tracking the front with sub-grid linear interpolation of the n = 0.5 crossing.
    D, rho = 1.0e4, 0.8
    v_exact = fisher_kpp_wave_speed(D, rho)          # 178.9 um/day
    pf = SpheroidParams(radius=12000.0, n_r=3000, t_end=40.0, n_t=81,
                        rho=rho, cell_diffusivity=D,
                        oxygen_consumption=0.0, prolif_km=1e-12,
                        oxygen_km=1.0, dose_times=(), n0_core=0.0)
    rr = pf.grid()
    n0 = np.where(rr < 6000.0, 1.0, 0.0)             # front starts far from r=0
    res = simulate_spheroid([], pf, n_init=n0)

    def _front(nprof):
        """Sub-grid r where n crosses 0.5, by linear interpolation."""
        idx = np.where(nprof < 0.5)[0]
        if idx.size == 0 or idx[0] == 0:
            return np.nan
        i = idx[0]
        n1, n2 = nprof[i - 1], nprof[i]
        w = (n1 - 0.5) / (n1 - n2) if n1 != n2 else 0.0
        return rr[i - 1] + w * (rr[i] - rr[i - 1])

    fronts = np.array([_front(res.n[i]) for i in range(len(res.t))])
    ok = np.isfinite(fronts) & (fronts < 0.85 * pf.radius) & (res.t > 20.0)
    speed = np.polyfit(res.t[ok], fronts[ok], 1)[0] if ok.sum() > 3 else np.nan
    out["kpp_speed_numeric"] = float(speed)
    out["kpp_speed_analytic"] = float(v_exact)
    out["kpp_rel_err"] = float(abs(speed - v_exact) / v_exact)

    # -- 3. spatial mesh refinement ------------------------------------------
    drug = DrugSpec(name="test", emax=1.0, hill=1.0, surface_conc=2.0)
    burdens = {}
    for nr in (80, 160, 320):
        pp = SpheroidParams(n_r=nr, t_end=6.0, n_t=13)
        rres = simulate_spheroid([drug], pp)
        burdens[nr] = float(rres.burden()[-1])
    out["mesh_80_160_rel"] = abs(burdens[80] - burdens[160]) / abs(burdens[160])
    out["mesh_160_320_rel"] = abs(burdens[160] - burdens[320]) / abs(burdens[320])
    # second-order scheme => halving dr should cut the error ~4x
    out["mesh_convergence_ratio"] = (out["mesh_80_160_rel"]
                                     / max(out["mesh_160_320_rel"], 1e-30))

    if verbose:
        for k, v in out.items():
            print(f"{k:28s} {v:.6g}")
    return out


if __name__ == "__main__":
    verify_solver()
