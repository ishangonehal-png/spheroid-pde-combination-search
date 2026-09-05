#!/usr/bin/env python
"""Render simulated spheroids as 3-D volumes.

The solver is 1-D radial, but the model it represents is a genuinely
three-dimensional, spherically symmetric object: the radial profile n(r,t) IS the
3-D field n(x,y,z,t) with r = |x|.  Revolving the profile is therefore not an
artistic flourish -- it is the exact reconstruction of the simulated state, and
it makes visible the thing a well-mixed ODE cannot represent: a treated spheroid
that is dead at the rim and alive at the centre.

Each panel is a cutaway (half the volume clipped away) so the interior structure
-- viable rim, hypoxic shell, necrotic core -- is directly readable.

Usage
-----
    python scripts/37_render_spheroid_3d.py --out results/pde/fig3d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spheroid_rd import (  # noqa: E402
    DrugSpec,
    SpheroidParams,
    hill_params_to_drugspec,
    simulate_spheroid,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import imageio.v3 as iio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.volume_render import render_volume  # noqa: E402


# --------------------------------------------------------------------------- #
# Scenario construction
# --------------------------------------------------------------------------- #
def build_drugs(runner, fits: pd.DataFrame, cell_line: str,
                nsc1: int, nsc2: int, conc1: float, conc2: float) -> list[DrugSpec]:
    """Build DrugSpecs with the RUNNER's own builder.

    Re-implementing the Hill-fit -> DrugSpec conversion here would risk the
    rendered spheroid drifting from the simulated one; calling the same function
    the study used guarantees the picture shows the state that produced the
    numbers.
    """
    out = []
    for nsc, conc in ((nsc1, conc1), (nsc2, conc2)):
        row = fits[(fits.nsc == nsc) & (fits.cell_line == cell_line)]
        if row.empty:
            continue
        spec = runner.build_drugspec(int(nsc), float(conc),
                                     next(row.head(1).itertuples()), {})
        if spec is not None:
            out.append(spec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pde/fig3d")
    ap.add_argument("--cell-line", default="OVCAR-8")
    ap.add_argument("--t-end", type=float, default=21.0)
    ap.add_argument("--px", type=int, default=760)
    ap.add_argument("--matched-pair", default=None,
                    help="JSON with a cell_line + lo/hi combination matched in vitro")
    args = ap.parse_args()

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    fits = pd.read_parquet(ROOT / "results/almanac_hill_fits.parquet")
    sel = pd.read_csv(ROOT / "results/pde/bo_selected_combinations.csv")
    sel = sel[(sel.cell_line == args.cell_line) & (sel.row_kind == "bo_selection")]

    # Import the runner itself so the rendered spheroids use exactly the same
    # drug construction as the simulation study.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sim_runner", ROOT / "scripts" / "34_simulate_combinations.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    params = SpheroidParams(t_end=args.t_end, n_t=int(args.t_end * 5) + 1)

    scenarios = {}
    scenarios["control"] = simulate_spheroid([], params)

    if args.matched_pair:
        # Render two combinations that are MATCHED in vitro but far apart in the
        # spheroid.  Comparing two BO methods' picks would confound potency with
        # spatial behaviour -- different methods generally choose combinations of
        # different in vitro potency, so any difference in the render could just
        # be "one is a stronger drug".  Holding percent-growth fixed isolates the
        # spatial mechanism, which is the claim the figure exists to support.
        mp = json.loads(Path(args.matched_pair).read_text())
        for key in ("lo", "hi"):
            spec = mp[key]
            drugs = build_drugs(runner, fits, mp["cell_line"],
                                int(spec["nsc1"]), int(spec["nsc2"]),
                                float(spec["conc1_M"]), float(spec["conc2_M"]))
            if drugs:
                scenarios[key] = simulate_spheroid(drugs, params)
    else:
        for method in ["ours+zero", "tanimoto+zero"]:
            row = sel[(sel.method == method) & (sel.seed == 0)]
            if row.empty:
                continue
            row = row.iloc[0]
            drugs = build_drugs(runner, fits, args.cell_line, int(row.nsc1),
                                int(row.nsc2), float(row.conc1_M), float(row.conc2_M))
            if not drugs:
                continue
            scenarios[method] = simulate_spheroid(drugs, params)

    manifest: dict = {}
    for name, res in scenarios.items():
        tag = name.replace("+", "_")
        # Render at each arm's BURDEN NADIR, not at the end of the horizon.  By
        # day 21 every arm has relaxed back toward the same carrying-capacity
        # attractor, so end-state snapshots of treated and control look alike;
        # the nadir is where the treatment effect actually lives.
        b = res.burden()
        it = int(np.argmin(b)) if name != "control" else len(res.t) - 1
        for field_name, arr, cmap in [
            ("cells", res.n[it], plt.get_cmap("viridis")),
            ("oxygen", res.oxygen[it], plt.get_cmap("plasma")),
        ]:
            img = render_volume(
                res.r, np.asarray(arr, dtype=float),
                colormap=cmap, width=args.px, height=args.px,
                elev_deg=20.0, azim_deg=35.0, n_steps=340,
                opacity_scale=4.0, vmin=0.0, vmax=1.0,
                clip_normal=np.array([0.0, 1.0, 0.0]),
                background=(1.0, 1.0, 1.0),
            )
            path = outdir / f"{tag}_{field_name}.png"
            iio.imwrite(path, (img * 255).astype(np.uint8))
            manifest[f"{tag}_{field_name}"] = str(path.relative_to(ROOT))
            print(f"  rendered {path.name}")

        # radial profile for the accompanying line plot
        manifest.setdefault("profiles", {})[tag] = {
            "r": res.r.tolist(),
            "n": res.n[it].tolist(),
            "oxygen": res.oxygen[it].tolist(),
            "burden": res.burden().tolist(),
            "t": res.t.tolist(),
        }

    (outdir / "manifest.json").write_text(json.dumps(manifest))
    print(f"rendered {len([k for k in manifest if k != 'profiles'])} volumes -> {outdir}")


if __name__ == "__main__":
    main()
