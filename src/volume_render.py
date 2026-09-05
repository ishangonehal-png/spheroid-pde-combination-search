"""CPU volume ray-caster for spherically symmetric spheroid fields.

Why this exists
---------------
The spheroid solver is one-dimensional in radius, but the object it models is a
genuine 3-D sphere: the radial profile f(r) *is* the 3-D field f(|x|).  Showing
that field as a volume is therefore an exact reconstruction, not an artist's
impression -- and it makes visible the one thing a well-mixed ODE structurally
cannot represent: a treated spheroid that is dead at the rim and alive at the
centre.

There is no OpenGL context in this environment (no X server; VTK ships no OSMesa
build for macOS), so instead of a GPU rasteriser we integrate the standard
emission-absorption volume rendering integral

    C = \\int_0^L c(s) \\, \\sigma(s) \\, exp(-\\int_0^s \\sigma(u) du) ds

directly with numpy alongeach camera ray.  For a spherically symmetric field the
sample lookup is a 1-D interpolation on |x(s)|, so the whole image is a few
vectorised array operations and runs in seconds on CPU.  This is a physically
standard renderer, not an approximation of one.

A clip plane produces the cutaway views: rays are only accumulated once they
pass the plane, so the interior structure is directly readable.
"""
from __future__ import annotations

import numpy as np

__all__ = ["render_volume", "shade_surface"]


def _camera_rays(width: int, height: int, elev_deg: float, azim_deg: float,
                 distance: float, fov_deg: float = 32.0):
    """Build ray origins/directions for a pinhole camera looking at the origin."""
    el = np.radians(elev_deg)
    az = np.radians(azim_deg)
    eye = distance * np.array([np.cos(el) * np.cos(az),
                               np.cos(el) * np.sin(az),
                               np.sin(el)])
    fwd = -eye / np.linalg.norm(eye)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    half = np.tan(np.radians(fov_deg) / 2.0)
    ys, xs = np.mgrid[0:height, 0:width]
    # pixel centres in [-1, 1], y flipped so +z is up in the image
    px = (2.0 * (xs + 0.5) / width - 1.0) * half
    py = (1.0 - 2.0 * (ys + 0.5) / height) * half * (height / width)

    dirs = (fwd[None, None, :]
            + px[..., None] * right[None, None, :]
            + py[..., None] * up[None, None, :])
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return eye, dirs


def render_volume(r: np.ndarray, f: np.ndarray, *,
                  colormap,
                  width: int = 900, height: int = 900,
                  elev_deg: float = 20.0, azim_deg: float = 35.0,
                  n_steps: int = 320,
                  opacity_scale: float = 3.2,
                  opacity_gamma: float = 1.0,
                  clip_normal: np.ndarray | None = np.array([0.0, 1.0, 0.0]),
                  vmin: float | None = None, vmax: float | None = None,
                  background=(1.0, 1.0, 1.0),
                  ambient: float = 0.30,
                  shell_alpha: float = 0.14,
                  shell_colour=(0.55, 0.55, 0.60),
                  light_dir: np.ndarray = np.array([0.4, -0.7, 0.6])):
    """Ray-cast a spherically symmetric field ``f(r)`` into an RGBA image.

    Parameters
    ----------
    r, f : radial grid and the field sampled on it.
    colormap : a matplotlib colormap mapping normalised value -> RGB.
    opacity_scale : overall extinction; higher = more opaque.
    clip_normal : if given, the half-space ``x . n > 0`` is cut away, producing a
        cutaway view of the interior.  Pass ``None`` for the intact sphere.
    ambient, light_dir : the accumulated field gradient is used as a surface
        normal for simple diffuse shading, which is what gives the render its
        sense of depth.

    Returns
    -------
    (H, W, 4) float array in [0, 1].
    """
    R = float(r[-1])
    vmin = float(np.nanmin(f)) if vmin is None else vmin
    vmax = float(np.nanmax(f)) if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1e-9

    eye, dirs = _camera_rays(width, height, elev_deg, azim_deg, distance=4.3 * R)

    # Ray-sphere intersection against the bounding sphere of radius R.
    o = eye[None, None, :]
    b = np.einsum("...i,...i->...", dirs, np.broadcast_to(o, dirs.shape))
    c = float(eye @ eye) - R * R
    disc = b * b - c
    hit = disc > 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = -b - sq
    t1 = -b + sq
    t0 = np.maximum(t0, 0.0)

    rgb = np.zeros((height, width, 3), dtype=float)
    alpha = np.zeros((height, width), dtype=float)
    normal_acc = np.zeros((height, width, 3), dtype=float)
    # A ray whose first contributing sample lies on the clip plane is looking at
    # the flat cut face, which must be shaded with the PLANE normal -- shading it
    # with the radial direction makes the cutaway read as a curved surface and
    # destroys the "you are seeing inside" cue that motivates the figure.
    first_on_plane = np.zeros((height, width), dtype=bool)
    seen_any = np.zeros((height, width), dtype=bool)

    ts = np.linspace(0.0, 1.0, n_steps)
    seg = np.where(hit, (t1 - t0) / max(n_steps - 1, 1), 0.0)
    ldir = light_dir / np.linalg.norm(light_dir)

    lo, hi = float(r[0]), float(r[-1])
    for s in ts:
        t = t0 + s * (t1 - t0)
        pos = np.broadcast_to(o, dirs.shape) + dirs * t[..., None]
        rad = np.linalg.norm(pos, axis=-1)
        inside = hit & (rad <= R)
        on_plane = np.zeros_like(inside)
        if clip_normal is not None:
            n_hat = clip_normal / np.linalg.norm(clip_normal)
            sd = pos @ n_hat
            inside &= sd <= 0.0
            # samples within one step of the cut plane constitute the flat face
            on_plane = inside & (sd > -(1.5 * R / n_steps))
        if not inside.any():
            continue
        newly = inside & ~seen_any
        first_on_plane |= newly & on_plane
        seen_any |= inside

        val = np.interp(np.clip(rad, lo, hi), r, f)
        norm = np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0)

        # extinction proportional to the (normalised) field
        sigma = opacity_scale * np.power(norm, opacity_gamma)
        a_step = 1.0 - np.exp(-sigma * seg / R * n_steps / 64.0)
        a_step = np.where(inside, a_step, 0.0)

        col = colormap(norm)[..., :3]

        # accumulate an outward normal from the radial direction (the field's
        # gradient is radial by symmetry), weighted by what each step contributes
        with np.errstate(invalid="ignore", divide="ignore"):
            rad_hat = pos / np.maximum(rad, 1e-9)[..., None]
        w = (a_step * (1.0 - alpha))[..., None]
        normal_acc += w * rad_hat

        trans = (1.0 - alpha)[..., None]
        rgb += trans * (a_step[..., None] * col)
        alpha += (1.0 - alpha) * a_step

    # diffuse shading from the accumulated normal
    nn = np.linalg.norm(normal_acc, axis=-1, keepdims=True)
    normals = np.where(nn > 1e-9, normal_acc / np.maximum(nn, 1e-9), 0.0)
    if clip_normal is not None:
        n_hat = clip_normal / np.linalg.norm(clip_normal)
        normals = np.where(first_on_plane[..., None], n_hat[None, None, :], normals)
    lam = np.clip(normals @ ldir, 0.0, 1.0)
    shade = ambient + (1.0 - ambient) * lam
    # keep the flat face evenly lit so the interior field reads as data, not relief
    shade = np.where(first_on_plane, 0.90, shade)
    rgb = rgb * shade[..., None]

    # A spheroid that has been cleared to ~0 density is almost transparent, which
    # renders as an EMPTY FRAME -- visually indistinguishable from a missing
    # panel, even though "nearly all cells are gone" is precisely the result.
    # Draw a faint shell at the spheroid boundary so an empty sphere still reads
    # as a sphere.
    if shell_alpha > 0.0:
        shell = hit & (alpha < 0.98)
        if clip_normal is not None:
            n_hat = clip_normal / np.linalg.norm(clip_normal)
            # front-most visible point along each ray
            p_entry = np.broadcast_to(o, dirs.shape) + dirs * t0[..., None]
            shell &= (p_entry @ n_hat) <= 0.0
        a_sh = np.where(shell, shell_alpha * (1.0 - alpha), 0.0)
        sh_col = np.asarray(shell_colour, dtype=float)
        rgb += a_sh[..., None] * sh_col
        alpha += a_sh

    out = np.zeros((height, width, 4), dtype=float)
    bg = np.asarray(background, dtype=float)
    out[..., :3] = rgb + (1.0 - alpha)[..., None] * bg
    out[..., 3] = alpha
    return np.clip(out, 0.0, 1.0)


def shade_surface(r: np.ndarray, f: np.ndarray, **kw):
    """Convenience wrapper: opaque isosurface-like view (high extinction)."""
    kw.setdefault("opacity_scale", 14.0)
    return render_volume(r, f, **kw)
