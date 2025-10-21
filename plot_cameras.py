import numpy as np
import torch
import matplotlib.pyplot as plt

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def plot_cameras_topdown(c2ws, ref_idx=0, arrow_scale=0.3, z_forward=True,
                         draw_radius=True, title="Top-down camera view (X–Z)"):
    """
    Top-down plot (looking down +Y) of camera centers & forward directions.

    Args:
      c2ws: list of (4,4) c2w matrices (torch or np)
      ref_idx: index of reference camera to highlight
      arrow_scale: length multiplier for the direction arrows (world units)
      z_forward: True if +Z is the camera-forward axis. Set False if your convention is -Z forward.
      draw_radius: draw a circle at |C_ref| in XZ plane (useful for orbit sanity)
    """
    Cs = []
    Fs = []

    for i, c2w in enumerate(c2ws):
        M = to_numpy(c2w)
        R = M[:3, :3]
        t = M[:3, 3]
        # camera center in world coords is t (since this is c2w)
        C = t
        # forward axis is the 3rd column of R if +Z forward; flip if needed
        fwd = R[:, 2] if z_forward else -R[:, 2]

        # keep only X,Z components for top-down
        Cs.append([C[0], C[2]])
        Fs.append([fwd[0], fwd[2]])

    Cs = np.array(Cs)   # shape (N,2) with columns [X,Z]
    Fs = np.array(Fs)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)

    # scatter all cameras
    ax.scatter(Cs[:, 0], Cs[:, 1], c="tab:gray", s=30, label="cameras")

    # highlight reference
    ax.scatter(Cs[ref_idx, 0], Cs[ref_idx, 1], c="tab:red", s=60, label=f"ref {ref_idx}", zorder=3)

    # draw forward arrows
    for i in range(len(c2ws)):
        ax.arrow(Cs[i, 0], Cs[i, 1],
                 Fs[i, 0] * arrow_scale, Fs[i, 1] * arrow_scale,
                 head_width=arrow_scale * 0.15, head_length=arrow_scale * 0.2,
                 fc=("tab:red" if i == ref_idx else "tab:blue"),
                 ec=("tab:red" if i == ref_idx else "tab:blue"),
                 length_includes_head=True, alpha=0.9)

        # label each camera with its index
        ax.text(Cs[i, 0], Cs[i, 1], f"{i}", fontsize=8, ha="left", va="bottom")

    # draw origin
    ax.scatter(0.0, 0.0, c="k", s=20, label="origin")

    # optional radius circle (distance of ref cam in XZ)
    if draw_radius:
        r = np.linalg.norm(Cs[ref_idx])  # distance in XZ
        circ = plt.Circle((0, 0), r, color="tab:green", fill=False, linestyle="--", alpha=0.6, label=f"radius≈{r:.3g}")
        ax.add_artist(circ)

    ax.set_xlabel("X (world)")
    ax.set_ylabel("Z (world)")
    ax.set_aspect("equal", adjustable="box")
    # expand limits with a small margin
    pad = max(1e-3, 0.1 * max(1.0, np.linalg.norm(Cs, axis=1).max()))
    xmin, xmax = Cs[:, 0].min() - pad, Cs[:, 0].max() + pad
    zmin, zmax = Cs[:, 1].min() - pad, Cs[:, 1].max() + pad
    ax.set_xlim(min(xmin, -pad), max(xmax, pad))
    ax.set_ylim(min(zmin, -pad), max(zmax, pad))
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="best")
    plt.show()
    return fig, ax

def _to_numpy(M):
    if isinstance(M, torch.Tensor):
        return M.detach().cpu().numpy()
    return np.asarray(M)

def _plane_axes(up_axis: str):
    """
    Return the (horizontal_x, horizontal_y) axis indices for the plot plane,
    given which axis is 'up'.

    up='y' -> plot XZ plane (x right, z up on canvas)
    up='z' -> plot XY plane
    up='x' -> plot YZ plane
    """
    up_axis = up_axis.lower()
    if up_axis == 'y':
        return (0, 2)   # X,Z
    elif up_axis == 'z':
        return (0, 1)   # X,Y
    elif up_axis == 'x':
        return (1, 2)   # Y,Z
    else:
        raise ValueError("up_axis must be one of {'x','y','z'}")

def plot_cameras_topdown_flexible(
    c2ws,
    ref_idx: int = 0,
    up_axis: str = 'y',
    forward_axis: str = 'z',
    forward_sign: int = +1,   # +1 for +axis forward, -1 for -axis forward
    arrow_scale: float = 0.3,
    draw_radius: bool = True,
    title: str = None,
):
    """
    Top-down plot of camera centers and forward directions with configurable up-axis.

    Args:
      c2ws: list of (4,4) camera-to-world matrices (torch.Tensor or np.ndarray)
      ref_idx: which camera to highlight
      up_axis: which world axis is 'up' in your convention: 'x' | 'y' | 'z'
      forward_axis: which camera axis is 'forward' in your convention: 'x' | 'y' | 'z'
      forward_sign: +1 if forward is +axis (e.g., +z), -1 if forward is -axis
      arrow_scale: length multiplier for direction arrows (world units)
      draw_radius: draw a circle at the ref camera distance in the plot plane
      title: optional plot title
    """
    # map axis letter -> index
    ax2idx = {'x': 0, 'y': 1, 'z': 2}
    f_idx = ax2idx[forward_axis.lower()]
    hx, hy = _plane_axes(up_axis)

    # Collect centers and forward directions
    centers = []
    forwards = []

    for M in c2ws:
        M = _to_numpy(M)
        R, t = M[:3, :3], M[:3, 3]
        centers.append([t[hx], t[hy]])

        fwd_vec_world = R[:, f_idx] * float(forward_sign)  # world-space forward
        forwards.append([fwd_vec_world[hx], fwd_vec_world[hy]])

    centers = np.asarray(centers)   # (N,2)
    forwards = np.asarray(forwards) # (N,2)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    if title is None:
        title = f"Top-down camera view (plane: {['X','Y','Z'][hx]}–{['X','Y','Z'][hy]}; up={up_axis.upper()})"
    ax.set_title(title)

    # all cameras
    ax.scatter(centers[:, 0], centers[:, 1], c="tab:gray", s=30, label="cameras")

    # highlight ref
    ax.scatter(centers[ref_idx, 0], centers[ref_idx, 1], c="tab:red", s=60, label=f"ref {ref_idx}", zorder=3)

    # arrows & labels
    for i in range(len(c2ws)):
        ax.arrow(
            centers[i, 0], centers[i, 1],
            forwards[i, 0] * arrow_scale, forwards[i, 1] * arrow_scale,
            head_width=arrow_scale * 0.15, head_length=arrow_scale * 0.2,
            fc=("tab:red" if i == ref_idx else "tab:blue"),
            ec=("tab:red" if i == ref_idx else "tab:blue"),
            length_includes_head=True, alpha=0.95,
        )
        ax.text(centers[i, 0], centers[i, 1], f"{i}", fontsize=8, ha="left", va="bottom")

    # origin
    ax.scatter(0.0, 0.0, c="k", s=20, label="origin")

    # optional: radius circle (distance of ref center in the plot plane)
    if draw_radius:
        r = float(np.linalg.norm(centers[ref_idx]))
        circ = plt.Circle((0, 0), r, color="tab:green", fill=False, linestyle="--", alpha=0.6, label=f"radius≈{r:.3g}")
        ax.add_artist(circ)

    # cosmetics
    ax.set_xlabel(f"{['X','Y','Z'][hx]} (world)")
    ax.set_ylabel(f"{['X','Y','Z'][hy]} (world)")
    ax.set_aspect("equal", adjustable="box")

    # nice bounds
    if len(centers) > 0:
        pad = max(1e-3, 0.1 * max(1.0, np.linalg.norm(centers, axis=1).max()))
        xmin, xmax = centers[:, 0].min() - pad, centers[:, 0].max() + pad
        ymin, ymax = centers[:, 1].min() - pad, centers[:, 1].max() + pad
        ax.set_xlim(min(xmin, -pad), max(xmax, pad))
        ax.set_ylim(min(ymin, -pad), max(ymax, pad))
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="best")
    plt.show()
    return fig, ax
