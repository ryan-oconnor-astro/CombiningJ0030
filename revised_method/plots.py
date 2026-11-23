"""
Figures of the revised combination, written to revised_method/figures/.

    python3 revised_method/plots.py              # all figures
    python3 revised_method/plots.py 3x3 tempering  # a subset

  compactness  KDECompactness.png               counterpart of manuscript Fig. 1
  3x3          3x3Posterior.png                 counterpart of manuscript Fig. 2
  marginals    MarginalPosteriors.png           counterpart of manuscript Fig. 3
  other_ns     PosteriorvsOtherNeutronStars.png counterpart of manuscript Fig. 4 (left)
  tempering    CorrelationTempering.png         rho=0 vs rho=1 vs the default, all (M, C); no marginal panels

All "combined" curves are the default posterior (computed in (M, C),
rho ~ Uniform(0,1), Sigma_sys marginalized). The 2D ones are mapped to (M, R)
with the Jacobian K M / R^2; the compactness one is the C marginal of the same
2D posterior, so all headline numbers come from one computation. Reads
results/combined_MC.npz.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import core
from core import MODELS, trapz

DEFAULT = core.DEFAULT_SCENARIO


def pct():
    return r"\%" if plt.rcParams["text.usetex"] else "%"


def rgba(color, alpha):
    return mpl.colors.to_rgba(color, alpha)


def save(fig, name, dpi=300, **kw):
    os.makedirs(core.FIG_DIR, exist_ok=True)
    path = os.path.join(core.FIG_DIR, name)
    fig.savefig(path, dpi=dpi, **kw)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, core.REPO_ROOT)}")


def kde_on_grid(samples_xy, x_grid, y_grid):
    """Silverman KDE of 2 x N samples, evaluated on (x_grid, y_grid) with
    indexing (len(x_grid), len(y_grid)) and normalized."""
    XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
    d = gaussian_kde(samples_xy, bw_method="silverman")(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    return d / trapz(trapz(d, y_grid, axis=1), x_grid)


def hdr_extent(post, M_grid, R_grid, prob=0.95, pad=0.15):
    """(R_lo, R_hi), (M_lo, M_hi) of the prob HDR, padded by a fraction of its size."""
    inside = post >= core.hdr_levels(post, (prob,))[0]
    Ms, Rs = M_grid[inside.any(axis=1)], R_grid[inside.any(axis=0)]
    dR, dM = np.ptp(Rs) * pad, np.ptp(Ms) * pad
    return ((max(Rs.min() - dR, R_grid[0]), min(Rs.max() + dR, R_grid[-1])),
            (max(Ms.min() - dM, M_grid[0]), min(Ms.max() + dM, M_grid[-1])))


def draw_hdr(ax, x, y, density, color, fill_alphas, line_alpha=0.7, widths=(1.3, 1.8)):
    lvl95, lvl68 = core.hdr_levels(density)
    ax.contourf(x, y, density, levels=[lvl95, lvl68, density.max()],
                colors=[rgba(color, fill_alphas[0]), rgba(color, fill_alphas[1])], antialiased=True)
    ax.contour(x, y, density, levels=[lvl95, lvl68], colors=[rgba(color, line_alpha)],
               linewidths=list(widths), linestyles=["--", "-"])


# ----------------------------------------------------------------------------
# Fig. 1: compactness
# ----------------------------------------------------------------------------


def fig_compactness(res):
    # Individual 1D KDEs exactly as in 1D_Compactness_Combination.ipynb.
    C_samples = {}
    for key in MODELS:
        M, R = core.load_samples(key)
        C_samples[key] = core.K_KM * M / R
    C_grid, P_C = res["C_grid"], core.marginal_C(res, DEFAULT)
    C_fine = np.linspace(min(C_grid[0], min(c.min() for c in C_samples.values())),
                         max(C_grid[-1], max(c.max() for c in C_samples.values())), 400)
    q = core.quantiles(C_grid, P_C)
    q16, q50, q84 = q["median"] - q["lo"], q["median"], q["median"] + q["hi"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, meta in MODELS.items():
        kde = gaussian_kde(C_samples[key], bw_method="silverman").evaluate(C_fine)
        ax.plot(C_fine, kde / trapz(kde, C_fine), color=meta["color"], lw=2, alpha=0.5, label=meta["label"])
    for prob, val in [(16, q16), (50, q50), (84, q84)]:
        ax.axvline(val, linestyle="-." if prob == 50 else "--", color="black",
                   linewidth=1.5 if prob == 50 else 0.9, label=rf"{prob}{pct()}: {val:.3f}")
    ax.plot(C_grid, P_C, color="black", lw=2, label="Combined Posterior")
    ax.set_xlim(C_fine[0], C_fine[-1])
    ax.set_xlabel("Compactness $GM/(Rc^2)$", fontsize=27)
    ax.set_ylabel("Density", fontsize=27)
    ax.set_title("Combined Compactness Posterior with Individual KDEs", fontsize=24)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=13)
    save(fig, "KDECompactness.png", bbox_inches="tight")


# ----------------------------------------------------------------------------
# Fig. 2: 3x3 panels
# ----------------------------------------------------------------------------


def fig_3x3(res):
    M_grid, R_grid = res["M_grid"], res["R_grid"]
    post, _, _ = core.marginals_MR(res, DEFAULT)
    R_lim, M_lim = hdr_extent(post, M_grid, R_grid)
    # Inputs: plain (M, R) Silverman KDEs, as in 2D_MassRadius_Combination.ipynb,
    # on a display grid (some inputs have up to 10^6 samples).
    M_disp, R_disp = np.linspace(*M_lim, 90), np.linspace(*R_lim, 120)
    g = {key: kde_on_grid(np.vstack(core.load_samples(key)), M_disp, R_disp) for key in MODELS}

    fig, axes = plt.subplots(3, 3, figsize=(24, 18), constrained_layout=True)
    axes = axes.flatten()

    def style(ax, title):
        ax.set_title(title, fontsize=26)
        ax.set_xlabel("Radius (km)", fontsize=22)
        ax.set_ylabel("Mass ($M_\\odot$)", fontsize=22)
        ax.set_xlim(*R_lim)
        ax.set_ylim(*M_lim)
        ax.minorticks_on()
        ax.tick_params(labelsize=14)

    for ax, (key, meta) in zip(axes, MODELS.items()):
        draw_hdr(ax, R_grid, M_grid, post, "black", (0.10, 0.25), widths=(1.8, 2.5))
        draw_hdr(ax, R_disp, M_disp, g[key], meta["color"], (0.20, 0.45))
        style(ax, meta["label"])
        ax.legend(handles=[mpl.lines.Line2D([], [], color=meta["color"], lw=2, label=meta["label"]),
                           mpl.lines.Line2D([], [], color="black", lw=2, label="Posterior")],
                  fontsize=18, loc="upper left", frameon=True, fancybox=True, framealpha=0.8)

    ax_all = axes[len(MODELS)]
    for key, meta in MODELS.items():
        draw_hdr(ax_all, R_disp, M_disp, g[key], meta["color"], (0.10, 0.20))
    draw_hdr(ax_all, R_grid, M_grid, post, "black", (0.10, 0.25), line_alpha=1.0, widths=(1.8, 2.5))
    style(ax_all, "All Models")

    fig.suptitle("Model M--R Posteriors Compared to Posterior Combination", fontsize=36)
    save(fig, "3x3Posterior.png", dpi=100)


# ----------------------------------------------------------------------------
# Fig. 3: joint posterior with marginals
# ----------------------------------------------------------------------------


def fig_marginals(res):
    import seaborn as sns

    M_grid, R_grid = res["M_grid"], res["R_grid"]
    post, P_M, P_R = core.marginals_MR(res, DEFAULT)
    qM, qR = core.quantiles(M_grid, P_M), core.quantiles(R_grid, P_R)
    R_lim, M_lim = hdr_extent(post, M_grid, R_grid, pad=0.05)
    lvl95, lvl68 = core.hdr_levels(post)

    g = sns.JointGrid(height=8, space=0)
    g.ax_joint.set_xlim(*R_lim)
    g.ax_joint.set_ylim(*M_lim)
    g.ax_joint.contourf(R_grid, M_grid, post, levels=[lvl95, lvl68, post.max()],
                        colors=[rgba("black", 0.12), rgba("black", 0.28)], antialiased=True)
    g.ax_joint.contour(R_grid, M_grid, post, levels=[lvl95, lvl68], colors="black",
                       linewidths=[1.0, 1.6], linestyles=["--", "-"])
    g.ax_marg_x.plot(R_grid, P_R, color="black", lw=2)
    g.ax_marg_x.fill_between(R_grid, P_R, color="black", alpha=0.25)
    g.ax_marg_y.plot(P_M, M_grid, color="black", lw=2)
    g.ax_marg_y.fill_betweenx(M_grid, P_M, color="black", alpha=0.25)

    for q, vline, ax_joint_line, ax_marg in [(qR, True, g.ax_joint.axvline, g.ax_marg_x.axvline),
                                            (qM, False, g.ax_joint.axhline, g.ax_marg_y.axhline)]:
        for v in (q["median"] - q["lo"], q["median"] + q["hi"]):
            ax_joint_line(v, color="black", linestyle=":", linewidth=1.8, alpha=0.9, zorder=10)
            ax_marg(v, color="black", linestyle=":", linewidth=1.8)
        ax_joint_line(q["median"], color="black", linestyle="-.", linewidth=2.2, zorder=10)
        ax_marg(q["median"], color="black", linestyle="-.", linewidth=2.2)

    g.ax_joint.set_xlim(*R_lim)
    g.ax_joint.set_ylim(*M_lim)
    g.ax_joint.set_xlabel("Radius (km)", fontsize=20)
    g.ax_joint.set_ylabel("Mass ($M_\\odot$)", fontsize=20)
    g.ax_joint.tick_params(labelsize=16, which="both", top=True, right=True)
    g.ax_marg_x.tick_params(labelsize=14)
    g.ax_marg_y.tick_params(labelsize=14)
    g.fig.suptitle("Combined Posterior with Marginal Distributions", fontsize=27, y=1.05)
    save(g.fig, "MarginalPosteriors.png", bbox_inches="tight")


# ----------------------------------------------------------------------------
# Fig. 4 (left): other neutron stars
# ----------------------------------------------------------------------------


def fig_other_ns(res):
    post, _, _ = core.marginals_MR(res, DEFAULT)
    M_grid, R_grid = res["M_grid"], res["R_grid"]
    R_lim, _ = hdr_extent(post, M_grid, R_grid)
    R_lim = (min(R_lim[0], 9.0), max(R_lim[1], 17.0))
    M_lim = (M_grid[0], M_grid[-1])   # the combination's M domain
    # Other stars: KDEs of their samples on a display grid.
    R_disp = np.linspace(*R_lim, 120)
    M_disp = np.linspace(0.9, 2.3, 90)

    series = []
    for fname, color, lab in [("GW170817_M1", "tab:blue", "GW170817: M1"),
                              ("GW170817_M2", "tab:green", "GW170817: M2"),
                              ("J0437", "tab:red", "J0437")]:
        M, R = np.loadtxt(os.path.join(core.DATA_DIR, fname)).T[:2]
        series.append((M_disp, R_disp, kde_on_grid(np.vstack([M, R]), M_disp, R_disp), color, lab))
    series.append((M_grid, R_grid, post, "black", "J0030: Combined Posterior"))

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    handles = []
    for Mg, Rg, density, color, lab in series:
        lvl95, lvl68 = core.hdr_levels(density)
        ax.contour(Rg, Mg, density, levels=[lvl95, lvl68], colors=[color, color],
                   linewidths=[2.0, 2.6], linestyles=["--", "-"])
        ax.contourf(Rg, Mg, density, levels=[lvl95, lvl68, density.max()],
                    colors=[rgba(color, 0.25), rgba(color, 0.45)], antialiased=True)
        handles.append(mpl.lines.Line2D([], [], color=color, lw=2, label=lab))
    ax.legend(handles=handles, fontsize=16, loc="upper left")
    ax.set_xlabel("Radius (km)", fontsize=27)
    ax.set_ylabel("Mass ($M_\\odot$)", fontsize=27)
    ax.set_title("J0030 MR Posterior vs. Other Neutron Stars", fontsize=27)
    ax.set_xlim(*R_lim)
    ax.set_ylim(*M_lim)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True, labelsize=16)
    save(fig, "PosteriorvsOtherNeutronStars.png")


# ----------------------------------------------------------------------------
# Effect of tempering and of (M, C)
# ----------------------------------------------------------------------------

TEMPERING_COLORS = dict(rho0="#1f1f1e", rho1="#eb6834", default="#1baf7a")
MUTED = "#6b6a63"


def fig_tempering(res):
    # All three are the (M, C) combination, mapped to (M, R).
    curves = [
        (res, "rho0", r"Independent ($\rho=0$, no $\Sigma_{\rm sys}$)", TEMPERING_COLORS["rho0"]),
        (res, "rho1", r"Fully dependent ($\rho=1$, no $\Sigma_{\rm sys}$)", TEMPERING_COLORS["rho1"]),
        (res, "uniform_sys", r"$\rho \sim U(0,1)$ + $\Sigma_{\rm sys}$ (default)", TEMPERING_COLORS["default"]),
    ]

    fig, ax = plt.subplots(figsize=(13, 10))

    handles = []
    for r, scen, lab, color in curves:
        post, P_M, P_R = core.marginals_MR(r, scen)
        M_grid, R_grid = r["M_grid"], r["R_grid"]
        lvl95, lvl68 = core.hdr_levels(post)
        ax.contourf(R_grid, M_grid, post, levels=[lvl95, lvl68, post.max() * 1.01],
                    colors=[rgba(color, 0.08), rgba(color, 0.18)], antialiased=True)
        ax.contour(R_grid, M_grid, post, levels=[lvl95, lvl68], colors=[color],
                   linewidths=[1.8, 2.5], linestyles=["--", "-"])
        qM, qR = core.quantiles(M_grid, P_M), core.quantiles(R_grid, P_R)
        handles.append(mpl.lines.Line2D([], [], color=color, lw=2, label=(
            f"{lab}\n"
            f"  $M = {qM['median']:.2f}^{{+{qM['hi']:.2f}}}_{{-{qM['lo']:.2f}}}\\,M_\\odot$,  "
            f"$R = {qR['median']:.2f}^{{+{qR['hi']:.2f}}}_{{-{qR['lo']:.2f}}}$ km")))

    # Frame the widest (rho = 1) 95% HDR.
    R_lim, M_lim = hdr_extent(core.marginals_MR(res, "rho1")[0], res["M_grid"], res["R_grid"], pad=0.05)
    ax.set_xlim(*R_lim)
    ax.set_ylim(*M_lim)
    ax.set_xlabel("Radius (km)", fontsize=22)
    ax.set_ylabel("Mass ($M_\\odot$)", fontsize=22)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=16)
    handles.append(mpl.lines.Line2D([], [], color="black", lw=2.5, ls="-", label=f"68{pct()} HDR"))
    handles.append(mpl.lines.Line2D([], [], color="black", lw=1.8, ls="--", label=f"95{pct()} HDR"))
    ax.legend(handles=handles, loc="lower right", fontsize=14, frameon=True,
              fancybox=True, framealpha=0.9, labelspacing=0.8)
    fig.suptitle(r"Effect of Correlation Tempering", y=0.975, fontsize=27)
    fig.text(0.08, 0.012,
             r"All curves are combined in $(M, C)$ and mapped to $(M, R)$. $\rho=0$ treats the 8 inputs as "
             r"independent;" "\n"
             r"$\rho=1$ as one effective measurement ($\kappa = 1/8$). The default marginalizes $\rho$ over "
             r"Uniform(0,1)" "\n"
             r"and adds the group-level systematic $\Sigma_{\rm sys}$.",
             fontsize=13, color=MUTED, ha="left", va="bottom")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.17)
    save(fig, "CorrelationTempering.png")


FIGURES = dict(compactness=fig_compactness, **{"3x3": fig_3x3}, marginals=fig_marginals,
               other_ns=fig_other_ns, tempering=fig_tempering)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", metavar="figure", help=f"any of {list(FIGURES)} (default: all)")
    args = parser.parse_args()
    unknown = [n for n in args.names if n not in FIGURES]
    if unknown:
        parser.error(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    names = args.names or list(FIGURES)
    core.use_paper_style()
    res = core.load_results("MC")
    for n in names:
        FIGURES[n](res)


if __name__ == "__main__":
    main()
