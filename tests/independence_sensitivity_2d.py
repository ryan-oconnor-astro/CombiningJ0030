"""
2D (M, R) counterpart to independence_sensitivity.py -- see that file's
docstring for the full method description, scenario definitions, and the
explicit list of assumptions that need author input (the correlation numbers
used in the "referee_informed" scenario in particular).

This applies the same power-likelihood / design-effect tempering to the exact
2D M-R good/bad combination pipeline from 2D_MassRadius_Combination.ipynb, so
we can report the sensitivity of the manuscript's actual headline numbers
(M = 1.46 +0.09/-0.08, R = 12.69 +0.64/-0.55) rather than only the 1D
compactness result.

The expensive step (marginalizing the bad component over alpha, sigma_M,
sigma_R for each of the 8 models) does not depend on the weights, so it is
computed once and reused across every rho in the sweep -- only the cheap
"combine across models" step is repeated per scenario.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm, gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, "datafiles")
FIG_DIR = os.path.join(HERE, "independence_sensitivity_figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = {
    "ST_PST19": dict(file="ST_PST19"),
    "ST_PDT":   dict(file="ST_PDT"),
    "ST_PST":   dict(file="ST_PST"),
    "PDT_U":    dict(file="PDT_U"),
    "ST_U":     dict(file="ST_U"),
    "2spot":    dict(file="2spot"),
    "3spot":    dict(file="3spot"),
    "PDT_U26":  dict(file="PDT_U26"),
}
MODEL_KEYS = list(MODELS)

GROUPS = {
    "Riley":       ["ST_PST19"],
    "Miller":      ["2spot", "3spot"],
    "Vinciguerra": ["ST_PDT", "ST_PST", "PDT_U", "ST_U"],
    "Kini":        ["PDT_U26"],
}

MANUSCRIPT_M = dict(median=1.46, lo=0.08, hi=0.09)
MANUSCRIPT_R = dict(median=12.69, lo=0.55, hi=0.64)

# ----------------------------------------------------------------------------
# Load data, build good (KDE) densities on the (M, R) grid
# ----------------------------------------------------------------------------

samples_MR, M_med, S_Mi, R_med, S_Ri = {}, {}, {}, {}, {}
KDEs_2D = {}
for key, meta in MODELS.items():
    data = np.loadtxt(os.path.join(DATA_DIR, meta["file"]))
    M, R = data.T[0], data.T[1]
    samples_MR[key] = np.vstack([M, R])
    M_med[key] = np.median(M)
    S_Mi[key] = max(M.max() - M_med[key], M_med[key] - M.min())
    R_med[key] = np.median(R)
    S_Ri[key] = max(R.max() - R_med[key], R_med[key] - R.min())
    KDEs_2D[key] = gaussian_kde(samples_MR[key], bw_method="silverman")

M_grid = np.linspace(0.95, 2.2, 80)
R_grid = np.linspace(9.0, 17.0, 80)
MM, RR = np.meshgrid(M_grid, R_grid, indexing="ij")
pos = np.vstack([MM.ravel(), RR.ravel()])

g_dict = {}
for key in MODEL_KEYS:
    g = KDEs_2D[key](pos).reshape(MM.shape)
    g /= trapezoid(trapezoid(g, R_grid, axis=1), M_grid)
    g_dict[key] = g
g_stack = np.array([g_dict[key] for key in MODEL_KEYS])  # (8, NM, NR)

# Hyperparameter grids -- reduced resolution relative to
# 2D_MassRadius_Combination.ipynb to keep the rho-sweep tractable; the
# rho=0 ("manuscript") case below is checked against the manuscript to
# confirm this reduction doesn't change the answer materially.
avg_S_Mi = np.mean(list(S_Mi.values()))
avg_S_Ri = np.mean(list(S_Ri.values()))
sigma_M_grid = np.logspace(-10 * avg_S_Mi, -0.1 * avg_S_Mi, 6)
sigma_R_grid = np.logspace(-10 * avg_S_Ri, -0.1 * avg_S_Ri, 6)
alpha_grid = np.linspace(1.0, 10, 40)
p_grid = np.linspace(0.0, 1.0, 25)

prior_sigma_M = (1.0 / sigma_M_grid)
prior_sigma_M /= trapezoid(prior_sigma_M, sigma_M_grid)
prior_sigma_R = (1.0 / sigma_R_grid)
prior_sigma_R /= trapezoid(prior_sigma_R, sigma_R_grid)
prior_alpha = np.exp(-alpha_grid)
prior_alpha /= trapezoid(prior_alpha, alpha_grid)

print("Building bad-component grids for all 8 models (weight-independent, cached)...")
b_dict = {}
for key in MODEL_KEYS:
    mu_M, mu_R = M_med[key], R_med[key]
    b_sigma = []
    for sigma_M in sigma_M_grid:
        row_sigma_R = []
        for sigma_R in sigma_R_grid:
            var_M = (alpha_grid * S_Mi[key]) ** 2 + sigma_M ** 2
            var_R = (alpha_grid * S_Ri[key]) ** 2 + sigma_R ** 2
            b_alpha = (
                norm.pdf(MM[None, :, :], loc=mu_M, scale=np.sqrt(var_M)[:, None, None])
                * norm.pdf(RR[None, :, :], loc=mu_R, scale=np.sqrt(var_R)[:, None, None])
            )
            norm_const = trapezoid(trapezoid(b_alpha, R_grid, axis=2), M_grid, axis=1)
            b_alpha /= norm_const[:, None, None]

            b_marg = trapezoid(b_alpha * prior_alpha[:, None, None], alpha_grid, axis=0)
            b_marg /= trapezoid(trapezoid(b_marg, R_grid, axis=1), M_grid)
            row_sigma_R.append(b_marg)
        b_sigma.append(row_sigma_R)
    b_dict[key] = np.array(b_sigma)  # (NsM, NsR, NM, NR)
print("Done.")


def combine_2d(weights):
    w = np.array([weights[key] for key in MODEL_KEYS])[:, None, None, None]
    NsM, NsR = len(sigma_M_grid), len(sigma_R_grid)

    evidence_sigma_MR = np.zeros((NsM, NsR))
    P_MR_given_sigma = np.zeros((NsM, NsR, len(M_grid), len(R_grid)))

    for iM in range(NsM):
        for iR in range(NsR):
            b_stack = np.array([b_dict[key][iM, iR] for key in MODEL_KEYS])
            g_exp = g_stack[:, None, :, :]
            b_exp = b_stack[:, None, :, :]
            p_exp = p_grid[None, :, None, None]

            mixture = p_exp * g_exp + (1 - p_exp) * b_exp
            mixture_w = np.clip(mixture, 1e-300, None) ** w

            L_p = np.prod(mixture_w, axis=0)
            likelihood = trapezoid(L_p, p_grid, axis=0)
            Z = trapezoid(trapezoid(likelihood, R_grid, axis=1), M_grid)

            evidence_sigma_MR[iM, iR] = Z
            P_MR_given_sigma[iM, iR] = likelihood / Z

    posterior_sigma_MR = evidence_sigma_MR * prior_sigma_M[:, None] * prior_sigma_R[None, :]
    posterior_sigma_MR /= trapezoid(trapezoid(posterior_sigma_MR, sigma_R_grid, axis=1), sigma_M_grid)

    posterior = trapezoid(
        trapezoid(P_MR_given_sigma * posterior_sigma_MR[:, :, None, None], sigma_R_grid, axis=1),
        sigma_M_grid, axis=0,
    )
    posterior /= trapezoid(trapezoid(posterior, R_grid, axis=1), M_grid)

    P_M = trapezoid(posterior, R_grid, axis=1)
    P_M /= trapezoid(P_M, M_grid)
    P_R = trapezoid(posterior, M_grid, axis=0)
    P_R /= trapezoid(P_R, R_grid)

    def quantiles(grid, pdf):
        cdf = np.cumsum(pdf) * (grid[1] - grid[0])
        cdf /= cdf[-1]
        return np.interp([0.16, 0.50, 0.84], cdf, grid)

    M16, M50, M84 = quantiles(M_grid, P_M)
    R16, R50, R84 = quantiles(R_grid, P_R)
    return dict(
        posterior=posterior, P_M=P_M, P_R=P_R,
        M_median=M50, M_lo=M50 - M16, M_hi=M84 - M50,
        R_median=R50, R_lo=R50 - R16, R_hi=R84 - R50,
    )


def design_effect_power(n, rho):
    return 1.0 / (1.0 + (n - 1) * rho)


def global_weights(rho):
    n = len(MODEL_KEYS)
    w = design_effect_power(n, rho)
    return {key: w for key in MODEL_KEYS}


def grouped_weights(rho):
    weights = {}
    for members in GROUPS.values():
        w = design_effect_power(len(members), rho)
        for key in members:
            weights[key] = w
    return weights


def referee_informed_weights():
    n = len(MODEL_KEYS)
    idx = {key: i for i, key in enumerate(MODEL_KEYS)}
    R_mat = np.eye(n)

    riley_miller = ["ST_PST19", "2spot", "3spot"]
    vinciguerra = ["ST_PDT", "ST_PST", "PDT_U", "ST_U"]
    kini = ["PDT_U26"]

    def set_rho(a, b, rho):
        R_mat[idx[a], idx[b]] = rho
        R_mat[idx[b], idx[a]] = rho

    for a in riley_miller:
        for b in riley_miller:
            if a != b:
                set_rho(a, b, 1.0)
    for a in vinciguerra:
        for b in vinciguerra:
            if a != b:
                set_rho(a, b, 1.0)
    for a in riley_miller:
        for b in vinciguerra:
            set_rho(a, b, 0.98)
    for a in riley_miller + vinciguerra:
        for b in kini:
            set_rho(a, b, 1 / 1.5)

    n_eff = n ** 2 / R_mat.sum()
    w = n_eff / n
    return {key: w for key in MODEL_KEYS}, n_eff


print("\nRunning scenarios (manuscript sanity check first)...")
manuscript = combine_2d({key: 1.0 for key in MODEL_KEYS})
print("=" * 70)
print("SANITY CHECK against manuscript.tex Sec 3.2 (note: coarser grid than the")
print("original notebook, so agreement is approximate, not exact):")
print(f"  script:     M = {manuscript['M_median']:.3f} +{manuscript['M_hi']:.3f}/-{manuscript['M_lo']:.3f}   "
      f"R = {manuscript['R_median']:.3f} +{manuscript['R_hi']:.3f}/-{manuscript['R_lo']:.3f}")
print(f"  manuscript: M = {MANUSCRIPT_M['median']:.3f} +{MANUSCRIPT_M['hi']:.3f}/-{MANUSCRIPT_M['lo']:.3f}   "
      f"R = {MANUSCRIPT_R['median']:.3f} +{MANUSCRIPT_R['hi']:.3f}/-{MANUSCRIPT_R['lo']:.3f}")
print("=" * 70)

rho_values = np.linspace(0.0, 1.0, 6)
print(f"\nSweeping rho = {list(np.round(rho_values, 2))} ...")

global_results = [combine_2d(global_weights(r)) for r in rho_values]
print("  global sweep done")
grouped_results = [combine_2d(grouped_weights(r)) for r in rho_values]
print("  grouped sweep done")

ref_weights, ref_n_eff = referee_informed_weights()
referee_result = combine_2d(ref_weights)
grouped_rho1 = combine_2d(grouped_weights(1.0))
global_rho1 = combine_2d(global_weights(1.0))
print("  discrete scenarios done")

print(f"\nreferee_informed effective N = {ref_n_eff:.2f} (nominal N=8)")
print("\nScenario                          M (median)     R (median)     M-width/ms    R-width/ms")
manuscript_M_width = manuscript["M_lo"] + manuscript["M_hi"]
manuscript_R_width = manuscript["R_lo"] + manuscript["R_hi"]

def _row(label, res):
    mw = (res["M_lo"] + res["M_hi"]) / manuscript_M_width
    rw = (res["R_lo"] + res["R_hi"]) / manuscript_R_width
    print(f"{label:<34} {res['M_median']:.3f}          {res['R_median']:.3f}          {mw:.2f}x         {rw:.2f}x")

_row("manuscript (rho=0)", manuscript)
_row("grouped_by_paper, rho=1", grouped_rho1)
_row("referee_informed", referee_result)
_row("global, rho=1", global_rho1)

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

widths_M_global = [(r["M_lo"] + r["M_hi"]) / manuscript_M_width for r in global_results]
widths_M_grouped = [(r["M_lo"] + r["M_hi"]) / manuscript_M_width for r in grouped_results]
widths_R_global = [(r["R_lo"] + r["R_hi"]) / manuscript_R_width for r in global_results]
widths_R_grouped = [(r["R_lo"] + r["R_hi"]) / manuscript_R_width for r in grouped_results]

ax = axes[0]
ax.plot(rho_values, widths_M_global, "o-", color="black", label="global, all 8")
ax.plot(rho_values, widths_M_grouped, "s-", color="tab:blue", label="grouped by paper")
ax.scatter([1.0], [(referee_result["M_lo"] + referee_result["M_hi"]) / manuscript_M_width],
           marker="*", s=250, color="tab:red", zorder=5, label="referee-informed")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel(r"assumed within-group correlation $\rho$")
ax.set_ylabel("68% width / manuscript width")
ax.set_title("Mass posterior width", fontsize=13)
ax.legend(fontsize=9)

ax2 = axes[1]
ax2.plot(rho_values, widths_R_global, "o-", color="black", label="global, all 8")
ax2.plot(rho_values, widths_R_grouped, "s-", color="tab:blue", label="grouped by paper")
ax2.scatter([1.0], [(referee_result["R_lo"] + referee_result["R_hi"]) / manuscript_R_width],
            marker="*", s=250, color="tab:red", zorder=5, label="referee-informed")
ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax2.set_xlabel(r"assumed within-group correlation $\rho$")
ax2.set_ylabel("68% width / manuscript width")
ax2.set_title("Radius posterior width", fontsize=13)
ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "MR_width_vs_rho.png"), dpi=200)

fig2, (axM, axR) = plt.subplots(1, 2, figsize=(13, 5))
for label, res, color in [
    ("manuscript", manuscript, "black"),
    ("grouped_by_paper, rho=1", grouped_rho1, "tab:blue"),
    ("referee_informed", referee_result, "tab:red"),
    ("global, rho=1", global_rho1, "tab:gray"),
]:
    axM.plot(M_grid, res["P_M"], color=color, lw=2, label=label)
    axR.plot(R_grid, res["P_R"], color=color, lw=2, label=label)
axM.set_xlabel("Mass ($M_\\odot$)")
axM.set_ylabel("Density")
axM.set_title("Marginal mass posterior")
axM.legend(fontsize=9)
axR.set_xlabel("Radius (km)")
axR.set_title("Marginal radius posterior")
axR.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "MR_posteriors_by_scenario.png"), dpi=200)


def rgba(color, alpha):
    return mpl.colors.to_rgba(color, alpha)


def compute_hdr_levels(pdf, probs=(0.68, 0.95)):
    flat = pdf.ravel()
    idx = np.argsort(flat)[::-1]
    sorted_pdf = flat[idx]
    cumsum = np.cumsum(sorted_pdf)
    cumsum /= cumsum[-1]
    return [sorted_pdf[np.searchsorted(cumsum, p)] for p in probs]


fig3, ax3 = plt.subplots(figsize=(8, 7))
for label, res, color in [
    ("manuscript", manuscript, "black"),
    ("grouped_by_paper, rho=1", grouped_rho1, "tab:blue"),
    ("referee_informed", referee_result, "tab:red"),
    ("global, rho=1", global_rho1, "tab:gray"),
]:
    lvl68, lvl95 = compute_hdr_levels(res["posterior"], (0.68, 0.95))
    ax3.contourf(
        R_grid, M_grid, res["posterior"],
        levels=[lvl95, lvl68, res["posterior"].max()],
        colors=[rgba(color, 0.10), rgba(color, 0.20)],
        antialiased=True,
    )
    ax3.contour(
        R_grid, M_grid, res["posterior"],
        levels=[lvl95, lvl68],
        colors=[rgba(color, 0.8)],
        linewidths=[1.3, 1.8],
        linestyles=["--", "-"],
    )

ax3.set_xlabel("Radius (km)")
ax3.set_ylabel("Mass ($M_\\odot$)")
ax3.set_title("Combined M--R posterior by scenario\n(dashed = 95% HDR, solid = 68% HDR)", fontsize=12)
ax3.minorticks_on()
ax3.tick_params(which="both", top=True, right=True)
ax3.legend(
    handles=[
        mpl.lines.Line2D([], [], color="black", lw=2, label="manuscript"),
        mpl.lines.Line2D([], [], color="tab:blue", lw=2, label="grouped_by_paper, rho=1"),
        mpl.lines.Line2D([], [], color="tab:red", lw=2, label="referee_informed"),
        mpl.lines.Line2D([], [], color="tab:gray", lw=2, label="global, rho=1"),
    ],
    fontsize=9, loc="upper left", frameon=True, framealpha=0.85,
)
fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, "MR_contours_by_scenario.png"), dpi=200)

print(f"\nFigures written to {FIG_DIR}")
