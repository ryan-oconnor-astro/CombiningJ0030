"""
Correlation-tempered KDE pooling with good/bad measurement components (2D
mass-radius case).

Direct (M, R) counterpart to tests/correlation_tempered_pooling_1d.py -- see
that file's docstring for the full method description (kappa(rho) tempering,
rho and sigma_sys marginalization, deviations from
correlation_tempered_kde_pooling_framework.pdf's literal pseudocode). This
file applies the same construction to the 2D good/bad pipeline from
2D_MassRadius_Combination.ipynb / tests/independence_sensitivity_2d.py, so the
manuscript's actual headline numbers (M = 1.46 +0.09/-0.08, R = 12.69
+0.64/-0.55) get a tempered-pooling counterpart, not just the 1D compactness
proxy.

Three deliberate simplifications relative to the PDF's Sec. 4 (2D) formalism,
confirmed with Ryan (2026-09-21) before writing this file:
  - Coordinate space is (M, R), not the PDF's (C, M) -- matches
    2D_MassRadius_Combination.ipynb and independence_sensitivity_2d.py
    directly, so scenario numbers are directly comparable to the manuscript's
    headline M, R values without a C->R back-transform.
  - The bad-component covariance Sigma_B,i stays DIAGONAL (independent
    Gaussians in M and R, exactly as in the manuscript notebook and
    independence_sensitivity_2d.py), not the PDF's recommended full
    covariance. This does NOT yet address the referee's secondary criticism
    that Sigma_S,i should have M-R cross-correlation -- deferred, not solved.
  - The group-level systematic Sigma_sys also stays DIAGONAL (no r_sys
    correlation term), rather than the PDF's full 2x2 with
    r_sys ~ Uniform(-1,1). This keeps the systematic convolution
    axis-aligned and, crucially, SEPARABLE: see marginalize_sigma_sys_2d
    below for why this makes the 2D marginalization as cheap as two
    sequential 1D marginalizations rather than a joint 2D grid.

Grids (sigma_M_grid, sigma_R_grid, alpha_grid, p_grid, M_grid, R_grid) reuse
independence_sensitivity_2d.py's already-validated reduced resolution
(relative to the finer manuscript-notebook grids) so the rho sweep stays
tractable. RHO_GRID uses 11 points rather than the 1D script's 21 -- each
combine_2d_given_kappa call is intrinsically more expensive (2D grid, two
sigma dimensions) than its 1D counterpart, so the rho-sweep resolution is
deliberately halved. See the runtime note printed at the bottom of this
script for measured cost; re-run with a finer RHO_GRID only after checking
with Ryan.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import beta as beta_dist
from scipy.stats import halfnorm, norm, gaussian_kde

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, "datafiles")
FIG_DIR = os.path.join(HERE, "correlation_tempered_pooling_figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = {
    "ST_PST19": dict(file="ST_PST19", color="tab:pink",   label="Riley19: ST+PST"),
    "ST_PDT":   dict(file="ST_PDT",   color="tab:blue",   label="Vinciguerra24: ST+PDT"),
    "ST_PST":   dict(file="ST_PST",   color="tab:orange", label="Vinciguerra24: ST+PST"),
    "PDT_U":    dict(file="PDT_U",    color="tab:green",  label="Vinciguerra24: PDT--U"),
    "ST_U":     dict(file="ST_U",     color="tab:cyan",   label="Vinciguerra24: ST--U"),
    "2spot":    dict(file="2spot",    color="tab:red",    label="Miller19: 2spot"),
    "3spot":    dict(file="3spot",    color="tab:purple", label="Miller19: 3spot"),
    "PDT_U26":  dict(file="PDT_U26",  color="tab:olive",  label="Kini26: PDT--U"),
}
MODEL_KEYS = list(MODELS)
N_MODELS = len(MODEL_KEYS)

MANUSCRIPT_M = dict(median=1.46, lo=0.08, hi=0.09)   # manuscript.tex Sec 3.2
MANUSCRIPT_R = dict(median=12.69, lo=0.55, hi=0.64)  # manuscript.tex Sec 3.2

# ----------------------------------------------------------------------------
# Load data, build good (KDE) densities on the (M, R) grid -- identical
# construction to independence_sensitivity_2d.py
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

# Data-anchored scales for the group-level systematic priors (PDF Eqs. 46-47),
# analogous to S_C_SYS_SCALE in the 1D script.
S_M_SYS_SCALE = np.std(list(M_med.values()))
S_R_SYS_SCALE = np.std(list(R_med.values()))

# Grid: same fixed domain independence_sensitivity_2d.py already validated
# (approximately reproduces the manuscript M, R numbers at rho=0 -- see the
# sanity check printed below), which already carries margin beyond the raw
# per-model sample ranges. That margin is what the 1D script's C_grid
# deliberately did NOT have (padding broke that script's sanity check); here
# the margin comes for free from the fixed domain and was already shown not
# to break the (coarser-grid) (M,R) sanity check in independence_sensitivity_2d.py.
M_grid = np.linspace(0.95, 2.2, 80)
R_grid = np.linspace(9.0, 17.0, 80)
MM, RR = np.meshgrid(M_grid, R_grid, indexing="ij")
dM = M_grid[1] - M_grid[0]
dR = R_grid[1] - R_grid[0]
pos = np.vstack([MM.ravel(), RR.ravel()])

g_dict = {}
for key in MODEL_KEYS:
    g = KDEs_2D[key](pos).reshape(MM.shape)
    g /= np.trapz(np.trapz(g, R_grid, axis=1), M_grid)
    g_dict[key] = g
g_stack = np.array([g_dict[key] for key in MODEL_KEYS])  # (8, NM, NR)

# Hyperparameter grids -- identical reduced resolution to
# independence_sensitivity_2d.py (already validated to keep a multi-point rho
# sweep tractable).
avg_S_Mi = np.mean(list(S_Mi.values()))
avg_S_Ri = np.mean(list(S_Ri.values()))
sigma_M_grid = np.logspace(-10 * avg_S_Mi, -0.1 * avg_S_Mi, 6)
sigma_R_grid = np.logspace(-10 * avg_S_Ri, -0.1 * avg_S_Ri, 6)
alpha_grid = np.linspace(1.0, 10, 40)
p_grid = np.linspace(0.0, 1.0, 25)

prior_sigma_M = 1.0 / sigma_M_grid
prior_sigma_M /= np.trapz(prior_sigma_M, sigma_M_grid)
prior_sigma_R = 1.0 / sigma_R_grid
prior_sigma_R /= np.trapz(prior_sigma_R, sigma_R_grid)
prior_alpha = np.exp(-alpha_grid)
prior_alpha /= np.trapz(prior_alpha, alpha_grid)

# Bad distributions b_dict[key]: shape (NsM, NsR, NM, NR). Independent of
# kappa/rho/sigma_sys, so built once and reused for every scenario below.
print("Building bad-component grids for all 8 models (kappa-independent, cached)...")
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
            norm_const = np.trapz(np.trapz(b_alpha, R_grid, axis=2), M_grid, axis=1)
            b_alpha /= norm_const[:, None, None]

            b_marg = np.trapz(b_alpha * prior_alpha[:, None, None], alpha_grid, axis=0)
            b_marg /= np.trapz(np.trapz(b_marg, R_grid, axis=1), M_grid)
            row_sigma_R.append(b_marg)
        b_sigma.append(row_sigma_R)
    b_dict[key] = np.array(b_sigma)  # (NsM, NsR, NM, NR)
print("Done.")


def design_effect_power(n, rho):
    """kappa(rho) = 1 / (1 + (n-1)*rho); PDF Eq. 6. Identical to the 1D script."""
    return 1.0 / (1.0 + (n - 1) * rho)


def combine_2d_given_kappa(kappa):
    """Good/bad combination with every measurement's mixture raised to the
    same power kappa before the product over models -- 2D counterpart of
    combine_1d_given_kappa. kappa=1 reproduces the manuscript (untempered)
    method exactly (same computation as independence_sensitivity_2d.py's
    combine_2d with a uniform weight dict). Returns the final posterior
    density on the (M, R) grid, normalized to 1.
    """
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
            mixture_w = np.clip(mixture, 1e-300, None) ** kappa

            L_p = np.prod(mixture_w, axis=0)
            likelihood = np.trapz(L_p, p_grid, axis=0)
            Z = np.trapz(np.trapz(likelihood, R_grid, axis=1), M_grid)

            evidence_sigma_MR[iM, iR] = Z
            P_MR_given_sigma[iM, iR] = likelihood / Z

    posterior_sigma_MR = evidence_sigma_MR * prior_sigma_M[:, None] * prior_sigma_R[None, :]
    posterior_sigma_MR /= np.trapz(np.trapz(posterior_sigma_MR, sigma_R_grid, axis=1), sigma_M_grid)

    final_posterior = np.trapz(
        np.trapz(P_MR_given_sigma * posterior_sigma_MR[:, :, None, None], sigma_R_grid, axis=1),
        sigma_M_grid, axis=0,
    )
    final_posterior /= np.trapz(np.trapz(final_posterior, R_grid, axis=1), M_grid)
    return final_posterior


def quantiles_2d(posterior):
    P_M = np.trapz(posterior, R_grid, axis=1)
    P_M /= np.trapz(P_M, M_grid)
    P_R = np.trapz(posterior, M_grid, axis=0)
    P_R /= np.trapz(P_R, R_grid)

    def _q(grid, pdf, spacing):
        cdf = np.cumsum(pdf) * spacing
        cdf /= cdf[-1]
        return np.interp([0.16, 0.50, 0.84], cdf, grid)

    M16, M50, M84 = _q(M_grid, P_M, dM)
    R16, R50, R84 = _q(R_grid, P_R, dR)
    return dict(
        P_M=P_M, P_R=P_R,
        M_median=M50, M_lo=M50 - M16, M_hi=M84 - M50,
        R_median=R50, R_lo=R50 - R16, R_hi=R84 - R50,
    )


# ----------------------------------------------------------------------------
# New: marginalize over rho (correlation tempering)
# ----------------------------------------------------------------------------

RHO_GRID = np.linspace(0.0, 1.0, 11)


def rho_prior_beta41(rho):
    return beta_dist.pdf(rho, 4, 1)


def rho_prior_uniform(rho):
    return np.ones_like(rho)


def compute_rho_sweep():
    posts = []
    for rho in RHO_GRID:
        posts.append(combine_2d_given_kappa(design_effect_power(N_MODELS, rho)))
    return np.array(posts)  # (n_rho, NM, NR)


print(f"\nSweeping rho over {len(RHO_GRID)} points (this is the expensive step)...")
_t0 = time.time()
RHO_SWEEP_POSTERIORS = compute_rho_sweep()
print(f"  done in {time.time() - _t0:.1f} s")


def marginalize_rho(prior_fn):
    """Marginal p(M,R) = integral over rho of p(M,R|rho) * pi(rho) drho -- 2D
    counterpart of the 1D script's marginalize_rho. Same convex-combination
    argument applies: pi(rho) integrates to 1 and each p(M,R|rho) is already
    a proper density, so this is a genuine mixture, not an evidence-reweighted
    average."""
    prior_vals = prior_fn(RHO_GRID)
    prior_vals /= np.trapz(prior_vals, RHO_GRID)
    marginal = np.trapz(RHO_SWEEP_POSTERIORS * prior_vals[:, None, None], RHO_GRID, axis=0)
    marginal /= np.trapz(np.trapz(marginal, R_grid, axis=1), M_grid)
    return marginal


# ----------------------------------------------------------------------------
# New: marginalize over sigma_sys (group-level systematic convolution),
# DIAGONAL Sigma_sys only (no r_sys cross-term -- see module docstring).
# ----------------------------------------------------------------------------

SIGMA_M_SYS_GRID = np.linspace(1e-6, 4.0 * S_M_SYS_SCALE, 15)
SIGMA_R_SYS_GRID = np.linspace(1e-6, 4.0 * S_R_SYS_SCALE, 15)


def _convolve_axis(density, sigma_phys, axis, spacing):
    if sigma_phys <= 0:
        return density
    sigma_pix = sigma_phys / spacing
    smoothed = gaussian_filter1d(density, sigma=sigma_pix, axis=axis, mode="constant", cval=0.0)
    smoothed /= np.trapz(np.trapz(smoothed, R_grid, axis=1), M_grid)
    return smoothed


def marginalize_sigma_sys_2d(density_in, scale_M, scale_R):
    """Marginalize over a DIAGONAL Sigma_sys = diag(sigma_M_sys^2, sigma_R_sys^2)
    as two SEQUENTIAL 1D marginalizations (R axis, then M axis) rather than a
    joint (sigma_M_sys, sigma_R_sys) grid.

    This is valid, not just convenient: for a diagonal covariance the 2D
    Gaussian convolution kernel factors into independent 1D kernels along M
    and R, so the convolution operators T_M(sigma_M) and T_R(sigma_R) act on
    orthogonal axes and commute. By linearity of convolution and Fubini's
    theorem, the double integral over the independent priors pi_M, pi_R
    factors as

        integral[ T_M(s_M) T_R(s_R) L * piM(s_M) piR(s_R) ] ds_M ds_R
      = integral[ piM(s_M) T_M(s_M) [ integral( piR(s_R) T_R(s_R) L ) ds_R ] ] ds_M
      = M_marginalize( R_marginalize( L ) ),

    i.e. exactly two sequential applications of the 1D script's
    marginalize_sigma_sys, one per axis, instead of a 15x15 joint grid. If
    Sigma_sys ever gets an r_sys cross-term this shortcut no longer applies
    (the kernel stops being separable) and the joint-grid version would be
    needed.
    """
    prior_R = halfnorm.pdf(SIGMA_R_SYS_GRID, scale=scale_R)
    prior_R /= np.trapz(prior_R, SIGMA_R_SYS_GRID)
    conv_R_stack = np.array([_convolve_axis(density_in, s, axis=1, spacing=dR) for s in SIGMA_R_SYS_GRID])
    marginal_R = np.trapz(conv_R_stack * prior_R[:, None, None], SIGMA_R_SYS_GRID, axis=0)
    marginal_R /= np.trapz(np.trapz(marginal_R, R_grid, axis=1), M_grid)

    prior_M = halfnorm.pdf(SIGMA_M_SYS_GRID, scale=scale_M)
    prior_M /= np.trapz(prior_M, SIGMA_M_SYS_GRID)
    conv_M_stack = np.array([_convolve_axis(marginal_R, s, axis=0, spacing=dM) for s in SIGMA_M_SYS_GRID])
    marginal = np.trapz(conv_M_stack * prior_M[:, None, None], SIGMA_M_SYS_GRID, axis=0)
    marginal /= np.trapz(np.trapz(marginal, R_grid, axis=1), M_grid)
    return marginal


# ----------------------------------------------------------------------------
# Diagnostic scenarios (PDF Sec. 7), same 5 as the 1D script
# ----------------------------------------------------------------------------

manuscript = RHO_SWEEP_POSTERIORS[0]                                        # rho=0, sigma_sys=0
fully_correlated = RHO_SWEEP_POSTERIORS[-1]                                 # rho=1, sigma_sys=0
rho_uniform = marginalize_rho(rho_prior_uniform)                           # rho~U(0,1), sigma_sys=0
rho_beta41_nosys = marginalize_rho(rho_prior_beta41)                       # rho~Beta(4,1), sigma_sys=0
rho_beta41_sys = marginalize_sigma_sys_2d(rho_beta41_nosys, S_M_SYS_SCALE, S_R_SYS_SCALE)  # main conservative

scenarios = {
    "manuscript (rho=0, sigma_sys=0)": manuscript,
    "fully correlated (rho=1, sigma_sys=0)": fully_correlated,
    "rho ~ Uniform(0,1), sigma_sys=0": rho_uniform,
    "rho ~ Beta(4,1), sigma_sys=0": rho_beta41_nosys,
    "rho ~ Beta(4,1), sigma_sys marginalized": rho_beta41_sys,
}

# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

q_manuscript = quantiles_2d(manuscript)
print("=" * 90)
print("SANITY CHECK against manuscript.tex Sec 3.2 (rho=0, sigma_sys=0 limit; note this")
print("uses the SAME coarser grid as independence_sensitivity_2d.py, so agreement is")
print("approximate, not bit-for-bit like the 1D script's C sanity check):")
print(f"  script:     M = {q_manuscript['M_median']:.3f} +{q_manuscript['M_hi']:.3f}/-{q_manuscript['M_lo']:.3f}   "
      f"R = {q_manuscript['R_median']:.3f} +{q_manuscript['R_hi']:.3f}/-{q_manuscript['R_lo']:.3f}")
print(f"  manuscript: M = {MANUSCRIPT_M['median']:.3f} +{MANUSCRIPT_M['hi']:.3f}/-{MANUSCRIPT_M['lo']:.3f}   "
      f"R = {MANUSCRIPT_R['median']:.3f} +{MANUSCRIPT_R['hi']:.3f}/-{MANUSCRIPT_R['lo']:.3f}")
print("=" * 90)

print(f"\nData-driven systematic scales: s_M = std(median(M_i)) = {S_M_SYS_SCALE:.4f}, "
      f"s_R = std(median(R_i)) = {S_R_SYS_SCALE:.4f}")

manuscript_M_width = q_manuscript["M_lo"] + q_manuscript["M_hi"]
manuscript_R_width = q_manuscript["R_lo"] + q_manuscript["R_hi"]

quantiles_by_scenario = {label: quantiles_2d(post) for label, post in scenarios.items()}

print("\nScenario                                             M (median)   R (median)   M-width/ms    R-width/ms")
for label, q in quantiles_by_scenario.items():
    mw = (q["M_lo"] + q["M_hi"]) / manuscript_M_width
    rw = (q["R_lo"] + q["R_hi"]) / manuscript_R_width
    print(f"{label:<52} {q['M_median']:.3f}        {q['R_median']:.3f}        {mw:.2f}x         {rw:.2f}x")

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

plt.rcParams.update({"font.size": 13})
colors = ["black", "tab:gray", "tab:blue", "tab:orange", "tab:red"]

fig, (axM, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
for (label, q), color in zip(quantiles_by_scenario.items(), colors):
    axM.plot(M_grid, q["P_M"], color=color, lw=2, label=label)
    axR.plot(R_grid, q["P_R"], color=color, lw=2, label=label)
axM.set_xlabel("Mass ($M_\\odot$)")
axM.set_ylabel("Density")
axM.set_title("Marginal mass posterior")
axM.legend(fontsize=8)
axR.set_xlabel("Radius (km)")
axR.set_title("Marginal radius posterior")
axR.legend(fontsize=8)
fig.suptitle("Combined M-R posterior: correlation-tempered pooling diagnostic scenarios")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "MR_posteriors_diagnostic_scenarios.png"), dpi=200)


def rgba(color, alpha):
    return mpl.colors.to_rgba(color, alpha)


def compute_hdr_levels(pdf, probs=(0.68, 0.95)):
    flat = pdf.ravel()
    idx = np.argsort(flat)[::-1]
    sorted_pdf = flat[idx]
    cumsum = np.cumsum(sorted_pdf)
    cumsum /= cumsum[-1]
    return [sorted_pdf[np.searchsorted(cumsum, p)] for p in probs]


CONTOUR_SCENARIOS = [
    "manuscript (rho=0, sigma_sys=0)",
    "fully correlated (rho=1, sigma_sys=0)",
    "rho ~ Uniform(0,1), sigma_sys=0",
]

fig2, ax2 = plt.subplots(figsize=(8, 7))
legend_handles = []
for (label, post), color in zip(scenarios.items(), colors):
    if label not in CONTOUR_SCENARIOS:
        continue
    lvl68, lvl95 = compute_hdr_levels(post, (0.68, 0.95))
    ax2.contourf(
        R_grid, M_grid, post,
        levels=[lvl95, lvl68, post.max()],
        colors=[rgba(color, 0.10), rgba(color, 0.20)],
        antialiased=True,
    )
    ax2.contour(
        R_grid, M_grid, post,
        levels=[lvl95, lvl68],
        colors=[rgba(color, 0.8)],
        linewidths=[1.3, 1.8],
        linestyles=["--", "-"],
    )
    legend_handles.append(mpl.lines.Line2D([], [], color=color, lw=2, label=label))
ax2.set_xlabel("Radius (km)")
ax2.set_ylabel("Mass ($M_\\odot$)")
ax2.set_title("Combined M-R posterior by scenario\n(dashed = 95% HDR, solid = 68% HDR)", fontsize=12)
ax2.minorticks_on()
ax2.tick_params(which="both", top=True, right=True)
ax2.legend(handles=legend_handles, fontsize=8, loc="upper left", frameon=True, framealpha=0.85)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "MR_contours_diagnostic_scenarios.png"), dpi=200)

fig3, (ax3M, ax3R) = plt.subplots(1, 2, figsize=(13, 5))
ax3M.plot(M_grid, quantiles_by_scenario["rho ~ Beta(4,1), sigma_sys=0"]["P_M"],
          color="tab:orange", lw=2, label=r"rho ~ Beta(4,1), no $\Sigma_{sys}$")
ax3M.plot(M_grid, quantiles_by_scenario["rho ~ Beta(4,1), sigma_sys marginalized"]["P_M"],
          color="tab:red", lw=2, label=r"rho ~ Beta(4,1), $\Sigma_{sys}$ marginalized")
ax3M.set_xlabel("Mass ($M_\\odot$)")
ax3M.set_ylabel("Density")
ax3M.set_title("Mass: effect of the group-level\nsystematic convolution")
ax3M.legend(fontsize=9)
ax3R.plot(R_grid, quantiles_by_scenario["rho ~ Beta(4,1), sigma_sys=0"]["P_R"],
          color="tab:orange", lw=2, label=r"rho ~ Beta(4,1), no $\Sigma_{sys}$")
ax3R.plot(R_grid, quantiles_by_scenario["rho ~ Beta(4,1), sigma_sys marginalized"]["P_R"],
          color="tab:red", lw=2, label=r"rho ~ Beta(4,1), $\Sigma_{sys}$ marginalized")
ax3R.set_xlabel("Radius (km)")
ax3R.set_title("Radius: effect of the group-level\nsystematic convolution")
ax3R.legend(fontsize=9)
fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, "sigma_sys_effect_MR.png"), dpi=200)

print(f"\nFigures written to {FIG_DIR}")

print(
    "\n"
    "QUESTIONS FOR THE AUTHORS:\n"
    "  1. This script's Sigma_B,i and Sigma_sys are both DIAGONAL (deferred, not\n"
    "     solved, per 2026-09-21 decision) -- the referee's secondary criticism\n"
    "     that the bad-component covariance should have M-R cross-correlation is\n"
    "     still open. Worth a follow-up script/version with full covariance?\n"
    "  2. RHO_GRID uses 11 points here vs. 21 in the 1D script, and the base\n"
    "     (M,R) grid/sigma grids match independence_sensitivity_2d.py's already-\n"
    "     reduced resolution rather than the finer manuscript-notebook grids --\n"
    "     is this resolution sufficient, or should specific scenarios be re-run\n"
    "     at higher resolution once the qualitative picture here is checked?\n"
    "  3. Same open questions as the 1D script: is Beta(4,1) or Uniform(0,1) the\n"
    "     right 'main' rho prior; is std(median) a reasonable sigma_sys scale;\n"
    "     model-weight sensitivity (evidence weights) is still not implemented.\n"
)
