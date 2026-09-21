"""
Correlation-tempered KDE pooling with good/bad measurement components (1D
compactness case).

Implements the "Option A/C" methodology fix described in
correlation_tempered_kde_pooling_framework.pdf (see CLAUDE.md's "Open
direction on the independence fix" note): rather than sweeping or fixing a
handful of discrete correlation scenarios (tests/independence_sensitivity.py,
"Option B"), the inter-posterior correlation rho is treated as a genuine
hyperparameter with its own prior and is marginalized out -- exactly as
sigma/alpha/p are already marginalized in the published method -- alongside a
new group-level systematic width sigma_sys that convolves the pooled
posterior (PDF Eqs. 5-31).

Deviations from the PDF's literal description, and why:
  - The mixing probability p is kept SHARED across all 8 measurements, as in
    the published notebooks / independence_sensitivity.py, rather than given
    an independent p_i per measurement (PDF Eq. 17-18 allows either). This
    isolates rho-tempering + sigma_sys as the only new ingredients relative
    to the published method, so the rho=0, sigma_sys=0 limit reproduces the
    published C = 0.172 result exactly (PDF Sec. 9 checklist item 6). Ryan
    confirmed this choice explicitly (2026-09-10).
  - sigma (bad-component shift-prior width), alpha, and delta remain
    marginalized by the exact same deterministic trapezoid quadrature as the
    published method (not Monte Carlo sampling, as the PDF's pseudocode
    literally suggests). This is mathematically equivalent to prior-
    predictive marginalization: the published code's "evidence-weighted
    posterior over sigma" and a plain prior-weighted average over sigma are
    the same integral (the per-sigma normalization Z(sigma) cancels exactly
    when you substitute one into the other). Reusing the identical grids and
    quadrature lets this script reproduce the published numbers bit-for-bit
    at rho=0, sigma_sys=0, and isolates what's actually new.
  - rho and sigma_sys, being new dimensions with no legacy behavior to
    reproduce, are marginalized with proper trapezoidal integration weights
    (np.trapz over the grid), unlike the published sigma marginalization
    (which sums over its log-spaced grid without a dsigma weighting factor --
    a published quirk this script deliberately does not propagate to the new
    dimensions).

Recommended diagnostic scenarios (PDF Sec. 7), all reproduced below:
  1. Old product endpoint:        rho=0,           sigma_sys=0
  2. Fully correlated endpoint:   rho=1,           sigma_sys=0
  3. Correlation-marginalized:    rho~Uniform(0,1), sigma_sys=0
  4. Conservative main result:    rho~Beta(4,1),    sigma_sys marginalized
  (5. Model-weight sensitivity (equal vs. evidence weights) is NOT
      implemented here -- it needs within-analysis Bayesian evidence values
      for each of the 8 models, which are not present in datafiles/ and
      would need to be pulled from the original papers.)
"""

import os

import matplotlib
matplotlib.use("Agg")
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

G = 6.67430e-11        # m^3 kg^-1 s^-2
c = 2.99792458e8       # m s^-1
M_sun = 1.98847e30     # kg

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

PUBLISHED_C = dict(median=0.172, lo=0.007, hi=0.006)  # manuscript.tex Sec 3.1

# ----------------------------------------------------------------------------
# Load data, build good (KDE) densities for compactness -- identical to
# 1D_Compactness_Combination.ipynb / independence_sensitivity.py
# ----------------------------------------------------------------------------

compactness = {}
C_med, C_lo_raw, C_hi_raw, S_i = {}, {}, {}, {}
KDEs_1D = {}

for key, meta in MODELS.items():
    data = np.loadtxt(os.path.join(DATA_DIR, meta["file"]))
    M, R = data.T[0], data.T[1]
    C = G * (M * M_sun) / (R * 1e3 * c ** 2)
    compactness[key] = C
    C_med[key] = np.median(C)
    C_lo_raw[key] = C.min()
    C_hi_raw[key] = C.max()
    S_i[key] = max(C_hi_raw[key] - C_med[key], C_med[key] - C_lo_raw[key])
    KDEs_1D[key] = gaussian_kde(C, bw_method="silverman")

# Data-anchored scale for the new group-level systematic prior (PDF Eq. 30):
# the scatter of the 8 models' own compactness medians.
S_C_SYS_SCALE = np.std(list(C_med.values()))

# Grid is deliberately IDENTICAL to the published notebook/independence_sensitivity.py
# (raw data range, 200 points, no padding) -- NOT padded for the sigma_sys
# convolution margin the PDF recommends (Sec 4.1). Padding was tried and
# rejected: it changes how the bad component's Gaussian tails get truncated
# and renormalized within-domain, which shifts the rho=0 result enough to
# break the "reproduce the published number" sanity check (0.0128 unpadded
# vs. 0.0093 with a +/-4*S_C_SYS_SCALE pad, against a published width of
# 0.013). Consequence: the sigma_sys convolution below (grid_sys section)
# loses a little mass near the domain edges for models whose posteriors sit
# close to C_min_raw/C_max_raw -- acceptable for this diagnostic script, but
# worth flagging if sigma_sys marginalization becomes the adopted result.
C_min_raw = min(C_lo_raw.values())
C_max_raw = max(C_hi_raw.values())
C_grid = np.linspace(C_min_raw, C_max_raw, 200)
dC = C_grid[1] - C_grid[0]

g_dict = {}
for key in MODEL_KEYS:
    g = KDEs_1D[key].evaluate(C_grid)
    g_dict[key] = g / np.trapz(g, C_grid)
g_stack = np.array([g_dict[key] for key in MODEL_KEYS])  # (8, NC)

# Hyperparameter grids -- identical to the published notebook.
sigma_grid = np.logspace(-4, -0.5, 15)
alpha_grid = np.linspace(1.0, 5.0, 30)
p_grid = np.linspace(0.0, 1.0, 60)

prior_sigma = (1.0 / sigma_grid)
prior_sigma /= np.trapz(prior_sigma, sigma_grid)
prior_alpha = np.exp(-alpha_grid)
prior_alpha /= np.trapz(prior_alpha, alpha_grid)

# Bad distributions b_dict[key]: shape (Nsigma, NC). Independent of rho and
# sigma_sys, so built once and reused for every scenario below.
b_dict = {}
for key in MODEL_KEYS:
    mu, S = C_med[key], S_i[key]
    b_sigma = []
    for sigma in sigma_grid:
        delta_grid = np.linspace(-4 * sigma, 4 * sigma, 25)
        prior_delta = norm.pdf(delta_grid, loc=0, scale=sigma)
        prior_delta /= np.trapz(prior_delta, delta_grid)

        b_alpha = []
        for alpha in alpha_grid:
            b_delta = norm.pdf(C_grid[None, :], loc=mu + delta_grid[:, None], scale=S * alpha)
            b_delta_marg = np.trapz(b_delta * prior_delta[:, None], delta_grid, axis=0)
            b_delta_marg /= np.trapz(b_delta_marg, C_grid)
            b_alpha.append(b_delta_marg)
        b_alpha = np.array(b_alpha)

        b_marg = np.trapz(b_alpha * prior_alpha[:, None], alpha_grid, axis=0)
        b_marg /= np.trapz(b_marg, C_grid)
        b_sigma.append(b_marg)
    b_dict[key] = np.array(b_sigma)  # (Nsigma, NC)


def design_effect_power(n, rho):
    """kappa(rho) = 1 / (1 + (n-1)*rho); PDF Eq. 6. rho=0 -> kappa=1 (no
    tempering, published limit); rho=1 -> kappa=1/n (one effective dataset)."""
    return 1.0 / (1.0 + (n - 1) * rho)


def combine_1d_given_kappa(kappa):
    """Good/bad combination with every measurement's mixture raised to the
    same power kappa before the product over models (PDF Eq. 5 with equal
    weights w_i=1). kappa=1 reproduces the published (untempered) method
    exactly -- same computation as tests/independence_sensitivity.py's
    combine_1d with a uniform weight dict.

    Returns a dict with:
      posterior       -- final combined density over C_grid (normalized to 1)
      posterior_sigma -- posterior weights over sigma_grid (normalized to 1,
                          via the published plain-sum convention); exposed so
                          the per-model credibility diagnostic can reuse the
                          identical sigma-marginalization used here instead
                          of recomputing it
      evidence        -- total marginal evidence Z(kappa) = integral over
                          sigma of evidence_sigma(sigma)*prior_sigma(sigma).
                          NOT used anywhere in the published or tempered
                          posterior itself (that's fully renormalized away);
                          exposed only for the "data-implied rho" diagnostic,
                          which asks how this un-normalized total mass moves
                          with kappa/rho.
    """
    P_C_given_sigma, evidence_sigma = [], []
    for s_index in range(len(sigma_grid)):
        b_stack = np.array([b_dict[key][s_index] for key in MODEL_KEYS])
        g_exp = g_stack[:, None, :]
        b_exp = b_stack[:, None, :]
        p_exp = p_grid[None, :, None]

        mixture = p_exp * g_exp + (1 - p_exp) * b_exp
        mixture_w = np.clip(mixture, 1e-300, None) ** kappa

        L_p = np.prod(mixture_w, axis=0)
        likelihood = np.trapz(L_p, p_grid, axis=0)
        Z = np.trapz(likelihood, C_grid)

        P_C_given_sigma.append(likelihood / Z)
        evidence_sigma.append(Z)

    P_C_given_sigma = np.array(P_C_given_sigma)
    evidence_sigma = np.array(evidence_sigma)

    total_evidence = np.trapz(evidence_sigma * prior_sigma, sigma_grid)

    posterior_sigma = evidence_sigma * prior_sigma
    posterior_sigma /= np.trapz(posterior_sigma, sigma_grid)

    final_posterior = np.sum(P_C_given_sigma * posterior_sigma[:, None], axis=0)
    final_posterior /= np.trapz(final_posterior, C_grid)
    return dict(posterior=final_posterior, posterior_sigma=posterior_sigma, evidence=total_evidence)


def quantiles(posterior):
    cdf = np.cumsum(posterior) * dC
    cdf /= cdf[-1]
    q16, q50, q84 = np.interp([0.16, 0.50, 0.84], cdf, C_grid)
    return dict(median=q50, lo=q50 - q16, hi=q84 - q50)


# ----------------------------------------------------------------------------
# New: marginalize over rho (correlation tempering)
# ----------------------------------------------------------------------------

RHO_GRID = np.linspace(0.0, 1.0, 21)


def rho_prior_beta41(rho):
    return beta_dist.pdf(rho, 4, 1)


def rho_prior_uniform(rho):
    return np.ones_like(rho)


def compute_rho_sweep():
    """Run combine_1d_given_kappa once per RHO_GRID point and cache the
    results -- reused by every rho-related diagnostic below (marginalization
    under different priors, the published/fully-correlated endpoints, and
    the data-implied-rho evidence plot) instead of recomputing the same
    kappa(rho) values redundantly per prior."""
    posts, sigmas, evids = [], [], []
    for rho in RHO_GRID:
        result = combine_1d_given_kappa(design_effect_power(N_MODELS, rho))
        posts.append(result["posterior"])
        sigmas.append(result["posterior_sigma"])
        evids.append(result["evidence"])
    return np.array(posts), np.array(sigmas), np.array(evids)


RHO_SWEEP_POSTERIORS, RHO_SWEEP_POSTERIOR_SIGMA, RHO_SWEEP_EVIDENCE = compute_rho_sweep()


def marginalize_rho(prior_fn):
    """Marginal p(C) = integral over rho of p(C|rho) * pi(rho) drho, with
    p(C|rho) itself a proper density (already normalized in C by
    combine_1d_given_kappa). Since pi(rho) integrates to 1 and each p(C|rho)
    integrates to 1 in C, this is a genuine convex combination of proper
    densities -- correct without any evidence reweighting of rho itself
    (unlike sigma, rho has no data-informed update here; it is marginalized
    purely against its prior, matching PDF Eq. 9)."""
    prior_vals = prior_fn(RHO_GRID)
    prior_vals /= np.trapz(prior_vals, RHO_GRID)
    marginal = np.trapz(RHO_SWEEP_POSTERIORS * prior_vals[:, None], RHO_GRID, axis=0)
    marginal /= np.trapz(marginal, C_grid)
    return marginal


# ----------------------------------------------------------------------------
# New: marginalize over sigma_sys (group-level systematic convolution)
# ----------------------------------------------------------------------------

SIGMA_SYS_GRID = np.linspace(1e-6, 4.0 * S_C_SYS_SCALE, 25)


def convolve_gaussian_1d(density, sigma_phys):
    if sigma_phys <= 0:
        return density
    sigma_pix = sigma_phys / dC
    smoothed = gaussian_filter1d(density, sigma=sigma_pix, mode="constant", cval=0.0)
    smoothed /= np.trapz(smoothed, C_grid)
    return smoothed


def marginalize_sigma_sys(density_in, scale):
    prior_vals = halfnorm.pdf(SIGMA_SYS_GRID, scale=scale)
    prior_vals /= np.trapz(prior_vals, SIGMA_SYS_GRID)
    conv_stack = np.array([convolve_gaussian_1d(density_in, s) for s in SIGMA_SYS_GRID])
    marginal = np.trapz(conv_stack * prior_vals[:, None], SIGMA_SYS_GRID, axis=0)
    marginal /= np.trapz(marginal, C_grid)
    return marginal


# ----------------------------------------------------------------------------
# Diagnostic scenarios (PDF Sec. 7)
# ----------------------------------------------------------------------------

published = RHO_SWEEP_POSTERIORS[0]                                        # rho=0 (first RHO_GRID point), sigma_sys=0
fully_correlated = RHO_SWEEP_POSTERIORS[-1]                                # rho=1 (last RHO_GRID point), sigma_sys=0
rho_uniform = marginalize_rho(rho_prior_uniform)                           # rho~U(0,1), sigma_sys=0
rho_beta41_nosys = marginalize_rho(rho_prior_beta41)                       # rho~Beta(4,1), sigma_sys=0
rho_beta41_sys = marginalize_sigma_sys(rho_beta41_nosys, S_C_SYS_SCALE)    # main conservative result

scenarios = {
    "published (rho=0, sigma_sys=0)": published,
    "fully correlated (rho=1, sigma_sys=0)": fully_correlated,
    "rho ~ Uniform(0,1), sigma_sys=0": rho_uniform,
    "rho ~ Beta(4,1), sigma_sys=0": rho_beta41_nosys,
    "rho ~ Beta(4,1), sigma_sys marginalized": rho_beta41_sys,
}

# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

print("=" * 78)
print("SANITY CHECK against manuscript.tex Sec 3.1 (rho=0, sigma_sys=0 limit):")
q_pub = quantiles(published)
print(f"  script:     C = {q_pub['median']:.4f} +{q_pub['hi']:.4f} / -{q_pub['lo']:.4f}")
print(f"  manuscript: C = {PUBLISHED_C['median']:.4f} +{PUBLISHED_C['hi']:.4f} / -{PUBLISHED_C['lo']:.4f}")
print("=" * 78)

print(f"\nData-driven group-systematic scale s_C = std(median(C_i)) = {S_C_SYS_SCALE:.5f}")

pub_width = q_pub["lo"] + q_pub["hi"]
print("\nScenario                                             C (median)   width (68%)   width / published")
for label, post in scenarios.items():
    q = quantiles(post)
    width = q["lo"] + q["hi"]
    print(f"{label:<52} {q['median']:.4f}       {width:.4f}        {width/pub_width:.2f}x")

# ----------------------------------------------------------------------------
# New diagnostic: data-implied rho (evidence vs. rho)
# ----------------------------------------------------------------------------
# NOT used to set rho in any scenario above -- rho is deliberately marginalized
# against a fixed prior (Beta(4,1)/Uniform), not fit to the data. This block
# checks empirically what a naive evidence-based choice of rho WOULD do, and
# the result is worth stating precisely rather than guessing: Z(rho) turns out
# to be monotonically DECREASING in rho (maximized at rho=0, i.e. the
# published independence assumption) -- the opposite of the first guess
# written here originally ("tempering trivially raises the pooled integral").
# The actual mechanism: most of the 8 good KDEs peak in overlapping regions of
# C, so the untempered product (kappa=1) reinforces that overlap into a very
# sharp, very large pointwise density there; tempering (kappa<1) suppresses
# exactly that reinforcement. So a naive evidence-maximizing choice of rho
# would always snap back to rho=0 -- i.e. to treating the 8 measurements as
# fully independent, which is precisely the overconfident, pseudo-replication-
# prone limit the referee objected to. This is a real, useful point for the
# response letter: it is not merely inconvenient to let evidence pick rho,
# it is actively circular, since the naive evidence comparison itself
# encodes the same double-counting the correlation-tempering is meant to fix.

log_evidence_ratio = np.log(RHO_SWEEP_EVIDENCE) - np.log(RHO_SWEEP_EVIDENCE[0])
print("\nData-implied rho: log[Z(rho)/Z(0)] (negative = data 'prefers' LESS tempering, i.e. rho=0)")
for rho, dlogZ in zip(RHO_GRID[::4], log_evidence_ratio[::4]):
    print(f"  rho = {rho:.2f}   log[Z(rho)/Z(0)] = {dlogZ:+.3f}")
monotonic_decreasing = np.all(np.diff(RHO_SWEEP_EVIDENCE) <= 1e-12)
print(f"  Z(rho) monotonically non-increasing in rho: {monotonic_decreasing}"
      f" (if True: a naive evidence-based choice of rho would always collapse"
      f" to rho=0 -- i.e. to the very independence assumption under dispute --"
      f" which is why rho must be fixed by prior/design-effect reasoning, not fit)")

# ----------------------------------------------------------------------------
# New diagnostic: per-model credibility P_i across scenarios
# ----------------------------------------------------------------------------
# Reimplements, in spirit, the manuscript's Table 2 "P_i" diagnostic
# (2D_MassRadius_Combination.ipynb, "Posterior values of P_i") in 1D
# compactness space: for each model, the combined-posterior-weighted average
# probability that ITS OWN good KDE (rather than its own bad component)
# explains the data. Two deliberate differences from the published notebook:
#   (1) computed here in 1D compactness, not 2D (M,R) -- so it will NOT
#       exactly reproduce the manuscript's printed P_i numbers (0.20-0.81),
#       only their rough ordering, if that.
#   (2) omits that notebook's unexplained "1000 *" scaling factor in its
#       prob_grid computation -- this version is a properly normalized [0,1]
#       probability by construction. Worth checking with Chun Huang whether
#       that factor was intentional (e.g. for a plotting scale) or a bug.
# The per-model good/bad split itself is NOT re-tempered by kappa/rho -- it
# always uses the untempered (kappa=1) sigma-marginalized bad component,
# i.e. RHO_SWEEP_POSTERIOR_SIGMA[0] (rho=0). Only the combined posterior used
# to WEIGHT each model's credibility changes across scenarios. This isolates
# "how does the tempering change which region of C we're asking about" from
# "how good is this particular model's own good/bad split" -- flagged as a
# modeling choice worth confirming, not an obviously-only-correct option.

def per_model_bad_marginal(posterior_sigma_weights):
    b_marginal = {}
    for key in MODEL_KEYS:
        b_m = np.sum(b_dict[key] * posterior_sigma_weights[:, None], axis=0)
        b_m /= np.trapz(b_m, C_grid)
        b_marginal[key] = b_m
    return b_marginal


def per_model_credibility(combined_posterior, b_marginal):
    credibility = {}
    for key in MODEL_KEYS:
        g = g_dict[key]
        b = b_marginal[key]
        p_term = np.trapz(
            (p_grid[:, None] * g[None, :])
            / (p_grid[:, None] * g[None, :] + (1 - p_grid[:, None]) * b[None, :] + 1e-300),
            p_grid, axis=0,
        )
        credibility[key] = np.trapz(p_term * combined_posterior, C_grid)
    return credibility


b_marginal_untempered = per_model_bad_marginal(RHO_SWEEP_POSTERIOR_SIGMA[0])
credibility_by_scenario = {
    label: per_model_credibility(post, b_marginal_untempered) for label, post in scenarios.items()
}

MANUSCRIPT_P_I_REFERENCE = {"PDT_U26": 0.81, "ST_PST": 0.20}  # manuscript Table 2 (2D M-R basis), for context only

print("\nPer-model credibility P_i (P[model's own good KDE explains the data], combined-posterior-weighted):")
header = f"{'model':<24}" + "".join(f"{label[:18]:>20}" for label in scenarios)
print(header)
for key in MODEL_KEYS:
    row = f"{MODELS[key]['label']:<24}" + "".join(f"{credibility_by_scenario[label][key]:>20.3f}" for label in scenarios)
    print(row)
print("(manuscript Table 2, 2D M-R basis, for context only -- not expected to match exactly:"
      f" Kini26 PDT-U = {MANUSCRIPT_P_I_REFERENCE['PDT_U26']}, Vinciguerra24 ST+PST = {MANUSCRIPT_P_I_REFERENCE['ST_PST']})")

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

plt.rcParams.update({"font.size": 13})

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["black", "tab:gray", "tab:blue", "tab:orange", "tab:red"]
for (label, post), color in zip(scenarios.items(), colors):
    ax.plot(C_grid, post, color=color, lw=2, label=label)
ax.set_xlabel("Compactness $GM/(Rc^2)$")
ax.set_ylabel("Density")
ax.set_title("Combined compactness posterior:\ncorrelation-tempered pooling diagnostic scenarios")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "compactness_posteriors_diagnostic_scenarios.png"), dpi=200)

fig2, ax2 = plt.subplots(figsize=(8, 5.5))
ax2.plot(C_grid, rho_beta41_nosys, color="tab:orange", lw=2, label=r"rho ~ Beta(4,1), no $\sigma_{sys}$")
ax2.plot(C_grid, rho_beta41_sys, color="tab:red", lw=2, label=r"rho ~ Beta(4,1), $\sigma_{sys}$ marginalized")
ax2.set_xlabel("Compactness $GM/(Rc^2)$")
ax2.set_ylabel("Density")
ax2.set_title("Effect of the group-level systematic convolution")
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "sigma_sys_effect.png"), dpi=200)

fig3, ax3 = plt.subplots(figsize=(8, 5.5))
ax3.plot(RHO_GRID, log_evidence_ratio, "o-", color="black")
ax3.axhline(0.0, color="gray", linestyle=":", linewidth=1)
ax3.set_xlabel(r"$\rho$")
ax3.set_ylabel(r"$\log[Z(\rho)/Z(0)]$")
ax3.set_title("Data-implied rho: marginal evidence vs. rho\n(maximized at rho=0 -- NOT used to set rho; diagnostic only)")
fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, "evidence_vs_rho.png"), dpi=200)

fig4, ax4 = plt.subplots(figsize=(10, 6))
x = np.arange(N_MODELS)
width = 0.15
for i, (label, color) in enumerate(zip(scenarios, colors)):
    vals = [credibility_by_scenario[label][key] for key in MODEL_KEYS]
    ax4.bar(x + (i - 2) * width, vals, width, label=label, color=color)
ax4.set_xticks(x)
ax4.set_xticklabels([MODELS[key]["label"] for key in MODEL_KEYS], rotation=45, ha="right")
ax4.set_ylabel("Credibility $P_i$ (combined-posterior-weighted)")
ax4.set_title("Per-model credibility across correlation-tempering scenarios")
ax4.legend(fontsize=8)
fig4.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, "per_model_credibility_by_scenario.png"), dpi=200)

print(f"\nFigures written to {FIG_DIR}")

print(
    "\n"
    "QUESTIONS FOR THE AUTHORS:\n"
    "  1. Is rho ~ Beta(4,1) the right 'main conservative' prior, or should the\n"
    "     main result use Uniform(0,1) instead (less committal, but the PDF\n"
    "     itself labels Beta(4,1) as the recommended main prior)?\n"
    "  2. Is the data-driven s_C = std(median(C_i)) a reasonable HalfNormal scale\n"
    "     for sigma_sys, or should it be set/bounded by something else (e.g. a\n"
    "     fraction of the typical single-measurement width S_i)?\n"
    "  3. Model-weight sensitivity (equal vs. within-analysis evidence weights,\n"
    "     PDF Sec. 7 item 5) is not implemented -- it needs Bayesian evidence\n"
    "     values from each of the 8 original analyses. Worth pulling those from\n"
    "     the source papers for the response letter?\n"
)
