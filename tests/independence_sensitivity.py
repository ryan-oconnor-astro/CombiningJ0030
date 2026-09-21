"""
Sensitivity of the J0030 combined posterior to the independence assumption.

The referee's core objection (feedback.md) is that the 8 input M-R posteriors
combined in the manuscript are not statistically independent: most of them
share the same underlying NICER/XMM photons, or are different model fits to
the exact same data. This script asks a narrower, Option-B-style question:
*given the published combination method, unchanged, how much do the position
and width of the final posterior move as we relax the independence
assumption?* It does not attempt to fix the method (that is Option A/C).

Method
------
We reuse the exact good/bad Press-mixture pipeline from
1D_Compactness_Combination.ipynb / 2D_MassRadius_Combination.ipynb unchanged,
except that each measurement's per-datum mixture term
    p * P_good,i(theta) + (1-p) * P_bad,i(theta)
is raised to a power w_i <= 1 before the product over the 8 measurements is
taken. This is the standard "power likelihood" / design-effect correction used
in meta-analysis to avoid pseudo-replication from correlated or overlapping
data (e.g. Kish's design effect for clustered samples). At w_i = 1 for all i
this reduces EXACTLY to the published method. For a group of n measurements
that are assumed to share a fraction rho of their statistical information
(compound-symmetric correlation rho), the design-effect-consistent power is

    w = 1 / (1 + (n-1) * rho)

applied to every member of that group (rho=0 -> w=1, no correction; rho=1 ->
w=1/n, the whole group counts as one independent measurement).

Scenarios explored
-------------------
1. "published"        - w_i = 1 for all 8 (sanity check against the paper).
2. "global sweep"     - all 8 treated as ONE group, rho swept 0->1. The most
                         conservative/simplest envelope.
3. "grouped sweep"    - the four paper-level groups used in
                         tests/1D_correlation_tempered.ipynb
                         (Riley / Miller / Vinciguerra / Kini), each with its
                         own internal rho, swept 0->1 together. Groups stay
                         independent OF EACH OTHER (rho_between=0). At rho=1
                         this is exactly the "collapse each paper to one
                         effective measurement" idea already sketched in
                         2D_MassRadius_WithClasses.ipynb / tests/*_correlation_tempered.ipynb.
4. "referee_informed" - a single discrete scenario using specific numbers
                         *read off the referee's own claims* (feedback.md) and
                         the manuscript's own text, rather than a free
                         sweep. See ASSUMPTIONS below -- these numbers are my
                         best reading of the available text, not measured
                         photon counts, and should be checked/replaced.

ASSUMPTIONS THAT NEED AUTHOR INPUT (flagged again at the bottom of this file
and repeated when the script runs):
  - Riley19 ST+PST and Miller19 2spot/3spot are treated as rho=1 with each
    other ("exact same dataset" -- feedback.md point 1).
  - The 4 Vinciguerra24 models are treated as rho=1 with each other (all 4 are
    fit to the exact same joint NICER+XMM dataset in that paper).
  - Vinciguerra24 vs. Riley19/Miller19 is set to rho=0.98, taken directly from
    the referee's "98% of total photons" claim (feedback.md point 2). This
    number describes NICER-photon overlap only; Vinciguerra24 additionally
    has XMM data that Riley19/Miller19 lack, which this single scalar does not
    capture.
  - Kini26 vs. everything else is set to rho=0.667 = 1/1.5, derived from the
    manuscript's own statement that Kini26 increases NICER counts "by about
    50%" relative to previous analyses while reusing the same XMM data as
    Vinciguerra24 (manuscript.tex Sec. 2.3). This is a back-of-envelope
    photon-count argument, not a quoted correlation.
  These are the numbers most worth double-checking against the original
  papers (or asking Chun Huang / the original authors) before using this
  analysis in a response letter.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, gaussian_kde

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, "datafiles")
FIG_DIR = os.path.join(HERE, "independence_sensitivity_figures")
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

# Paper-level grouping, matches tests/1D_correlation_tempered.ipynb
GROUPS = {
    "Riley":       ["ST_PST19"],
    "Miller":      ["2spot", "3spot"],
    "Vinciguerra": ["ST_PDT", "ST_PST", "PDT_U", "ST_U"],
    "Kini":        ["PDT_U26"],
}

PUBLISHED_C = dict(median=0.172, lo=0.007, hi=0.006)  # manuscript.tex Sec 3.1


# ----------------------------------------------------------------------------
# Load data, build good (KDE) and bad (Press 1996) densities for compactness
# ----------------------------------------------------------------------------

compactness = {}
C_med, S_i = {}, {}
KDEs_1D = {}

for key, meta in MODELS.items():
    data = np.loadtxt(os.path.join(DATA_DIR, meta["file"]))
    M, R = data.T[0], data.T[1]
    C = G * (M * M_sun) / (R * 1e3 * c ** 2)
    compactness[key] = C
    C_med[key] = np.median(C)
    S_i[key] = max(C.max() - C_med[key], C_med[key] - C.min())
    KDEs_1D[key] = gaussian_kde(C, bw_method="silverman")

C_min = min(compactness[k].min() for k in MODEL_KEYS)
C_max = max(compactness[k].max() for k in MODEL_KEYS)
C_grid = np.linspace(C_min, C_max, 200)
dC = C_grid[1] - C_grid[0]

g_dict = {}
for key in MODEL_KEYS:
    g = KDEs_1D[key].evaluate(C_grid)
    g_dict[key] = g / np.trapz(g, C_grid)
g_stack = np.array([g_dict[key] for key in MODEL_KEYS])  # (8, NC)

# Hyperparameter grids (identical to 1D_Compactness_Combination.ipynb)
sigma_grid = np.logspace(-4, -0.5, 15)
alpha_grid = np.linspace(1.0, 5.0, 30)
p_grid = np.linspace(0.0, 1.0, 60)

prior_sigma = (1.0 / sigma_grid)
prior_sigma /= np.trapz(prior_sigma, sigma_grid)
prior_alpha = np.exp(-alpha_grid)
prior_alpha /= np.trapz(prior_alpha, alpha_grid)

# Bad distributions b_dict[key]: shape (Nsigma, NC). Independent of any
# weighting, so this is built once and reused for every scenario below.
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


def combine_1d(weights):
    """Weighted good/bad combination. weights: dict[model_key] -> power w_i <= 1.

    w_i = 1 for all keys reproduces the published method exactly.
    """
    w = np.array([weights[key] for key in MODEL_KEYS])[:, None, None]

    P_C_given_sigma, evidence_sigma = [], []
    for s_index in range(len(sigma_grid)):
        b_stack = np.array([b_dict[key][s_index] for key in MODEL_KEYS])
        g_exp = g_stack[:, None, :]
        b_exp = b_stack[:, None, :]
        p_exp = p_grid[None, :, None]

        mixture = p_exp * g_exp + (1 - p_exp) * b_exp
        mixture_w = np.clip(mixture, 1e-300, None) ** w

        L_p = np.prod(mixture_w, axis=0)
        likelihood = np.trapz(L_p, p_grid, axis=0)
        Z = np.trapz(likelihood, C_grid)

        P_C_given_sigma.append(likelihood / Z)
        evidence_sigma.append(Z)

    P_C_given_sigma = np.array(P_C_given_sigma)
    evidence_sigma = np.array(evidence_sigma)

    posterior_sigma = evidence_sigma * prior_sigma
    posterior_sigma /= np.trapz(posterior_sigma, sigma_grid)

    final_posterior = np.sum(P_C_given_sigma * posterior_sigma[:, None], axis=0)
    final_posterior /= np.trapz(final_posterior, C_grid)

    cdf = np.cumsum(final_posterior) * dC
    q16, q50, q84 = np.interp([0.16, 0.50, 0.84], cdf, C_grid)
    return dict(grid=C_grid, posterior=final_posterior, median=q50, lo=q50 - q16, hi=q84 - q50)


# ----------------------------------------------------------------------------
# Weighting schemes
# ----------------------------------------------------------------------------

def design_effect_power(n, rho):
    """Power to apply to each of n compound-symmetric-correlated (rho) members
    so that the group's total information content equals n_eff = n/(1+(n-1)rho)."""
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
    """Single discrete scenario built from an explicit 8x8 correlation matrix
    using the numbers listed in the module docstring, then reduced to one
    scalar effective-N via n_eff = n^2 / sum(R), applied as a uniform global
    power. This deliberately does NOT reweight individual models relative to
    each other (that would start to be a methodology fix, i.e. Option A/C);
    it only asks how much the *published, equally-weighted* method's width
    would need to inflate given this correlation structure."""
    n = len(MODEL_KEYS)
    idx = {key: i for i, key in enumerate(MODEL_KEYS)}
    R = np.eye(n)

    riley_miller = ["ST_PST19", "2spot", "3spot"]
    vinciguerra = ["ST_PDT", "ST_PST", "PDT_U", "ST_U"]
    old_nicer_cluster = riley_miller + vinciguerra
    kini = ["PDT_U26"]

    def set_rho(a, b, rho):
        R[idx[a], idx[b]] = rho
        R[idx[b], idx[a]] = rho

    for a in riley_miller:
        for b in riley_miller:
            if a != b:
                set_rho(a, b, 1.0)          # "exact same dataset" (feedback.md pt.1)
    for a in vinciguerra:
        for b in vinciguerra:
            if a != b:
                set_rho(a, b, 1.0)          # same joint fit, 4 models, same data
    for a in riley_miller:
        for b in vinciguerra:
            set_rho(a, b, 0.98)             # "98% of total photons" (feedback.md pt.2)
    for a in old_nicer_cluster:
        for b in kini:
            set_rho(a, b, 1 / 1.5)          # Kini adds ~50% more counts (manuscript.tex Sec 2.3)

    n_eff = n ** 2 / R.sum()
    w = n_eff / n
    return {key: w for key in MODEL_KEYS}, n_eff, R


# ----------------------------------------------------------------------------
# Run scenarios
# ----------------------------------------------------------------------------

published = combine_1d({key: 1.0 for key in MODEL_KEYS})
print("=" * 70)
print("SANITY CHECK against manuscript.tex Sec 3.1:")
print(f"  script:     C = {published['median']:.4f} +{published['hi']:.4f} / -{published['lo']:.4f}")
print(f"  manuscript: C = {PUBLISHED_C['median']:.4f} +{PUBLISHED_C['hi']:.4f} / -{PUBLISHED_C['lo']:.4f}")
print("=" * 70)

rho_values = np.linspace(0.0, 1.0, 11)

global_results = [combine_1d(global_weights(r)) for r in rho_values]
grouped_results = [combine_1d(grouped_weights(r)) for r in rho_values]

ref_weights, ref_n_eff, ref_R = referee_informed_weights()
referee_result = combine_1d(ref_weights)

grouped_rho1 = combine_1d(grouped_weights(1.0))  # "4 effectively-independent groups"
global_rho1 = combine_1d(global_weights(1.0))    # "everything is 1 measurement"

print("\nEffective-N summary (nominal N=8):")
print(f"  grouped_by_paper, rho_within=1  -> N_eff = {sum(len(m) / (1 + (len(m)-1)*1.0) for m in GROUPS.values() for _ in [0]):.2f}"
      f"  (i.e. exactly {len(GROUPS)} groups)")
print(f"  referee_informed                -> N_eff = {ref_n_eff:.2f}"
      f"   (implied width inflation ~ sqrt(8/N_eff) = {np.sqrt(8/ref_n_eff):.2f}x)")

print("\nScenario                          C (median)   width (68%)   width / published")
def _row(label, res):
    width = res["lo"] + res["hi"]
    pub_width = published["lo"] + published["hi"]
    print(f"{label:<34} {res['median']:.4f}       {width:.4f}        {width/pub_width:.2f}x")

_row("published (rho=0)", published)
_row("grouped_by_paper, rho=1", grouped_rho1)
_row("referee_informed", referee_result)
_row("global, rho=1", global_rho1)

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

plt.rcParams.update({"font.size": 13})

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

ax = axes[0]
pub_width = published["lo"] + published["hi"]
widths_global = [(r["lo"] + r["hi"]) / pub_width for r in global_results]
widths_grouped = [(r["lo"] + r["hi"]) / pub_width for r in grouped_results]
ax.plot(rho_values, widths_global, "o-", color="black", label="global (all 8 as one group)")
ax.plot(rho_values, widths_grouped, "s-", color="tab:blue", label="grouped by paper (Riley/Miller/Vinciguerra/Kini)")
ax.scatter([1.0], [(referee_result["lo"] + referee_result["hi"]) / pub_width],
           marker="*", s=250, color="tab:red", zorder=5, label="referee-informed (discrete)")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel(r"assumed within-group correlation $\rho$")
ax.set_ylabel("68% width / published width")
ax.set_title("Posterior width vs.\nindependence assumption", fontsize=13)
ax.legend(fontsize=9)

ax2 = axes[1]
medians_global = [r["median"] for r in global_results]
medians_grouped = [r["median"] for r in grouped_results]
ax2.plot(rho_values, medians_global, "o-", color="black", label="global")
ax2.plot(rho_values, medians_grouped, "s-", color="tab:blue", label="grouped by paper")
ax2.scatter([1.0], [referee_result["median"]], marker="*", s=250, color="tab:red", zorder=5)
ax2.axhline(published["median"], color="gray", linestyle=":", linewidth=1, label="published median")
ax2.set_xlabel(r"assumed within-group correlation $\rho$")
ax2.set_ylabel("posterior median compactness $C$")
ax2.set_title("Posterior position vs.\nindependence assumption", fontsize=13)
ax2.legend(fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "compactness_width_vs_rho.png"), dpi=200)

fig2, ax3 = plt.subplots(figsize=(8, 5.5))
for label, res, color in [
    ("published (rho=0)", published, "black"),
    ("grouped_by_paper, rho=1\n(4 independent groups)", grouped_rho1, "tab:blue"),
    ("referee_informed", referee_result, "tab:red"),
    ("global, rho=1\n(1 independent measurement)", global_rho1, "tab:gray"),
]:
    ax3.plot(res["grid"], res["posterior"], color=color, lw=2, label=label)
ax3.set_xlabel("Compactness $GM/(Rc^2)$")
ax3.set_ylabel("Density")
ax3.set_title("Combined compactness posterior under different independence assumptions")
ax3.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "compactness_posteriors_by_scenario.png"), dpi=200)

print(f"\nFigures written to {FIG_DIR}")

print(
    "\n"
    "QUESTIONS FOR THE AUTHORS (see module docstring for full detail):\n"
    "  1. Is the referee_informed correlation matrix reasonable, or do you have\n"
    "     actual exposure-time / photon-count numbers from Riley19, Miller19,\n"
    "     Vinciguerra24, Kini26 to replace the 0.98 and 0.667 placeholders?\n"
    "  2. Should Riley19/Miller19 vs. Vinciguerra24 really be treated as a single\n"
    "     rho=0.98, or is the actual overlap different for ST+PST specifically\n"
    "     (the model Vinciguerra24 shares literally with Riley19) vs. the other\n"
    "     three Vinciguerra24 models?\n"
    "  3. Is 1/1.5 a fair proxy for Kini26's correlation with the rest, or is\n"
    "     that oversimplified given it also inherits Vinciguerra24's XMM data\n"
    "     exactly (not just an incremental NICER extension)?\n"
)
