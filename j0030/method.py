"""
Combination of the eight PSR J0030+0451 mass-radius posteriors.

Each published posterior enters through a good/bad mixture (Press 1996):

    L_i(M, C | p, sigma) = p * g_i(M, C) + (1 - p) * b_i(M, C | sigma),

where the good component g_i is a kernel density estimate (KDE) of the
published samples and the bad component b_i is a broad Gaussian centred on the
input's median. Its baseline width S_i is set so that the Gaussian's central
68% interval matches the input's central 99% interval. That width is inflated
by alpha >= 1 (prior exp(-alpha)) and widened by a shift scale sigma
(prior 1/sigma). The mixing probability p has a flat prior and is shared by
all inputs.

Two ingredients account for the inputs sharing data and for systematics common
to all of them:

  1. Correlation tempering. Each mixture is raised to
     kappa(rho) = 1 / (1 + (N - 1) rho) before the product over the N = 8
     inputs, and rho is marginalized over a prior (Uniform(0, 1) for the
     adopted result).
  2. A common systematic displacement Sigma_sys = diag(sigma_M^2, sigma_C^2),
     applied as a Gaussian convolution of the pooled posterior and
     marginalized over half-normal priors whose scales are the spread of the
     eight per-input medians.

The combination is carried out in mass and compactness C = G M / (R c^2), the
quantity pulse-profile modelling constrains most directly. The bad component
and Sigma_sys are diagonal in (M, C), so in (M, R) they follow lines of
constant compactness. Everything is evaluated on a fixed grid and integrated
with the trapezoid rule; M runs from 0.95 to 2.2 Msun and C over the pooled
range of the input samples (80 x 80 points). The tempered posteriors do not
fall to zero inside this domain, so quoted intervals are conditional on it.
"""

import os
import shutil

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.stats import beta as beta_dist
from scipy.stats import gaussian_kde, halfnorm, norm

# np.trapezoid only exists from numpy 2.0, and np.trapz may be removed after it.
trapz = getattr(np, "trapezoid", None) or np.trapz

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(REPO_ROOT, "figures")
RESULTS_FILE = os.path.join(RESULTS_DIR, "combined_MC.npz")
SAMPLES_FILE = os.path.join(RESULTS_DIR, "combined_posterior.txt")

G = 6.67430e-11        # m^3 kg^-1 s^-2
c = 2.99792458e8       # m s^-1
M_sun = 1.98847e30     # kg
K_KM = G * M_sun / (c ** 2) / 1e3   # C = K_KM * M[Msun] / R[km]  (~1.4766)

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

RHO_GRID = np.linspace(0.0, 1.0, 21)

# Scenario key -> description. "uniform_sys" is the adopted result.
SCENARIOS = {
    "rho0": "independent (rho = 0, no Sigma_sys)",
    "rho1": "fully dependent (rho = 1, no Sigma_sys)",
    "beta41_nosys": "rho ~ Beta(4, 1), no Sigma_sys",
    "uniform_nosys": "rho ~ Uniform(0, 1), no Sigma_sys",
    "uniform_sys": "rho ~ Uniform(0, 1), Sigma_sys marginalized (adopted)",
}
DEFAULT_SCENARIO = "uniform_sys"


def rho_prior_uniform(rho):
    return np.ones_like(rho)


def rho_prior_beta41(rho):
    return beta_dist.pdf(rho, 4, 1)


def design_effect_power(n, rho):
    """kappa(rho) = 1 / (1 + (n - 1) rho)."""
    return 1.0 / (1.0 + (n - 1) * rho)


def load_samples(key):
    """(M [Msun], R [km]) posterior samples of one input (or other file in data/)."""
    fname = MODELS[key]["file"] if key in MODELS else key
    data = np.loadtxt(os.path.join(DATA_DIR, fname))
    return data.T[0], data.T[1]


def label(key):
    """Plain-text label of an input (LaTeX dashes removed)."""
    return MODELS[key]["label"].replace("--", "-")


def quantiles(grid, pdf, probs=(0.16, 0.50, 0.84)):
    """Median and 68% half-widths of a gridded 1D density, from its trapezoid CDF."""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    lo, med, hi = np.interp(probs, cdf / cdf[-1], grid)
    return dict(median=med, lo=med - lo, hi=hi - med)


def sample_quantiles(x):
    """Median and 68% half-widths of a set of samples."""
    lo, med, hi = np.quantile(x, [0.16, 0.50, 0.84])
    return dict(median=med, lo=med - lo, hi=hi - med)


def fmt(q, nd):
    """'median +hi/-lo' with nd decimals."""
    return f"{q['median']:.{nd}f} +{q['hi']:.{nd}f}/-{q['lo']:.{nd}f}"


def width(q):
    return q["lo"] + q["hi"]


def hdr_levels(pdf, probs=(0.68, 0.95)):
    """Density thresholds enclosing each probability, returned increasing
    ([lvl95, lvl68]) as matplotlib's contour wants."""
    flat = np.sort(pdf.ravel())[::-1]
    cumsum = np.cumsum(flat)
    cumsum /= cumsum[-1]
    return sorted(flat[np.searchsorted(cumsum, p)] for p in probs)


def use_paper_style():
    """Matplotlib settings of the paper figures (LaTeX text if available)."""
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-white")
    plt.rc("text", usetex=shutil.which("latex") is not None)
    plt.rc("font", family="serif")


# ----------------------------------------------------------------------------
# Grids, good (KDE) and bad (broad Gaussian) components
# ----------------------------------------------------------------------------


class Setup:
    """Grids, priors and the good and bad components of every input, in (M, C).

    Building it evaluates eight 2D KDEs, some on 10^6 samples, and takes about
    a minute.
    """

    def __init__(self):
        samples, M_med, S_M, C_med, S_C, C_lo, C_hi = {}, {}, {}, {}, {}, {}, {}
        for key in MODEL_KEYS:
            M, R = load_samples(key)
            C = K_KM * M / R
            samples[key] = np.vstack([M, C])
            M_med[key], C_med[key] = np.median(M), np.median(C)
            # Baseline bad-component widths: the bad Gaussian's central 68%
            # width equals the input's central 99% width.
            S_M[key] = 0.5 * (np.quantile(M, 0.995) - np.quantile(M, 0.005)) / norm.ppf(0.84)
            S_C[key] = 0.5 * (np.quantile(C, 0.995) - np.quantile(C, 0.005)) / norm.ppf(0.84)
            C_lo[key], C_hi[key] = C.min(), C.max()
        self.samples = samples
        self.M_med, self.C_med, self.S_M, self.S_C = M_med, C_med, S_M, S_C

        # Common systematic scales: spread of the eight per-input medians.
        self.sys_scale_M = np.std(list(M_med.values()))
        self.sys_scale_C = np.std(list(C_med.values()))

        self.M_grid = np.linspace(0.95, 2.2, 80)
        self.C_grid = np.linspace(min(C_lo.values()), max(C_hi.values()), 80)
        # Shift scales sigma with prior 1/sigma.
        self.sigma_M_grid = np.logspace(-4, 0, 6)       # Msun
        self.sigma_C_grid = np.logspace(-4, -0.5, 6)
        self.alpha_grid = np.linspace(1.0, 10, 40)
        self.p_grid = np.linspace(0.0, 1.0, 25)

        self.prior_sigma_M = self._normed(1.0 / self.sigma_M_grid, self.sigma_M_grid)
        self.prior_sigma_C = self._normed(1.0 / self.sigma_C_grid, self.sigma_C_grid)
        self.prior_alpha = self._normed(np.exp(-self.alpha_grid), self.alpha_grid)

        self.dM = self.M_grid[1] - self.M_grid[0]
        self.dC = self.C_grid[1] - self.C_grid[0]
        MM, CC = np.meshgrid(self.M_grid, self.C_grid, indexing="ij")

        # Good components: 2D Silverman KDEs of the samples, shape (8, NM, NC).
        pos = np.vstack([MM.ravel(), CC.ravel()])
        g_stack = np.zeros((N_MODELS, MM.size))
        for i, key in enumerate(MODEL_KEYS):
            g_stack[i] = gaussian_kde(samples[key], bw_method="silverman")(pos)
        self.g_stack = self.norm2d(g_stack.reshape((N_MODELS,) + MM.shape))

        # Bad components with alpha marginalized: (8, N_sigma_M, N_sigma_C, NM, NC).
        # They do not depend on rho or Sigma_sys, so they are built once.
        a = self.alpha_grid[:, None, None]
        b_all = np.zeros((N_MODELS, len(self.sigma_M_grid), len(self.sigma_C_grid)) + MM.shape)
        for i, key in enumerate(MODEL_KEYS):
            for iM, sigma_M in enumerate(self.sigma_M_grid):
                for iC, sigma_C in enumerate(self.sigma_C_grid):
                    b_alpha = (
                        norm.pdf(MM[None], loc=M_med[key], scale=np.sqrt((a * S_M[key]) ** 2 + sigma_M ** 2))
                        * norm.pdf(CC[None], loc=C_med[key], scale=np.sqrt((a * S_C[key]) ** 2 + sigma_C ** 2))
                    )
                    b_alpha = self.norm2d(b_alpha)
                    b_all[i, iM, iC] = self.norm2d(
                        trapz(b_alpha * self.prior_alpha[:, None, None], self.alpha_grid, axis=0))
        self.b_all = b_all

    @staticmethod
    def _normed(f, x):
        return f / trapz(f, x)

    def norm2d(self, density):
        """Normalize density (..., NM, NC) to unit integral."""
        const = trapz(trapz(density, self.C_grid, axis=-1), self.M_grid, axis=-1)
        return density / np.expand_dims(const, axis=(-2, -1))

    # ------------------------------------------------------------------------
    # Tempered combination at fixed rho
    # ------------------------------------------------------------------------

    def combine_given_kappa(self, kappa):
        """Good/bad combination with each input's mixture raised to kappa.
        kappa = 1 treats the inputs as independent.

        Returns
          posterior   p(M, C | kappa) with p and sigma marginalized, unit integral;
          log_Z_norm  log evidence with every tempered mixture renormalized to
                      unit integral. pi(rho) Z_norm(rho) is the density-agreement
                      diagnostic on rho: it measures how much the inputs
                      disagree, not how correlated they are, and weights nothing
                      else;
          P_i         consistency probability of each input: its untempered
                      responsibility p g_i / (p g_i + (1-p) b_i) averaged over
                      the joint posterior of (M, C, p, sigma).
        """
        M_grid, C_grid, p_grid = self.M_grid, self.C_grid, self.p_grid
        NsM, NsC = len(self.sigma_M_grid), len(self.sigma_C_grid)
        evidence = np.zeros((NsM, NsC))
        log_Zn = np.zeros((NsM, NsC))
        good_num = np.zeros((NsM, NsC, N_MODELS))
        P_given_sigma = np.zeros((NsM, NsC, len(M_grid), len(C_grid)))
        p_exp = p_grid[None, :, None, None]
        g_exp = self.g_stack[:, None, :, :]

        for iM in range(NsM):
            for iC in range(NsC):
                b_exp = self.b_all[:, iM, iC][:, None, :, :]
                mixture = p_exp * g_exp + (1 - p_exp) * b_exp                  # (8, Np, NM, NC)
                mixture_c = np.clip(mixture, 1e-300, None)
                mixture_w = mixture_c ** kappa
                L_p = np.prod(mixture_w, axis=0)                               # (Np, NM, NC)
                likelihood = trapz(L_p, p_grid, axis=0)
                Z = trapz(trapz(likelihood, C_grid, axis=1), M_grid)
                evidence[iM, iC] = Z
                P_given_sigma[iM, iC] = likelihood / Z

                resp = p_exp * g_exp / mixture_c
                good_num[iM, iC] = trapz(
                    trapz(trapz(L_p[None] * resp, C_grid, axis=3), M_grid, axis=2), p_grid, axis=1)

                log_c = np.log(trapz(trapz(mixture_w, C_grid, axis=3), M_grid, axis=2))  # (8, Np)
                log_f = kappa * np.log(mixture_c).sum(axis=0) - log_c.sum(axis=0)[:, None, None]
                shift = log_f.max()
                log_Zn[iM, iC] = shift + np.log(trapz(trapz(trapz(
                    np.exp(log_f - shift), C_grid, axis=2), M_grid, axis=1), p_grid))

        sM, sC = self.sigma_M_grid, self.sigma_C_grid
        prior_sigma = self.prior_sigma_M[:, None] * self.prior_sigma_C[None, :]
        posterior_sigma = evidence * prior_sigma
        posterior_sigma /= trapz(trapz(posterior_sigma, sC, axis=1), sM)

        posterior = self.norm2d(trapz(
            trapz(P_given_sigma * posterior_sigma[:, :, None, None], sC, axis=1), sM, axis=0))

        log_w = log_Zn + np.log(prior_sigma)
        shift = log_w.max()
        log_Z_norm = shift + np.log(trapz(trapz(np.exp(log_w - shift), sC, axis=1), sM))

        P_i = (trapz(trapz(good_num * prior_sigma[:, :, None], sC, axis=1), sM, axis=0)
               / trapz(trapz(evidence * prior_sigma, sC, axis=1), sM))
        return posterior, log_Z_norm, P_i

    # ------------------------------------------------------------------------
    # Sigma_sys: diagonal, so the 2D convolution factors into two 1D ones and
    # the two marginalizations can be done one axis after the other.
    # ------------------------------------------------------------------------

    def _marginalize_sys_axis(self, density, scale, axis, spacing):
        sigma_grid = np.linspace(1e-6, 4.0 * scale, 15)
        prior = self._normed(halfnorm.pdf(sigma_grid, scale=scale), sigma_grid)
        stack = np.array([
            self.norm2d(gaussian_filter1d(density, sigma=s / spacing, axis=axis, mode="constant", cval=0.0))
            for s in sigma_grid
        ])
        return self.norm2d(trapz(stack * prior[:, None, None], sigma_grid, axis=0))

    def marginalize_sigma_sys(self, density):
        """Convolve a pooled (M, C) density with Sigma_sys and marginalize its scales."""
        density = self._marginalize_sys_axis(density, self.sys_scale_C, axis=1, spacing=self.dC)
        return self._marginalize_sys_axis(density, self.sys_scale_M, axis=0, spacing=self.dM)

    # ------------------------------------------------------------------------
    # (M, C) -> (M, R)
    # ------------------------------------------------------------------------

    def R_grid(self):
        """R grid covering the whole (M, C) domain (~6.6-32 km)."""
        return np.linspace(K_KM * self.M_grid[0] / self.C_grid[-1],
                           K_KM * self.M_grid[-1] / self.C_grid[0], 300)

    def to_MR(self, post_MC, R_grid):
        """p(M, R) = p(M, C = K M / R) * K M / R^2, normalized on R_grid."""
        interp = RegularGridInterpolator((self.M_grid, self.C_grid), post_MC,
                                         bounds_error=False, fill_value=0.0)
        MM, RR = np.meshgrid(self.M_grid, R_grid, indexing="ij")
        p = interp(np.stack([MM, K_KM * MM / RR], axis=-1)) * K_KM * MM / RR ** 2
        return p / trapz(trapz(p, R_grid, axis=1), self.M_grid)


# ----------------------------------------------------------------------------
# Full run: rho sweep, scenarios, consistency probabilities
# ----------------------------------------------------------------------------


def rho_mixture(sweep, prior_fn):
    """integral of pi(rho) X(rho) d rho over RHO_GRID, for X of any shape."""
    w = prior_fn(RHO_GRID)
    w = w / trapz(w, RHO_GRID)
    return trapz(sweep * w.reshape((-1,) + (1,) * (sweep.ndim - 1)), RHO_GRID, axis=0)


def run(S=None, log=print):
    """Run the whole combination and return a dict of arrays (what
    save_results writes). Pass an existing Setup to skip rebuilding it."""
    import time

    if S is None:
        S = Setup()
    log(f"Sweeping rho over {len(RHO_GRID)} points...")
    t0 = time.time()
    sweep = [S.combine_given_kappa(design_effect_power(N_MODELS, rho)) for rho in RHO_GRID]
    post_sweep, log_Zn, P_i_sweep = (np.array(x) for x in zip(*sweep))
    log(f"  done in {time.time() - t0:.0f} s")

    uniform_nosys = S.norm2d(rho_mixture(post_sweep, rho_prior_uniform))
    posts = dict(
        rho0=post_sweep[0],
        rho1=post_sweep[-1],
        beta41_nosys=S.norm2d(rho_mixture(post_sweep, rho_prior_beta41)),
        uniform_nosys=uniform_nosys,
        uniform_sys=S.marginalize_sigma_sys(uniform_nosys),
    )

    # Density-agreement diagnostic p(rho | D) under the Uniform prior.
    rho_post = np.exp(np.log(rho_prior_uniform(RHO_GRID)) + log_Zn - log_Zn.max())
    rho_post /= trapz(rho_post, RHO_GRID)

    R_grid = S.R_grid()
    out = dict(
        M_grid=S.M_grid, C_grid=S.C_grid, R_grid=R_grid, RHO_GRID=RHO_GRID,
        model_keys=np.array(MODEL_KEYS), log_Z_norm=log_Zn, rho_posterior_uniform=rho_post,
        P_i_sweep=P_i_sweep, sys_scale_M=S.sys_scale_M, sys_scale_C=S.sys_scale_C,
    )
    # Sigma_sys convolves the pooled posterior after the responsibilities are
    # set, so it cannot change P_i: the uniform_sys P_i are the uniform ones.
    out["P_i_rho0"] = P_i_sweep[0]
    out["P_i_rho1"] = P_i_sweep[-1]
    out["P_i_beta41_nosys"] = rho_mixture(P_i_sweep, rho_prior_beta41)
    out["P_i_uniform_nosys"] = out["P_i_uniform_sys"] = rho_mixture(P_i_sweep, rho_prior_uniform)
    for k, p in posts.items():
        out[f"{k}_MC"] = p
        out[f"{k}_MR"] = S.to_MR(p, R_grid)
    return out


# ----------------------------------------------------------------------------
# Saving, loading and summarizing results
# ----------------------------------------------------------------------------


def write_samples(res, path=SAMPLES_FILE, n=10000, seed=0):
    """Draw n grid points from the adopted (M, R) posterior and save them as
    two columns, M [Msun] and R [km]."""
    post, _, _ = marginals_MR(res, DEFAULT_SCENARIO)
    MM, RR = np.meshgrid(res["M_grid"], res["R_grid"], indexing="ij")
    w = post.ravel() / post.sum()
    idx = np.random.default_rng(seed).choice(w.size, size=n, p=w)
    np.savetxt(path, np.column_stack([MM.ravel()[idx], RR.ravel()[idx]]), header="M R")


def save_results(res):
    """Write results/combined_MC.npz and results/combined_posterior.txt."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(RESULTS_FILE, **res)
    write_samples(res)
    for path in (RESULTS_FILE, SAMPLES_FILE):
        print(f"wrote {os.path.relpath(path, REPO_ROOT)}")


def load_results():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"{os.path.relpath(RESULTS_FILE, REPO_ROOT)} not found: "
                                "run 2_Combined_Posterior.ipynb first.")
    return dict(np.load(RESULTS_FILE))


def marginals_MR(res, scenario):
    """(post_MR, P_M, P_R) of one scenario, each normalized."""
    post = res[f"{scenario}_MR"]
    M_grid, R_grid = res["M_grid"], res["R_grid"]
    post = post / trapz(trapz(post, R_grid, axis=1), M_grid)
    P_M = trapz(post, R_grid, axis=1)
    P_R = trapz(post, M_grid, axis=0)
    return post, P_M / trapz(P_M, M_grid), P_R / trapz(P_R, R_grid)


def marginal_C(res, scenario):
    post = res[f"{scenario}_MC"]
    P_C = trapz(post, res["M_grid"], axis=0)
    return P_C / trapz(P_C, res["C_grid"])


def summarize(res, scenario=DEFAULT_SCENARIO):
    """Median and 68% half-widths of M, R and C for one scenario."""
    _, P_M, P_R = marginals_MR(res, scenario)
    return dict(M=quantiles(res["M_grid"], P_M), R=quantiles(res["R_grid"], P_R),
                C=quantiles(res["C_grid"], marginal_C(res, scenario)))


def print_scenarios(res):
    """Table of every scenario, with 68% widths relative to rho = 0."""
    rows = {k: summarize(res, k) for k in SCENARIOS}
    ref = rows["rho0"]
    nd = dict(M=3, R=2, C=4)
    print(f"{'scenario':<56}" + "".join(f"{c + ' (68%)':>24}" for c in "MRC")
          + "".join(f"{c + '/rho0':>8}" for c in "MRC"))
    for k, desc in SCENARIOS.items():
        r = rows[k]
        print(f"{desc:<56}" + "".join(f"{fmt(r[c], nd[c]):>24}" for c in "MRC")
              + "".join(f"{width(r[c]) / width(ref[c]):>8.2f}" for c in "MRC"))
    print("(x/rho0 = 68% width relative to the rho = 0 result.)")


def print_P_i(res):
    """Consistency probability P_i of each input under each rho prescription."""
    keys = list(res["model_keys"])
    order = sorted(keys, key=lambda k: -res[f"P_i_{DEFAULT_SCENARIO}"][keys.index(k)])
    cols = [("uniform_sys", "Uniform"), ("rho0", "rho=0"), ("beta41_nosys", "Beta(4,1)"), ("rho1", "rho=1")]
    print(f"{'input':<24}" + "".join(f"{name:>11}" for _, name in cols))
    for k in order:
        i = keys.index(k)
        print(f"{label(k):<24}" + "".join(f"{res[f'P_i_{s}'][i]:>11.3f}" for s, _ in cols))
