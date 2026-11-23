"""
Shared machinery for the revised combination of the eight PSR J0030+0451
mass-radius posteriors: the manuscript's good/bad (Press 1996) mixture with a
KDE good component, plus

  1. correlation tempering: every measurement's mixture likelihood is raised
     to kappa(rho) = 1 / (1 + (N - 1) rho) before the product over models,
     and rho is marginalized over a prior (default Uniform(0, 1));
  2. a group-level systematic Sigma_sys = diag(sigma_M_sys^2, sigma_y_sys^2),
     applied as a Gaussian convolution of the pooled posterior and
     marginalized over half-normal priors whose scales are the spread of the
     eight per-model medians;
  3. the combination carried out in (M, C), C = G M / (R c^2), instead of
     (M, R). The bad component and Sigma_sys are diagonal in (M, C), so once
     mapped back to (M, R) they follow lines of constant compactness, i.e.
     they are M-R-correlated.

The same engine also runs in (M, R) (space="MR"), on the manuscript's
9-17 km grid, as an optional comparison (no figure uses it); the revised
method is space="MC".

The grid is the manuscript notebook's: M 0.95-2.2 Msun (80 points) and, in
(M, C), C over the pooled range of the input samples (80 points, R ~6.6-32 km).
Changed from 2D_MassRadius_Combination.ipynb (2026-11-23): the baseline
bad-component widths S use the paper's 99%-quantile definition instead of the
furthest sample from the median, and the sigma_M shift-scale grid has fixed
bounds, 1e-4 to 1 Msun. Everything else not named above (shared mixing
probability p with a flat prior, 1/sigma priors on the bad-component shifts,
exp(-alpha) prior on its width, deterministic trapezoid quadrature) is
unchanged from the notebook. Before these two changes, space="MR" with rho = 0
and Sigma_sys = 0 recovered the manuscript's M = 1.46, R = 12.69 (to within
about 0.02); it no longer does exactly.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, "datafiles")
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")

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

# Scenario key -> description. "uniform_sys" is the revised method's result.
SCENARIOS = {
    "rho0": "independent (rho=0, no Sigma_sys) [manuscript assumption]",
    "rho1": "fully correlated (rho=1, no Sigma_sys)",
    "beta41_nosys": "rho ~ Beta(4,1), no Sigma_sys [sensitivity]",
    "uniform_nosys": "rho ~ Uniform(0,1), no Sigma_sys",
    "uniform_sys": "rho ~ Uniform(0,1), Sigma_sys marginalized [default]",
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
    """(M [Msun], R [km]) posterior samples of one input model."""
    data = np.loadtxt(os.path.join(DATA_DIR, MODELS[key]["file"]))
    return data.T[0], data.T[1]


def quantiles(grid, pdf, probs=(0.16, 0.50, 0.84)):
    """Median and 68% half-widths of a gridded 1D density, from a trapezoid
    CDF. (The manuscript notebooks used cumsum(pdf) * spacing, which reads
    about half a grid cell low; every number in this folder uses this one.)"""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    lo, med, hi = np.interp(probs, cdf / cdf[-1], grid)
    return dict(median=med, lo=med - lo, hi=hi - med)


def hdr_levels(pdf, probs=(0.68, 0.95)):
    """Density thresholds enclosing each probability, returned increasing
    ([lvl95, lvl68]) as matplotlib's contour wants."""
    flat = np.sort(pdf.ravel())[::-1]
    cumsum = np.cumsum(flat)
    cumsum /= cumsum[-1]
    return sorted(flat[np.searchsorted(cumsum, p)] for p in probs)


def use_paper_style():
    """Matplotlib settings of the manuscript figures (LaTeX text if available)."""
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-white")
    plt.rc("text", usetex=shutil.which("latex") is not None)
    plt.rc("font", family="serif")


# ----------------------------------------------------------------------------
# Grids, good (KDE) and bad (broad Gaussian) components
# ----------------------------------------------------------------------------


class Setup:
    """Grids, priors and per-model good/bad densities for one coordinate
    choice. x is always M; y is C (space="MC") or R (space="MR")."""

    def __init__(self, space="MC"):
        if space not in ("MC", "MR"):
            raise ValueError(f"space must be 'MC' or 'MR', not {space!r}")
        self.space = space

        samples, M_med, S_M, y_med, S_y, C_lo, C_hi = {}, {}, {}, {}, {}, {}, {}
        for key in MODEL_KEYS:
            M, R = load_samples(key)
            C = K_KM * M / R
            y = C if space == "MC" else R
            samples[key] = np.vstack([M, y])
            M_med[key], y_med[key] = np.median(M), np.median(y)
            # Baseline bad-component widths, Eq. (S_calibration): the bad
            # Gaussian's central 68% width equals the input's central 99%
            # width. (The manuscript notebook used the furthest sample from the
            # median, which grows with the number of samples.)
            S_M[key] = 0.5 * (np.quantile(M, 0.995) - np.quantile(M, 0.005)) / norm.ppf(0.84)
            S_y[key] = 0.5 * (np.quantile(y, 0.995) - np.quantile(y, 0.005)) / norm.ppf(0.84)
            C_lo[key], C_hi[key] = C.min(), C.max()
        self.M_med, self.y_med, self.S_M, self.S_y = M_med, y_med, S_M, S_y

        # Group-level systematic scales: spread of the eight per-model medians.
        self.sys_scale_M = np.std(list(M_med.values()))
        self.sys_scale_y = np.std(list(y_med.values()))

        self.M_grid = np.linspace(0.95, 2.2, 80)
        if space == "MC":
            # Full pooled range of the input samples, no padding.
            self.y_grid = np.linspace(min(C_lo.values()), max(C_hi.values()), 80)
            self.sigma_y_grid = np.logspace(-4, -0.5, 6)
        else:
            # The manuscript notebook's 9-17 km domain.
            self.y_grid = np.linspace(9.0, 17.0, 80)
            avg_S_R = np.mean(list(S_y.values()))
            self.sigma_y_grid = np.logspace(-10 * avg_S_R, -0.1 * avg_S_R, 6)
        # Fixed bounds for the 1/sigma shift-scale prior on M (Msun). The
        # manuscript notebook used logspace(-10*mean(S_M), -0.1*mean(S_M)),
        # which tied the prior to the baseline widths.
        self.sigma_M_grid = np.logspace(-4, 0, 6)
        self.alpha_grid = np.linspace(1.0, 10, 40)
        self.p_grid = np.linspace(0.0, 1.0, 25)

        self.prior_sigma_M = self._normed(1.0 / self.sigma_M_grid, self.sigma_M_grid)
        self.prior_sigma_y = self._normed(1.0 / self.sigma_y_grid, self.sigma_y_grid)
        self.prior_alpha = self._normed(np.exp(-self.alpha_grid), self.alpha_grid)

        self.dM = self.M_grid[1] - self.M_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]
        MM, YY = np.meshgrid(self.M_grid, self.y_grid, indexing="ij")
        pos = np.vstack([MM.ravel(), YY.ravel()])

        # Good components: 2D Silverman KDEs of the samples, shape (8, NM, Ny).
        g_stack = np.zeros((N_MODELS, MM.size))
        for i, key in enumerate(MODEL_KEYS):
            g_stack[i] = gaussian_kde(samples[key], bw_method="silverman")(pos)
        self.g_stack = self.norm2d(g_stack.reshape((N_MODELS,) + MM.shape))

        # Bad components, alpha already marginalized: (8, NsM, Nsy, NM, Ny).
        # Independent of rho and Sigma_sys, so built once.
        a = self.alpha_grid[:, None, None]
        b_all = np.zeros((N_MODELS, len(self.sigma_M_grid), len(self.sigma_y_grid)) + MM.shape)
        for i, key in enumerate(MODEL_KEYS):
            for iM, sigma_M in enumerate(self.sigma_M_grid):
                for iy, sigma_y in enumerate(self.sigma_y_grid):
                    b_alpha = (
                        norm.pdf(MM[None], loc=M_med[key], scale=np.sqrt((a * S_M[key]) ** 2 + sigma_M ** 2))
                        * norm.pdf(YY[None], loc=y_med[key], scale=np.sqrt((a * S_y[key]) ** 2 + sigma_y ** 2))
                    )
                    b_alpha = self.norm2d(b_alpha)
                    b_all[i, iM, iy] = self.norm2d(
                        trapz(b_alpha * self.prior_alpha[:, None, None], self.alpha_grid, axis=0))
        self.b_all = b_all

    @staticmethod
    def _normed(f, x):
        return f / trapz(f, x)

    def norm2d(self, density):
        """Normalize density (..., NM, Ny) to unit integral."""
        const = trapz(trapz(density, self.y_grid, axis=-1), self.M_grid, axis=-1)
        return density / np.expand_dims(const, axis=(-2, -1))

    # ------------------------------------------------------------------------
    # Tempered combination at fixed rho
    # ------------------------------------------------------------------------

    def combine_given_kappa(self, kappa):
        """Good/bad combination with each model's mixture raised to kappa.
        kappa = 1 is the manuscript's (untempered) method.

        Returns
          posterior   p(M, y | kappa), sigma marginalized, unit integral;
          log_Z_norm  log evidence with every tempered mixture renormalized to
                      unit integral, so pi(rho) Z_norm(rho) is a
                      unit-independent diagnostic posterior on rho. It
                      measures how much the inputs disagree, not how
                      correlated they are, and weights nothing else;
          P_i_full    Eq. (credibility): untempered responsibility
                      p g_i / (p g_i + (1-p) b_i) averaged over the joint
                      (tempered) posterior of (M, y, p, sigma);
          P_i_table2  the estimator of the manuscript notebook's Table 2 cell
                      (flat average over p), without its "1000 *" factor.
        """
        M_grid, y_grid, p_grid = self.M_grid, self.y_grid, self.p_grid
        NsM, Nsy = len(self.sigma_M_grid), len(self.sigma_y_grid)
        evidence = np.zeros((NsM, Nsy))
        log_Zn = np.zeros((NsM, Nsy))
        good_num = np.zeros((NsM, Nsy, N_MODELS))
        P_given_sigma = np.zeros((NsM, Nsy, len(M_grid), len(y_grid)))
        p_exp = p_grid[None, :, None, None]
        g_exp = self.g_stack[:, None, :, :]

        for iM in range(NsM):
            for iy in range(Nsy):
                b_exp = self.b_all[:, iM, iy][:, None, :, :]
                mixture = p_exp * g_exp + (1 - p_exp) * b_exp                  # (8, Np, NM, Ny)
                mixture_c = np.clip(mixture, 1e-300, None)
                mixture_w = mixture_c ** kappa
                L_p = np.prod(mixture_w, axis=0)                               # (Np, NM, Ny)
                likelihood = trapz(L_p, p_grid, axis=0)
                Z = trapz(trapz(likelihood, y_grid, axis=1), M_grid)
                evidence[iM, iy] = Z
                P_given_sigma[iM, iy] = likelihood / Z

                resp = p_exp * g_exp / mixture_c
                good_num[iM, iy] = trapz(
                    trapz(trapz(L_p[None] * resp, y_grid, axis=3), M_grid, axis=2), p_grid, axis=1)

                log_c = np.log(trapz(trapz(mixture_w, y_grid, axis=3), M_grid, axis=2))  # (8, Np)
                log_f = kappa * np.log(mixture_c).sum(axis=0) - log_c.sum(axis=0)[:, None, None]
                shift = log_f.max()
                log_Zn[iM, iy] = shift + np.log(trapz(trapz(trapz(
                    np.exp(log_f - shift), y_grid, axis=2), M_grid, axis=1), p_grid))

        sM, sy = self.sigma_M_grid, self.sigma_y_grid
        prior_sigma = self.prior_sigma_M[:, None] * self.prior_sigma_y[None, :]
        posterior_sigma = evidence * prior_sigma
        posterior_sigma /= trapz(trapz(posterior_sigma, sy, axis=1), sM)

        posterior = self.norm2d(trapz(
            trapz(P_given_sigma * posterior_sigma[:, :, None, None], sy, axis=1), sM, axis=0))

        log_w = log_Zn + np.log(prior_sigma)
        shift = log_w.max()
        log_Z_norm = shift + np.log(trapz(trapz(np.exp(log_w - shift), sy, axis=1), sM))

        P_i_full = (trapz(trapz(good_num * prior_sigma[:, :, None], sy, axis=1), sM, axis=0)
                    / trapz(trapz(evidence * prior_sigma, sy, axis=1), sM))
        P_i_table2 = self._table2_credibility(posterior, posterior_sigma)
        return posterior, log_Z_norm, P_i_full, P_i_table2

    def _table2_credibility(self, posterior, posterior_sigma):
        out = np.zeros(N_MODELS)
        pg = self.p_grid[:, None, None]
        for i in range(N_MODELS):
            b_m = self.norm2d(trapz(trapz(self.b_all[i] * posterior_sigma[:, :, None, None],
                                          self.sigma_y_grid, axis=1), self.sigma_M_grid, axis=0))
            g = self.g_stack[i]
            p_term = trapz(pg * g / (pg * g + (1 - pg) * b_m + 1e-300), self.p_grid, axis=0)
            out[i] = trapz(trapz(p_term * posterior, self.y_grid, axis=1), self.M_grid)
        return out

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
        density = self._marginalize_sys_axis(density, self.sys_scale_y, axis=1, spacing=self.dy)
        return self._marginalize_sys_axis(density, self.sys_scale_M, axis=0, spacing=self.dM)

    # ------------------------------------------------------------------------
    # (M, C) -> (M, R)
    # ------------------------------------------------------------------------

    def R_grid_for_MC(self):
        """R grid covering the whole (M, C) domain (~6.6-32 km)."""
        return np.linspace(K_KM * self.M_grid[0] / self.y_grid[-1],
                           K_KM * self.M_grid[-1] / self.y_grid[0], 300)

    def to_MR(self, post_MC, R_grid):
        """p(M, R) = p(M, C = K M / R) * K M / R^2, normalized on R_grid."""
        interp = RegularGridInterpolator((self.M_grid, self.y_grid), post_MC,
                                         bounds_error=False, fill_value=0.0)
        MM, RR = np.meshgrid(self.M_grid, R_grid, indexing="ij")
        p = interp(np.stack([MM, K_KM * MM / RR], axis=-1)) * K_KM * MM / RR ** 2
        return p / trapz(trapz(p, R_grid, axis=1), self.M_grid)


# ----------------------------------------------------------------------------
# Full run: rho sweep, scenarios, per-model credibilities
# ----------------------------------------------------------------------------


def _rho_mixture(sweep, prior_fn):
    """integral of pi(rho) X(rho) d rho over RHO_GRID, for X of any shape."""
    w = prior_fn(RHO_GRID)
    w = w / trapz(w, RHO_GRID)
    return trapz(sweep * w.reshape((-1,) + (1,) * (sweep.ndim - 1)), RHO_GRID, axis=0)


def run(space="MC", log=print):
    """Run the whole combination and return a dict of arrays (what gets
    saved to results/*.npz). Takes ~1-2 minutes."""
    import time

    S = Setup(space)
    log(f"Grids for ({space[0]}, {space[1]}): M {S.M_grid[0]:.2f}-{S.M_grid[-1]:.2f} "
        f"({len(S.M_grid)} pts), {space[1]} {S.y_grid[0]:.4g}-{S.y_grid[-1]:.4g} ({len(S.y_grid)} pts)")
    log(f"Sweeping rho over {len(RHO_GRID)} points...")
    t0 = time.time()
    sweep = [S.combine_given_kappa(design_effect_power(N_MODELS, rho)) for rho in RHO_GRID]
    post_sweep, log_Zn, Pi_full, Pi_table2 = (np.array(x) for x in zip(*sweep))
    log(f"  done in {time.time() - t0:.0f} s")

    uniform_nosys = S.norm2d(_rho_mixture(post_sweep, rho_prior_uniform))
    posts = dict(
        rho0=post_sweep[0],
        rho1=post_sweep[-1],
        beta41_nosys=S.norm2d(_rho_mixture(post_sweep, rho_prior_beta41)),
        uniform_nosys=uniform_nosys,
        uniform_sys=S.marginalize_sigma_sys(uniform_nosys),
    )

    # Diagnostic p(rho | D) under the Uniform prior (see combine_given_kappa).
    rho_post = np.exp(np.log(rho_prior_uniform(RHO_GRID)) + log_Zn - log_Zn.max())
    rho_post /= trapz(rho_post, RHO_GRID)

    out = dict(
        space=space, M_grid=S.M_grid, RHO_GRID=RHO_GRID,
        model_keys=np.array(MODEL_KEYS), log_Z_norm=log_Zn, rho_posterior_uniform=rho_post,
        P_i_full_sweep=Pi_full, P_i_table2_sweep=Pi_table2,
        sys_scale_M=S.sys_scale_M, sys_scale_y=S.sys_scale_y,
    )
    # Sigma_sys convolves the pooled posterior after the responsibilities are
    # set, so it cannot change P_i: the uniform_sys P_i are the uniform ones.
    for est, sw in [("full", Pi_full), ("table2", Pi_table2)]:
        out[f"P_i_{est}_rho0"] = sw[0]
        out[f"P_i_{est}_rho1"] = sw[-1]
        out[f"P_i_{est}_beta41_nosys"] = _rho_mixture(sw, rho_prior_beta41)
        out[f"P_i_{est}_uniform_nosys"] = out[f"P_i_{est}_uniform_sys"] = _rho_mixture(sw, rho_prior_uniform)

    if space == "MC":
        R_grid = S.R_grid_for_MC()
        out.update(C_grid=S.y_grid, R_grid=R_grid)
        for k, p in posts.items():
            out[f"{k}_MC"] = p
            out[f"{k}_MR"] = S.to_MR(p, R_grid)
    else:
        out.update(R_grid=S.y_grid)
        for k, p in posts.items():
            out[f"{k}_MR"] = p
    return out


def results_path(space="MC"):
    return os.path.join(RESULTS_DIR, f"combined_{space}.npz")


def load_results(space="MC"):
    path = results_path(space)
    if not os.path.exists(path):
        raise SystemExit(f"{os.path.relpath(path, REPO_ROOT)} not found: run "
                         f"`python3 revised_method/combine.py --space {space}` (or `make results`) first.")
    return dict(np.load(path))


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
