2D (Mass-Radius) extension of correlation-tempered pooling
============================================================

This extends the 1D compactness result (see `summary.md` in this folder) to
the manuscript's actual headline quantities, M and R. Same tempering
machinery, applied to the full 2D good/bad pipeline from
`2D_MassRadius_Combination.ipynb`. Script: `tests/correlation_tempered_pooling_2d.py`.

Current manuscript (unpublished)
-------------------------------
Each measurement's likelihood is a good/bad mixture over the 2D point $(M,R)$:
$$\ell_i(M,R) = p\,P_{G,i}(M,R) + (1-p)\,P_{B,i}(M,R)$$

with a **diagonal** bad component (independent Gaussians in M and R):
$$P_{B,i}(M,R) = \mathcal{N}\!\big(M;\ \mu_{M,i}+\Delta_{M,i},\ (\alpha_i S_{M,i})^2\big)\cdot
\mathcal{N}\!\big(R;\ \mu_{R,i}+\Delta_{R,i},\ (\alpha_i S_{R,i})^2\big)$$

The 8 measurements are combined as a straight product (full independence assumed):
$$P(M,R\mid D)\ \propto\ \pi(M,R)\int dp\,d\alpha\,d\Delta\ \prod_{i=1}^{8}\ell_i(M,R)$$

Same referee objection as in 1D: this product double-counts the shared NICER/XMM data
across the 8 inputs.

What's implemented now (correlation-tempered pooling, 2D)
-----------------------------------------------------------
1. Same tempering exponent $\kappa(\rho)$ as the 1D case, applied to the 2D product:
$$L_\kappa(M,R) = \Big[\prod_{i=1}^8 \ell_i(M,R)\Big]^{\kappa(\rho)}, \qquad
\kappa(\rho) = \frac{1}{1+(N-1)\rho},\ N=8$$

2. Marginalize over $\rho$:
$$P(M,R\mid D)\ \propto\ \pi(M,R)\int_0^1 d\rho\ \pi(\rho)\ L_\kappa(M,R)\big|_{\kappa=\kappa(\rho)}$$
with $\pi(\rho)=\text{Beta}(4,1)$ or $\text{Uniform}(0,1)$, same as 1D.

3. Group-level systematic, now a **diagonal** 2x2 covariance
$\Sigma_{\rm sys} = \mathrm{diag}(\sigma_{M,\rm sys}^2,\ \sigma_{R,\rm sys}^2)$:
$$\tilde P(M,R\mid D) = \int d\sigma_{M,\rm sys}\,d\sigma_{R,\rm sys}\ \pi(\sigma_{M,\rm sys})\pi(\sigma_{R,\rm sys})\
\Big[\mathcal{N}(\cdot,\cdot;0,\Sigma_{\rm sys}) * P(M,R\mid D)\Big]$$
with $\sigma_{M,\rm sys}\sim\text{HalfNormal}(s_M)$, $\sigma_{R,\rm sys}\sim\text{HalfNormal}(s_R)$,
$s_M,s_R$ = scatter of the 8 models' own M, R medians. Because $\Sigma_{\rm sys}$ has
no M-R cross-term, this convolution is separable, so the double marginalization above
reduces exactly to two sequential 1D marginalizations (blur in R, then blur in M) —
computationally cheap, not a joint 2D grid search.

**Deliberately deferred, not solved**: the bad component $P_{B,i}$ and the systematic
$\Sigma_{\rm sys}$ are both diagonal (no M-R correlation), matching the manuscript's
existing diagonal treatment. This does *not* yet address the referee's secondary
criticism that $\Sigma_{S,i}$ should carry M-R correlation (since compactness $M/R$ is
the primary PPM observable). Flagged as an open follow-up, not attempted here.

Numeric results
----------------
Sanity check at $\rho=0,\ \sigma_{\rm sys}=0$ (should reproduce the manuscript):

| | M | R |
|---|---|---|
| script | $1.463^{+0.089}_{-0.084}$ | $12.677^{+0.638}_{-0.546}$ |
| manuscript | $1.460^{+0.090}_{-0.080}$ | $12.690^{+0.640}_{-0.550}$ |

Scenario comparison (widths relative to the current manuscript's 68% width):

| Scenario | M (median) | R (median) | M-width/pub | R-width/pub |
|---|---|---|---|---|
| current manuscript ($\rho=0$) | 1.463 | 12.677 | 1.00x | 1.00x |
| fully correlated ($\rho=1$) | 1.533 | 13.129 | 4.22x | 3.90x |
| $\rho\sim\text{Uniform}(0,1)$ | 1.497 | 12.991 | 3.23x | 2.96x |
| $\rho\sim\text{Beta}(4,1)$ | 1.525 | 13.128 | 4.01x | 3.69x |
| $\rho\sim\text{Beta}(4,1)$, $\Sigma_{\rm sys}$ marginalized | 1.534 | 13.123 | 4.02x | 3.68x |

The one-line difference to say out loud
-----------------------------------------
Same story as 1D, now on the actual $(M,R)$ posterior: the manuscript's product
($\rho=0$) is recovered as one diagnostic endpoint, not the adopted result. Tempering
by $\kappa(\rho)$ and marginalizing $\rho$ inflates the 68% widths in both M and R by
roughly **3.2-4.2x** relative to the current manuscript numbers, depending on the $\rho$
prior — and this matches the ~2.9-4.3x inflation already seen in the independent 1D
compactness calculation, which is a useful cross-check that the two aren't telling
different stories.

Figures: `MR_posteriors_diagnostic_scenarios.png` (marginal M, R), `MR_contours_diagnostic_scenarios.png`
(joint M-R, 68%/95% HDR), `sigma_sys_effect_MR.png`.
