Published manuscript (current)
Each measurement's likelihood is a good/bad mixture:
$$\ell_i(C) = p,P_{G,i}(C) + (1-p),P_{B,i}(C)$$

with bad component
$$P_{B,i}(C) = \mathcal{N}!\big(C;\ \mu_i+\Delta_i,\ (\alpha_i S_i)^2\big)$$

The 8 measurements are combined as a straight product (full independence assumed):
$$P(C\mid D)\ \propto\ \pi(C)\int dp,d\alpha,d\Delta\ \prod_{i=1}^{8}\ell_i(C)$$

That product is the referee's objection: it treats all 8 as carrying independent information, when several share the same underlying data.

What's implemented now (correlation-tempered pooling)
1. Temper the product by a shared exponent κ before combining:
$$\log L_\kappa(C) = \kappa(\rho)\sum_{i=1}^{8}\log \ell_i(C) \quad\Longleftrightarrow\quad L_\kappa(C) = \Big[\prod_{i=1}^8 \ell_i(C)\Big]^{\kappa(\rho)}$$

2. The tempering factor (design-effect correction, standard in meta-analysis for pseudo-replication):
$$\kappa(\rho) = \frac{1}{1+(N-1)\rho}, \qquad N=8$$

$\rho=0 \Rightarrow \kappa=1$: recovers the manuscript's product exactly.
$\rho=1 \Rightarrow \kappa=1/8$: treats all 8 as one effective measurement.
3. Marginalize over ρ instead of fixing it at 0:
$$P(C\mid D)\ \propto\ \pi(C)\int_0^1 d\rho\ \pi(\rho)\ L_\kappa(C)\big|_{\kappa=\kappa(\rho)}$$
with $\pi(\rho) = \text{Beta}(4,1)$ or $\text{Uniform}(0,1)$ (tried as two separate diagnostics, no single choice adopted yet).

4. Group-level systematic (new ingredient, not in the manuscript at all): convolve with a shared Gaussian blur,
$$\tilde P(C\mid D) = \int d\sigma_{\rm sys}\ \pi(\sigma_{\rm sys})\ \Big[\mathcal{N}(\cdot;0,\sigma_{\rm sys}^2) * P(C\mid D)\Big]$$
with $\sigma_{\rm sys}\sim\text{HalfNormal}(s_C)$, $s_C$ = the scatter of the 8 models' own medians.

The one-line difference to say out loud
The manuscript computes $\prod_i \ell_i$ (Eq. above, implicit $\rho=0$). The new method raises that product to a power $\kappa(\rho)\le1$ and integrates over how correlated the 8 inputs might be, instead of assuming $\rho=0$ outright — then optionally adds one more shared "unknown systematic" blur on top.

Everything else (the KDE-based good component, the per-measurement bad-component shift/scale machinery) is untouched — the new equations sit around the existing per-measurement pieces, they don't replace them.