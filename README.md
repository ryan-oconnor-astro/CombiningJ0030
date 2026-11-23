# Combining the mass–radius posteriors of PSR J0030+0451

Reproduction package for the combination of eight published NICER mass–radius (M–R) posteriors of PSR J0030+0451. The combination uses a good/bad mixture model (Press 1996) with a kernel-density-estimate (KDE) good component. It allows for correlations between the inputs, since they share data, and for unknown model systematics.

## Quick start

```bash
pip install -r requirements.txt
make all
```

Requires Python 3 with `numpy`, `scipy`, `matplotlib` and `seaborn`. Both numpy < 2 and numpy ≥ 2 work. Use another interpreter with `make all PYTHON=python3.11`.

## Revised method (`revised_method/`)

The combination is the mixture model above, plus three additions:

1. **Correlation tempering.** Each measurement's mixture likelihood is raised to the power κ(ρ) = 1 / (1 + (N − 1)ρ) before the product over the N = 8 inputs. The correlation ρ is marginalized over a Uniform(0, 1) prior. This way, data shared between analyses is not counted N times.
2. **Group-level systematic uncertainty.** Σ_sys is applied as a Gaussian convolution of the pooled posterior and marginalized over half-normal priors.
3. **Combination in (M, C)**, where C = GM / (Rc²) is the compactness. The bad component and Σ_sys are diagonal in (M, C), so once they are mapped back to (M, R) they are correlated in M and R.

Files:

| File | Purpose |
|---|---|
| `core.py` | Shared engine: data loading, KDEs, bad component, tempering, Σ_sys, quantiles |
| `combine.py` | Runs the combination and writes `results/` (~2 min) |
| `headline.py` | Prints the headline numbers from the saved results (seconds) |
| `plots.py` | Writes the figures to `figures/` |

### Make targets

| Target | What it does |
|---|---|
| `make numbers` | Prints the median and 68% interval of M, R and C, the per-measurement credibilities P_i, and the sensitivity scenarios, all read from `results/combined_MC.npz` |
| `make results` | Runs `combine.py` if `results/combined_MC.npz` is missing or older than `core.py`. `make -B results` forces a rerun. |
| `make figures` | Remakes all five figures |
| `make all` | `results`, then `figures`, then `numbers` |

### Outputs

`revised_method/results/`:

- `combined_MC.npz`: grids and posteriors for every ρ scenario of the (M, C) combination.
- `combined_posterior_MC.txt`: 10,000 samples of the default combined posterior, as two columns: mass (M_sun) and equatorial radius (km). **This is the headline result.**
- `combined_MR.npz`: the same combination run in (M, R) on a 9–17 km radius grid, as a comparison (`python3 revised_method/combine.py --space MR`). No figure uses it, and `make` does not rebuild it.

`revised_method/figures/`:

| File | Content |
|---|---|
| `KDECompactness.png` | Compactness KDE of each input and the combined C marginal |
| `3x3Posterior.png` | Each input's M–R posterior against the combined posterior |
| `MarginalPosteriors.png` | Combined M–R posterior with its M and R marginals |
| `PosteriorvsOtherNeutronStars.png` | Combined posterior next to PSR J0437−4715 and GW170817 |
| `CorrelationTempering.png` | Combined (M, C) posterior for ρ = 0, ρ = 1 and the default ρ ~ Uniform(0, 1) |

The grid is M 0.95–2.2 M_sun, with C over the range of the pooled input samples (80 × 80 points). The tempered posteriors do not fall to zero inside this domain, so the quoted intervals are conditional on it.

## Original submission (independent combination)

These files hold the earlier version of the analysis, which treated the eight inputs as independent. They are kept for comparison and are superseded by `revised_method/`.

- `1D_Compactness_Combination.ipynb`: the mixture-model combination of the compactness (M/R) posteriors, with KDE smoothing and numerical marginalization over the nuisance parameters (`p`, `alpha`, `sigma`).
- `2D_MassRadius_Combination.ipynb`: the same analysis in 2D M–R space. It computes marginal posteriors, credible intervals and per-measurement credibilities.
- `combined_posterior.txt`: 10,000 (M [M_sun], R [km]) samples of that combination.
- `figures/`: that version's figures.

## Data availability & attribution

This repository redistributes, **for reproducibility and convenience**, several third-party datasets that are widely used in NICER pulse-profile modeling. All such material is in `datafiles/`. **We are not the creators of these data.** Each item remains governed by its original license (Creative Commons, as specified on the source pages). Please cite and attribute the original creators listed below when using these files.

### Contents and provenance

Directory `datafiles/` contains posterior samples of neutron-star mass and radius from published analyses, which are the inputs to the combination. Unless noted, each entry is a two-column text file (mass in solar masses, radius in km).

- `ST_PST19`: Riley et al. 2019 (NICER analysis of PSR J0030+0451, ST+PST model)
- `ST_PDT`: Vinciguerra et al. 2024 (ST+PDT model)
- `ST_PST`: Vinciguerra et al. 2024 (ST+PST model)
- `PDT_U`: Vinciguerra et al. 2024 (PDT-U model)
- `ST_U`: Vinciguerra et al. 2024 (ST-U model)
- `2spot`: Miller et al. 2019 (2-spot model)
- `3spot`: Miller et al. 2019 (3-spot model)
- `PDT_U26`: Kini et al. 2026 (PDT-U model)
- `GW170817_M1`: mass–radius posterior for the primary neutron star in GW170817
- `GW170817_M2`: mass–radius posterior for the secondary neutron star in GW170817
- `GW170817`: the GW170817 samples from which the M1 and M2 files are derived (six columns: m1, m2 [M_sun], Lambda1, Lambda2, R1, R2 [km])
- `J0437`: mass–radius posterior for PSR J0437−4715

These datasets are redistributed verbatim from their original sources. The NICER data for PSR J0030+0451 originate from the Bogdanov et al. 2019 supplementary materials.

### Licensing and attribution

* The third-party materials above are released by their authors under **Creative Commons** licenses (see each Zenodo or journal page for the specific terms).
* If you use these files, please **cite the original works** (below) and **link to the original sources** (DOIs below).
* If any attribution is incomplete or incorrect, please open an issue and we will correct it promptly.

### Citations

* Miller, M. C., Lamb, F. K., Dittmann, A., et al. 2019, The Astrophysical Journal Letters, 887, L24
* M. C. Miller, "NICER PSR J0030+0451 Illinois-Maryland MCMC Samples", The Astrophysical Journal Letters, vol. 887. Zenodo, p. L24, Dec. 12, 2019. doi: 10.5281/zenodo.3473466.
* Riley, T. E., Watts, A. L., Bogdanov, S., et al. 2019, The Astrophysical Journal Letters, 887, L21
* T. E. Riley, "A NICER View of PSR J0030+0451: Nested Samples for Millisecond Pulsar Parameter Estimation", Astrophysical Journal Letters, vol. 887. Zenodo, p. L21, Sep. 20, 2022. doi: 10.5281/zenodo.7096789.
* Vinciguerra, S., Salmi, T., Watts, A. L., et al. 2024, ApJ, 961, 62, doi: 10.3847/1538-4357/acfb83
* S. Vinciguerra, "An updated mass-radius analysis of the 2017-2018 NICER data set of PSR J0030+0451". Zenodo, Nov. 06, 2023. doi: 10.5281/zenodo.8239000.
* Kini, Y., Mauviard, L., Salmi, T., et al. 2026, arXiv preprint arXiv:2602.23743
* Y. Kini, "A NICER View of PSR J0030+0451: Updated Constraints from Six Years of NICER Observations". Zenodo, Feb. 23, 2026. doi: 10.5281/zenodo.18741942.

### Repackaging note

We redistribute the above files **verbatim** (no scientific modifications), with only minimal repackaging (archive structure and path names) to enable turnkey runs of our examples.
