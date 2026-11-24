# Combining the mass–radius posteriors of PSR J0030+0451

Code and data to reproduce the results of *Combining the Mass–Radius Posteriors of PSR J0030+0451 Allowing for Unknown Model Systematics*.

Eight published NICER mass–radius posteriors of PSR J0030+0451 are combined into one, with three ingredients:

- **Good/bad mixture model.** Each input is a mixture of a "good" kernel density estimate of its samples and a broad "bad" Gaussian that allows for unknown model systematics (Press 1996).
- **Correlation tempering.** This accounts for the inputs sharing the same NICER data.
- **Common systematic displacement.** This allows for a bias shared by every analysis.

**Result:** M = 1.53 +0.31/−0.27 M☉, R = 13.44 +2.95/−2.38 km, C = 0.169 +0.018/−0.021 (medians and 68% intervals). Samples are in [`results/combined_posterior.txt`](results/combined_posterior.txt).

## Quick start

```bash
pip install -r requirements.txt
jupyter lab
```

Then run the notebooks in order from the repository root. They are saved with their outputs, so they can also be read without running anything.

## Notebooks

| Notebook | What it does | Run time |
|---|---|---|
| [`1_Input_Posteriors.ipynb`](1_Input_Posteriors.ipynb) | Loads the eight input posteriors and summarizes their mass, radius and compactness (input rows of the summary table) | seconds |
| [`2_Combined_Posterior.ipynb`](2_Combined_Posterior.ipynb) | Builds the combination step by step (good and bad components, correlation tempering, common systematics), reports the adopted result, the effect of tempering and the ρ diagnostic, and writes `results/` | 2–3 min |
| [`3_Figures_and_Consistency.ipynb`](3_Figures_and_Consistency.ipynb) | Makes the paper's figures in `figures/` and the consistency probability P_i of each input | ~2 min |

The method lives in the small package [`j0030/`](j0030/):
- `method.py` holds the combination;
- `plotting.py` holds the figures.

The notebooks call these and explain each step. Notebook 2 has a `RECOMPUTE` flag; set it to `False` to load the saved results instead of rerunning the combination.

The paper's R₁.₄ and Λ₁.₄ come from a separate equation-of-state-agnostic analysis (Huang et al. 2024, 2025) run on `results/combined_posterior.txt`. That analysis is not part of this repository.

## Repository layout

```
1_Input_Posteriors.ipynb          the eight inputs
2_Combined_Posterior.ipynb        the combination and the headline result
3_Figures_and_Consistency.ipynb   figures and per-input consistency
j0030/                            method.py (combination), plotting.py (figures)
data/                             published posterior samples (see below)
results/                          combined_posterior.txt (10,000 M, R samples), combined_MC.npz (all gridded posteriors)
figures/                          the paper's figures
archive/original_submission/      an earlier version that treated the inputs as independent (superseded)
```

Requires Python 3 with `numpy`, `scipy`, `matplotlib`, `seaborn` and Jupyter. Both numpy < 2 and numpy ≥ 2 work. Figures use LaTeX for their text when it is installed.

## Data availability & attribution

This repository redistributes, **for reproducibility and convenience**, several third-party datasets that are widely used in NICER pulse-profile modeling. All such material is in `data/`. **We are not the creators of these data.** Each item remains governed by its original license (Creative Commons, as specified on the source pages). Please cite and attribute the original creators listed below when using these files.

### Contents and provenance

`data/` contains posterior samples of neutron-star mass and radius from published analyses, which are the inputs to the combination. Unless noted, each entry is a two-column text file (mass in solar masses, radius in km).

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
