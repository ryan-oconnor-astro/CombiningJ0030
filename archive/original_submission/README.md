# Original submission (superseded)

This folder keeps the first version of the analysis, which treated the eight input posteriors as independent measurements. It has no correlation tempering and no common systematic displacement. Its results are narrower than the method in the main notebooks and should not be used. They are kept for reference only.

- `1D_Compactness_Combination.ipynb`: combination of the compactness posteriors.
- `2D_MassRadius_Combination.ipynb`: the same combination in mass and radius, with per-input consistency probabilities.
- `combined_posterior.txt`: 10,000 (M [Msun], R [km]) samples of that combination.
- `figures/`: that version's figures.

The notebooks read the input posteriors from `../../data/` and are run from this folder.
