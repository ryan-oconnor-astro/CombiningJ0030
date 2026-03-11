# Repository-specific guidance for GitHub Copilot and other AI agents

This repository is a small research codebase for combining posterior distributions of
neutron‑star mass–radius (M‑R) measurements.  Most of the work lives in Jupyter notebooks
and a handful of standalone Python scripts under `src/`.  An agent should read a few
notebooks to get the *big picture* before touching anything else.

## Big picture

* **Objective.**  Take a collection of published M‑R posterior samples (NICER pulse-profile
  results, Miller19, Vinciguerra23, Riley19, etc.) stored under `datafiles/` and
  produce a hierarchical combination with a "good/bad" mixture model.  There is also a
  companion analysis on compactness and a large TOV‑EOS polytrope sampler in
  `Run_new_combined_j0030_polytrope_speedup.py`.
* **Data flow.**  Each notebook/script reads one or more text files from `datafiles/`,
  loads mass and radius arrays with `np.loadtxt`, constructs a 2‑D or 1‑D KDE with
  `scipy.stats.gaussian_kde`, evaluates the KDE on a regular grid, and then builds a
  likelihood mixture over nuisance parameters (`p`, `\alpha`, `\sigma_M`,\`\sigma_R`).
  Posterior grids are normalized with `np.trapz`.  Plots are produced and written to
  `figures/`.
* **Utility functions.**  Several helper routines recur across notebooks:  
  `compute_hdr_levels`, `credible_intervals`/`compute_quantiles`, and the pattern of
  normalizing a KDE.  When you edit or extend, keep these idioms consistent or
  consider factoring them into a shared module under `src/`.

## Important files & folders

* `notebooks/` – additional exploratory notebooks; the active analysis lives in
  `NewMethodMassRadius.ipynb` (the other notebooks are earlier versions or testing).  
* `src/CompactnessCombination.py` and `src/MRCombo.py` – standalone scripts that
  perform lightweight combinations with hard‑coded absolute paths.  They are useful
  as examples of the procedural style the notebooks follow.
* `src/Run_new_combined_j0030_polytrope_speedup.py` – a large, numba‑accelerated
  TOV/EoS solver used for EOS inference.  It defines many physical constants and
  vectorised routines; treat it as a self‑contained library.  If you modify it,
  please add tests or notebook examples because it is the most fragile piece.
* `datafiles/` – contains the actual posterior samples for each model.  The names in
  the `MODELS` dictionary or `data_files` lists must match these subfolders.
* `figures/` – output directory for PNG figures; notebooks generally call
  `plt.savefig('figures/...')` without checking for existence.
* `requirements.txt` – currently only lists `seaborn`.  The code also depends on
  `numpy`, `scipy`, `matplotlib`, `numba`, `ultranest`, etc.  Agents should assume a
  conda/venv environment is prepared manually or extend `requirements.txt` accordingly.
* `tests/` – placeholder; there are no real tests yet.  Add unit tests here when you
  add new Python code (see patterns below).
* `README.md` – documents data provenance and licensing; not a technical guide but good
  context for why the repository exists.

## Development workflows

1. **Interactive exploration.**  Open `NewMethodMassRadius.ipynb` (or other notebooks)
   and run cells sequentially.  Adjust grid definitions (`M_grid`, `R_grid`,
   `sigma_M_grid`, etc.) at the top of the notebook to experiment with resolution.
   Use the `MODELS` dictionary to add/remove data sets.
2. **Script execution.**  Execute a script via `python src/CompactnessCombination.py`
   or `python src/MRCombo.py`.  These scripts are not packaged; they assume you are
   running from the repository root and may contain hard‑coded absolute paths.  If you
   change locations, convert them to relative paths using `Path(__file__).parent`.
3. **Environment setup.**  Run `pip install -r requirements.txt` and manually install
   any missing libraries shown by import errors.  Use the same Python interpreter for
   running notebooks and scripts to avoid inconsistencies.
4. **Plots and figures.**  Every notebook configures `matplotlib`/`seaborn` with
   LaTeX‑style fonts and saves to `figures/`.  When adding new plots, follow the
   existing styling (use `plt.style.use('seaborn-v0_8-white')`, set font sizes, etc.).
   Clean up old figures if you change naming schemes.
5. **Adding datasets.**  New posterior files go under `datafiles/` with the same two‑column
   format (M, R).  Add an entry to the `MODELS` dict or `data_files` list and ensure
   the label/color metadata are provided.

## Coding conventions & patterns

* **Numerical grids and normalization.**  Always normalise probability densities with
  `np.trapz` over the relevant grid.  When marginalising, integrate first along the
  appropriate axis and renormalise.  Do not assume uniform grid spacing if you change
  grid definitions; compute `grid[1] - grid[0]` for step size.
* **Mixture model structure.**  The likelihood for each model is `p * g + (1-p) * b`,
  where `g` is the KDE of the published posterior and `b` is a broad "bad" distribution
  (usually Gaussian in M/R or compactness).  The hierarchical marginalisation loops
  over `p_grid` (and sometimes `\alpha`, `\sigma_M`, `\sigma_R`).  Keep the code that
  computes `L_p` and the posterior consistent across notebooks; copy the existing
  nested-loop structure if you add new nuisance parameters.
* **Helper functions.**  There are two styles for quantiles/HDR calculations: the
  simple `credible_intervals` used early in `NewMethodMassRadius.ipynb` and the more
  general `compute_hdr_levels` used later.  Both work on flattened arrays and assume
  the pdf is already normalised.  If you need a new statistic, add it near the top of
  the notebook or script so it is easy to find.
* **Absolute paths.**  Many scripts use absolute paths (`/Users/ryanoconnor/...`).
  Before committing changes, convert them to relative or parameterise them with
  environment variables.  The notebooks generally assume the working directory is the
  repository root; run them from there.
* **Imports.**  Top‑level imports are the standard scientific Python stack.  Additional
  dependencies (e.g. `ultranest`, `numba`) live only in the big EOS script.  Avoid
  adding heavy dependencies unless strictly necessary for reproduction.
* **Notebook cells.**  Keep cells focused: data loading, grid setup, combination, and
  plotting are separate.  Avoid redefining variables later; if you need a fresh run,
  restart the kernel rather than reusing state.

## Testing & validation

* There are currently no automated tests.  When you add a new function or class, write a
  simple test in `tests/` that exercises it.  Example tests could verify that
  `credible_intervals` returns the correct percentiles for a known distribution or that
  `solveTOV_single_jit` produces monotonic mass–radius curves for a simple polytrope.
* Notebooks can serve as regression tests: rerun the notebook after modifications and
  check that saved figures or printed quantiles match previous runs.  Consider using
  `nbconvert`/`papermill` in a CI pipeline if this becomes important.

## Common pitfalls & tips

* **Normalization mistakes.**  Forgetting to renormalize after marginalising is the most
  common source of incorrect results in these scripts.  Watch for comments like
  "Integrate properly over σ grid" and don't skip them.
* **Grid resolution.**  The default 100×100 grids are fine for quick experiments but
  can hide structure when you change prior ranges.  Scale up cautiously; the nested
  loops have cubic cost in the number of `sigma` bins.
* **Hard‑coded constants.**  The EOS script defines many physical constants at the top
  (c, G, Msun, unit conversions).  If you need to port code to a different unit system,
  modify these values consistently.
* **Data provenance.**  The `README.md` emphasises that the included datasets are
  third‑party and must be cited.  When writing new code that reads `datafiles/`, do not
  modify the raw files.

---

Please review this draft and let me know if any sections are unclear or if there are
other patterns you rely on that should be documented.  I'm happy to expand or refine
these instructions.
