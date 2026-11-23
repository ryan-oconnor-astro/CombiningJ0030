"""
Run the revised combination and save the results.

    python3 revised_method/combine.py              # (M, C): the revised method
    python3 revised_method/combine.py --space MR   # (M, R), 9-17 km grid: optional comparison, not used by any figure

The grid is the manuscript notebook's, M 0.95-2.2 Msun (~2 minutes per run).

Each run writes revised_method/results/combined_<space>.npz.
The (M, C) run also writes revised_method/results/combined_posterior_MC.txt: 10,000
(M [Msun], R [km]) samples of the default posterior, the counterpart of
combined_posterior.txt (the input to the EoS-agnostic R_1.4 / Lambda_1.4 analysis).
At the end it prints the headline numbers (same output as headline.py).
"""

import argparse
import os

import numpy as np

import core
import headline


def write_samples(res, path, n=10000, seed=0):
    """Draw grid points from the default (M, R) posterior, as the manuscript
    notebook did for combined_posterior.txt (with a fixed seed here)."""
    post, _, _ = core.marginals_MR(res, core.DEFAULT_SCENARIO)
    MM, RR = np.meshgrid(res["M_grid"], res["R_grid"], indexing="ij")
    w = post.ravel() / post.sum()
    idx = np.random.default_rng(seed).choice(w.size, size=n, p=w)
    np.savetxt(path, np.column_stack([MM.ravel()[idx], RR.ravel()[idx]]), header="M R")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", choices=["MC", "MR"], default="MC")
    args = parser.parse_args()

    res = core.run(args.space)
    os.makedirs(core.RESULTS_DIR, exist_ok=True)
    path = core.results_path(args.space)
    np.savez(path, **res)
    print(f"wrote {os.path.relpath(path, core.REPO_ROOT)}")

    if args.space == "MC":
        samples_path = os.path.join(core.RESULTS_DIR, "combined_posterior_MC.txt")
        write_samples(res, samples_path)
        print(f"wrote {os.path.relpath(samples_path, core.REPO_ROOT)}")
        print()
        headline.report_MC(res)
    else:
        print()
        headline.report_MR(res)


if __name__ == "__main__":
    main()
