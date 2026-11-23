"""
Print the headline numbers of the revised combination from the saved results
(no recomputation; takes a second).

    python3 revised_method/headline.py

Needs revised_method/results/combined_MC.npz (from combine.py).

All intervals are medians with 68% (16th-84th percentile) half-widths.
"""

import argparse

import core

MANUSCRIPT = dict(M="1.46 +0.09/-0.08", R="12.69 +0.64/-0.55", C="0.172 +0.006/-0.007")
MANUSCRIPT_P_I = {  # manuscript.tex Table 2
    "PDT_U26": 0.8076, "3spot": 0.7895, "2spot": 0.7114, "ST_PDT": 0.6763,
    "ST_PST19": 0.6270, "PDT_U": 0.5798, "ST_U": 0.4599, "ST_PST": 0.1974,
}


def fmt(q, nd):
    return f"{q['median']:.{nd}f} +{q['hi']:.{nd}f}/-{q['lo']:.{nd}f}"


def width(q):
    return q["lo"] + q["hi"]


def label(key):
    return core.MODELS[key]["label"].replace("--", "-")


def summarize(res, scenario):
    _, P_M, P_R = core.marginals_MR(res, scenario)
    out = dict(M=core.quantiles(res["M_grid"], P_M), R=core.quantiles(res["R_grid"], P_R))
    if f"{scenario}_MC" in res:
        out["C"] = core.quantiles(res["C_grid"], core.marginal_C(res, scenario))
    return out


def print_scenarios(res, has_C):
    rows = {k: summarize(res, k) for k in core.SCENARIOS}
    ref = rows["rho0"]
    cols = ["M", "R"] + (["C"] if has_C else [])
    nd = dict(M=3, R=2, C=4)
    print(f"{'scenario':<56}" + "".join(f"{c + ' (68%)':>24}" for c in cols)
          + "".join(f"{c + '/rho0':>8}" for c in cols))
    for k, desc in core.SCENARIOS.items():
        r = rows[k]
        print(f"{desc:<56}" + "".join(f"{fmt(r[c], nd[c]):>24}" for c in cols)
              + "".join(f"{width(r[c]) / width(ref[c]):>8.2f}" for c in cols))
    print("(x/rho0 = 68% width relative to this run's own rho=0 result.)")


def print_P_i(res):
    keys = list(res["model_keys"])
    order = sorted(keys, key=lambda k: -res[f"P_i_full_{core.DEFAULT_SCENARIO}"][keys.index(k)])
    for est, desc in [("full", "Eq. (credibility): joint posterior over (M, C, p, sigma)"),
                      ("table2", "manuscript notebook's Table 2 estimator (flat average over p, no '1000 *')")]:
        print(f"\nPer-model credibility P_i, {desc}")
        print(f"{'model':<24}{'default':>10}{'rho=0':>10}{'rho=1':>10}{'Beta(4,1)':>11}{'ms Table 2':>12}")
        for k in order:
            i = keys.index(k)
            v = [res[f"P_i_{est}_{s}"][i] for s in ("uniform_sys", "rho0", "rho1", "beta41_nosys")]
            print(f"{label(k):<24}{v[0]:>10.3f}{v[1]:>10.3f}{v[2]:>10.3f}{v[3]:>11.3f}{MANUSCRIPT_P_I[k]:>12.3f}")
    print("(default = rho ~ Uniform(0,1); Sigma_sys does not change P_i. 'ms Table 2' is the"
          " manuscript's (M, R), rho=0 value.)")


def print_rho_posterior(res):
    rho, post = res["RHO_GRID"], res["rho_posterior_uniform"]
    q = core.quantiles(rho, post)
    print(f"\nDiagnostic p(rho | D) with a Uniform(0,1) prior: rho = {fmt(q, 2)}")
    print("(How much the inputs disagree, not how correlated they are. It is not used to weight"
          " any result: the M-R posterior averages over the prior pi(rho).)")


def domain(res):
    return (f"grid: M {res['M_grid'][0]:.2f}-{res['M_grid'][-1]:.2f} Msun, "
            f"C {res['C_grid'][0]:.3f}-{res['C_grid'][-1]:.3f}")


def report_MC(res):
    d = summarize(res, core.DEFAULT_SCENARIO)
    bar = "=" * 78
    print(bar)
    print("Revised combination: (M, C) space, rho ~ Uniform(0,1), Sigma_sys marginalized")
    print(f"  [{domain(res)}]")
    print(bar)
    print(f"  M = {fmt(d['M'], 2)} Msun        (manuscript: {MANUSCRIPT['M']})")
    print(f"  R = {fmt(d['R'], 2)} km         (manuscript: {MANUSCRIPT['R']})")
    print(f"  C = {fmt(d['C'], 3)}           (manuscript: {MANUSCRIPT['C']})")
    print(bar)
    print("\nAll scenarios of the (M, C) run:")
    print_scenarios(res, has_C=True)
    print_P_i(res)
    print_rho_posterior(res)
    print(f"\nSystematic scales: s_M = {float(res['sys_scale_M']):.4f} Msun, "
          f"s_C = {float(res['sys_scale_y']):.5f} (std of the eight per-model medians)")
    print("\nR_1.4 and Lambda_1.4 are not computed in this repository: they come from the EoS-agnostic"
          "\nanalysis of Huang et al. (2024, 2025), run on revised_method/results/combined_posterior_MC.txt.")


def report_MR(res):
    print(f"(M, R) comparison run, M {res['M_grid'][0]:.2f}-{res['M_grid'][-1]:.2f} Msun, "
          f"R {res['R_grid'][0]:.1f}-{res['R_grid'][-1]:.1f} km:")
    print_scenarios(res, has_C=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    report_MC(core.load_results("MC"))


if __name__ == "__main__":
    main()
