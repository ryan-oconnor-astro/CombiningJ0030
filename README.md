## Data availability & attribution

This repository redistributes, **for reproducibility and convenience**, several third-party datasets and instrument files used widely in NICER pulse-profile modeling. All such material is bundled in TK **We are not the creators of these data.** Each item remains governed by its original license (Creative Commons, as specified on the source pages). Please cite and attribute the original creators listed below when using these files.

### Contents and provenance

* **Neutron Star Mass-Radius Posterior Samples**

  * Directory: `datafiles/`
    This directory contains posterior samples for neutron star mass and radius measurements from various published analyses. Each entry is a two-column text file (mass in solar masses, radius in km) or a folder containing such files.

    - `ST_PST19`: Posterior from Riley et al. 2019 (NICER analysis of PSR J0030+0451, ST+PST model)
    - `ST_PDT`: Posterior from Vinciguerra et al. 2023 (NICER analysis of PSR J0030+0451, ST+PDT model)
    - `ST_PST`: Posterior from Vinciguerra et al. 2023 (NICER analysis of PSR J0030+0451, ST+PST model)
    - `PDT_U`: Posterior from Vinciguerra et al. 2023 (NICER analysis of PSR J0030+0451, PDT-U model)
    - `ST_U`: Posterior from Vinciguerra et al. 2023 (NICER analysis of PSR J0030+0451, ST-U model)
    - `2spot`: Posterior from Miller et al. 2019 (NICER analysis of PSR J0030+0451, 2-spot model)
    - `3spot`: Posterior from Miller et al. 2019 (NICER analysis of PSR J0030+0451, 3-spot model)
    - `PDT_U26`: Posterior from Kini et al. 2026 (NICER analysis of PSR J0030+0451, PDT-U model)
    - `GW170817_M1`: Mass-radius posterior for the primary neutron star in GW170817
    - `GW170817_M2`: Mass-radius posterior for the secondary neutron star in GW170817
    - `J0437`: Mass-radius posterior for PSR J0437-4715
    - `GW170817`: Additional GW170817 data from which M1 and M2 data are derived

    Notes: These datasets are redistributed verbatim from their original sources for reproducibility. The NICER data for PSR J0030+0451 originates from Bogdanov et al. 2019 supplementary materials.


### Licensing and attribution

* The third-party materials above are released by their authors under **Creative Commons** licenses (see each Zenodo/journal page for the specific terms).
* If you use these files, please **cite the original works** (examples below) and **link to the original sources** (DOIs provided above).
* If any attribution is incom ApJ 975 202plete or incorrect, please open an issue and we will correct it promptly.

### Suggested citations

* Miller, M. C., Lamb, F. K., Dittmann, A., et al. 2019, The
Astrophysical Journal Letters, 887, L24
* Riley, T. E., Watts, A. L., Bogdanov, S., et al. 2019, The
Astrophysical Journal Letters, 887, L21
* Vinciguerra, S., Salmi, T., Watts, A. L., et al. 2024, ApJ,
961, 62, doi: 10.3847/1538-4357/acfb83
* Kini, Y., Mauviard, L., Salmi, T., et al. 2026, arXiv
preprint arXiv:2602.23743


### Repackaging note

We redistribute the above files **verbatim** (no scientific modifications), with only minimal repackaging (archive structure and path names) to enable turnkey runs of our examples and benchmarks.

---