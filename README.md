# Active learning of spinel oxygen carriers

Code for "Prediction targets and surrogate models for active learning of spinel oxygen carriers" by Alfred Juhlin Onbeck, Ivana Staničić, Radhey Shyam Yadav, Tobias Mattisson and Anders Hellman (Chalmers University of Technology).

Ten active learning strategies, each run five times, search high-entropy spinel oxides for a high oxygen transfer capacity (OTC) in chemical looping. A strategy pairs a surrogate model (deep ensemble, Gaussian process or Monte Carlo dropout) with a prediction target (OTC, OTC clipped at the composition's ceiling, the 16 Gibbs energies of the two reactors, or the air-reactor Gibbs energy with the vacancy count as an input). Each candidate is evaluated with special quasirandom structures (SQS) at 0 to 7 oxygen vacancies, relaxed with CHGNet, with vibrational free energies from Phonopy.

The data of the article (run databases, run records, structures, derived data, figures) and the container image of the runs are in the Zenodo record [10.5281/zenodo.21283967](https://doi.org/10.5281/zenodo.21283967).

## Contents

| Path | Contents |
|---|---|
| `active-learning/src/` | Active learning loop (`active_learning_HEO.py`), Wasserstein autoencoder, surrogates, Thompson sampling, and the held-out test (`holdout_censoring.py`) |
| `metal/` | Composition space, stability screen (spinel tolerance factor, charge neutrality), SQS generation with icet, CHGNet relaxations, Phonopy calculations, Materials Project formation energies, Gibbs energy terms |
| `oxygen-carrier-discovery/heoc/pyscripts/` | Evaluation pipeline scripts, called by the active learning loop or run on their own |
| `analysis/` | Scripts that compute every number, table and figure of the article from the run databases (`run_all.sh`) |
| `run_arm.sh` | Runs one strategy with the settings of the article |
| `tests/` | Cycle-1 selections of the article runs and a script that compares a rerun with them |
| `container/` | Apptainer (`al.def`) and Docker recipes of the run environment, with the package versions |

## Reproducing the analysis

The analysis needs only the Zenodo data and Python 3.10 or later with numpy, pandas, scipy, matplotlib and ase.

1. Download `training_databases.zip`, `run_records.zip`, `sqs_validation.zip` and `analysis_outputs.zip` from the Zenodo record and unzip them into `data/`. `analysis_outputs.zip` holds the outputs of the article, for comparison, and the held-out test results that `run_all.sh` reuses.
2. Link the data into the directory layout of the runs, then run the scripts:

```bash
python3 analysis/scripts/restore_original_layout.py
analysis/run_all.sh
```

The outputs go to `analysis/derived_data/`, `analysis/tables/` and `analysis/figures/`, and match `data/analysis_outputs/` up to rounding in the last digit. `run_all.sh` takes a few minutes. `run_all.sh --with-calibration` also runs the simulation behind the Games–Howell calibration in the Supplementary Information, which takes about 1.5 h. `max_otc_ceiling.py` imports `metal`, so it needs the dependencies of the container, and `run_all.sh` skips it otherwise. The held-out test results in `analysis_outputs.zip` (`derived_data/holdout_test/`) come from `active-learning/src/holdout_censoring.py`, which trains every surrogate and takes several hours on a GPU. On a CPU the surrogates train slightly differently, so the metrics differ from the deposited ones by up to 0.13 wt% (seed 1 in our check, mostly the deep ensembles). `analysis/gibbs_pipeline.tex` was written by hand and is copied to `tables/`.

## Running the active learning

The runs used an NVIDIA A100 GPU and the container image `pytorch_al_img.sif` of the Zenodo record. The recipes in `container/` rebuild it (`apptainer build pytorch_al_img.sif container/al.def`, or `docker build -t spinel-al container/`).

1. Unzip `initial_dataset.zip` of the Zenodo record into `data/`. It holds the 1,000 initial compositions shared by all runs, the crystal space and the spinel prototype.
2. Put `pytorch_al_img.sif` in `container/`.
3. Run a strategy, for example the Gaussian process with the Gibbs-vector target and seed 1:

```bash
export PMG_MAPI_KEY=<your Materials Project API key>
./run_arm.sh mogp 1        # all 20 acquisition cycles
./run_arm.sh mogp 1 2      # or only two
```

| Arm | Strategy | Arm | Strategy |
|---|---|---|---|
| `de` | deep ensemble, scalar OTC | `gp` | Gaussian process, scalar OTC |
| `de_clip` | deep ensemble, clipped OTC | `gp_clip` | Gaussian process, clipped OTC |
| `de_land` | deep ensemble, Gibbs vector | `mogp` | Gaussian process, Gibbs vector |
| `dde` | deep ensemble, vacancy input | `dgp` | Gaussian process, vacancy input |
| `mc` | Monte Carlo dropout, scalar OTC | `mc_clip` | Monte Carlo dropout, clipped OTC |
| `ceil` | ceiling-only reference, no surrogate | | |

Seed *k* is run *k* of the article. A run writes to `oxygen-carrier-discovery/heoc/spinel_rev_<arm>_s<seed>/`, and the same command resumes it. On one A100, an acquisition cycle takes about 5 minutes for training and selection and about 3.5 hours for the CHGNet relaxations and phonon calculations of the 80 structures, so a run of 20 cycles takes about three days. The article runs spread each evaluation over five GPUs. Formation energies come from the current Materials Project database, which has changed since the runs, so Gibbs energies can differ slightly from the deposited values. Set `AL_RUNNER=docker` to use Docker instead of Apptainer.

To check the setup without the expensive evaluation, stop after the candidates of cycle 1 are selected and compare them with the article run:

```bash
AL_STOP_AFTER_SAMPLING=1 ./run_arm.sh mogp 1
python3 tests/compare_selection.py mogp 1
```

On A100 GPUs, seed 1 reproduced all ten selections for nine strategies and nine of ten for `de_land`, where two candidates tied to three decimals. On other hardware, small floating-point differences grow during autoencoder training, so the selections differ but should be similar compositions with similar scores.

## Evaluation pipeline

The active learning loop runs the pipeline after each selection. The scripts can also be run on their own from `oxygen-carrier-discovery/heoc/`, on a run directory `<dir>` that holds `training.db`, `crystal_space.spa`, `prototypes.db` and the candidates of cycle *N* in `<dir>/cycle<N>/sample.csv`.

| Step | Script | Output |
|---|---|---|
| Composition space | `build_prototype_database.py`, `create_crystal_space.py` | `prototypes.db`, `crystal_space.spa` |
| Initial dataset | `build_training_database.py` | 1,000 random compositions that pass the screen (unseeded, so a rerun gives another set) |
| Candidates to structures | `build_sample_database.py --directory <dir> --random-seed <seed>` | `cycle<N>/sample.db` |
| SQS at 0–7 vacancies | `build_sqs_databases.py --directory <dir> --random-seed <seed> --max-cations 8` | `cycle<N>/sqs_<formula>.db` |
| CHGNet relaxation and phonons | `perform_phonon_calculations.py --directory <dir> --cycle <N> --use_gpus` | Phonopy results in the SQS databases |
| Formation energies | `update_phase_stability_sqs_dbs.py --directory <dir> --cycle <N>` | Materials Project formation energies (needs `PMG_MAPI_KEY`) |
| Gibbs energies and OTC | `update_training_database.py --directory <dir> --cycle <N>` | new rows in `training.db` |

The reactor conditions and the Gibbs energy expression are given in the Methods of the article.

## Relation to the code of the runs

The first commit of this repository is the code of the runs: `active-learning` at `7066f5b`, `metal` at `1e9639e` and `oxygen-carrier-discovery` at `4f362f7`, with the changes that were in use from the start of the runs. The later commits remove code that the runs did not use (other interatomic potentials, perovskite settings, cluster expansions, plotting and job-scheduler helpers) and add the following:

- `get_computed_entry` in `metal` takes the POTCAR labels for the Materials Project corrections from the POTCAR symbols instead of reading the VASP POTCAR files, so a VASP license is not needed. The corrections are identical, as checked on 440 evaluated structures covering all 18 elements.
- The runs spread the phonon calculations of a cycle over several GPUs with SLURM. Here they run on one GPU, which gives the same calculations.
- The ceiling-only reference (`--model ceiling`) is ported from `active-learning` commit `82169fc`, which ran it, and the held-out test from the script that produced the results in the Supplementary Information.
- `create_crystal_space.py` writes the crystal space of the runs, and reproduces the deposited `crystal_space.spa`.

The cleaned code selects the same cycle-1 candidates as the code of the runs for all ten strategies and the ceiling-only reference (seed 1, compared on the same machine).

## Credits

`metal` and the oxygen carrier pipeline were developed by Joakim Brorsson for high-entropy perovskite oxygen carriers ([Brorsson et al., ACS Materials Au 2026](https://doi.org/10.1021/acsmaterialsau.5c00230), [Materials Today Energy 2026](https://doi.org/10.1016/j.mtener.2026.102239)). Joel Hildingsson developed the active learning framework of that work, with a Wasserstein autoencoder adapted from the code of [Rao, Tung et al., Science 2022](https://github.com/ziyuanrao11/Machine-learning-enabled-high-entropy-alloy-discovery). Alfred Juhlin Onbeck extended the code to spinels and developed the surrogates, prediction targets and analysis of this study.

`metal/metal/applications/crystals/data/` contains published reference data: Shannon ionic radii, electronegativities, NIST-JANAF thermochemical tables for O<sub>2</sub>, and element abundances and prices.

## License and citation

MIT (see `LICENSE`). Please cite the article and the Zenodo record (see `CITATION.cff`).
