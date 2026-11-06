"""Link the Zenodo data into the directory layout of the runs.

The analysis scripts and the active learning code read the runs from
oxygen-carrier-discovery/heoc/spinel_rev_<arm>_s<seed>/: training.db, the run record in
<timestamp>/ and the candidates and structures in cycle<N>/, where <arm> is the arm code
of the README. Unzip training_databases.zip, run_records.zip and, optionally,
initial_dataset.zip and the structures archives into data/, then run

    python3 analysis/scripts/restore_original_layout.py

from the repository root. This creates symbolic links only. Optional arguments give the
data directory and the target, which then has to be passed to the scripts through the
environment variable HEOC.
"""
import json
import sys
from pathlib import Path

ARM = {
    "de_scalar": "de",
    "de_clipped": "de_clip",
    "de_gibbs_vector": "de_land",
    "de_vacancy_input": "dde",
    "gp_scalar": "gp",
    "gp_clipped": "gp_clip",
    "gp_gibbs_vector": "mogp",
    "gp_vacancy_input": "dgp",
    "mc_scalar": "mc",
    "mc_clipped": "mc_clip",
    "ceiling_only": "ceil"
}


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src.resolve())


def main(src, dst):
    src, dst = Path(src), Path(dst)
    link(src / 'training_databases/initial_and_random_reference.db',
         dst / 'spinel_Li_random/training.db')
    # the initial dataset alone, read by the held-out test
    if (src / 'initial_dataset/training.db').exists():
        link(src / 'initial_dataset/training.db', dst / 'spinel_base_rev/training.db')
    for name, arm in ARM.items():
        for run in range(1, 6):
            origin = json.loads((src / f'run_records/{name}/run{run}/origin.json').read_text())
            run_dir = dst / origin['original_directory']
            link(src / f'training_databases/{name}/run{run}.db', run_dir / 'training.db')
            records = src / f'run_records/{name}/run{run}'
            for f in records.iterdir():
                if f.is_file() and f.name != 'origin.json':
                    link(f, run_dir / origin['run_record_directory'] / f.name)
            for f in (records / 'candidates').iterdir():
                cycle, fname = f.name.split('_', 1)
                link(f, run_dir / cycle / fname)
            structures = src / f'structures/{name}/run{run}'
            if structures.exists():
                for c in structures.iterdir():
                    for f in c.iterdir():
                        link(f, run_dir / c.name / f.name)


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[2]
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / 'data'
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else repo / 'oxygen-carrier-discovery' / 'heoc'
    main(source, target)
