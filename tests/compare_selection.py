"""Compare the candidates selected in cycle 1 of a rerun with those of the article run.

    python3 tests/compare_selection.py <arm> <seed>

Reads oxygen-carrier-discovery/heoc/spinel_rev_<arm>_s<seed>/cycle1/sample.csv and
tests/reference_cycle1/<arm>_s<seed>.csv. Compositions are compared as cation counts
in the 84-atom cell. Uses only the Python standard library.
"""
import csv
import sys
from pathlib import Path

N_ATOMS = 84


def read_selection(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    selection = []
    for row in rows:
        counts = tuple(sorted(
            (element, round(float(x) * N_ATOMS))
            for element, x in row.items()
            if element not in ('otc', 'O') and round(float(x) * N_ATOMS) > 0))
        selection.append((counts, float(row['otc'])))
    return selection


def formula(counts):
    return ''.join(f'{element}{n}' for element, n in counts)


def main():
    arm, seed = sys.argv[1], sys.argv[2]
    root = Path(__file__).resolve().parent.parent
    rerun = read_selection(root / 'oxygen-carrier-discovery/heoc'
                           / f'spinel_rev_{arm}_s{seed}/cycle1/sample.csv')
    reference = read_selection(root / f'tests/reference_cycle1/{arm}_s{seed}.csv')

    rerun_set = {c for c, _ in rerun}
    reference_set = {c for c, _ in reference}
    shared = rerun_set & reference_set
    print(f'{arm} seed {seed}: {len(shared)} of {len(reference_set)} '
          f'article selections reproduced')
    print(f'{"article":<40} {"score":>7}   {"rerun":<40} {"score":>7}')
    for (c_ref, s_ref), (c_new, s_new) in zip(reference, rerun):
        mark = ' ' if c_ref == c_new else '*'
        print(f'{formula(c_ref):<40} {s_ref:7.3f} {mark} {formula(c_new):<40} {s_new:7.3f}')
    sys.exit(0 if shared == reference_set else 1)


if __name__ == '__main__':
    main()
