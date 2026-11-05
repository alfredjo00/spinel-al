"""Load the spinel training database into arrays for multi-output GP experiments.

Self-contained: depends only on ase + numpy, no imports from active-learning/,
metal/ or oxygen-carrier-discovery/. Mirrors the featurisation performed by
`al_functions.create_df` + `DataManager.load_data` in the main pipeline.
"""
import numpy as np
from ase.db import connect

# Sublattice definitions (al_config.py:120-132). A occupies 1/7 of the 84-atom
# supercell (12 sites), B occupies 2/7 (24 sites), O occupies 4/7 (48 sites).
A_ELEMENTS = ['Mg', 'Zn', 'Cu', 'Co', 'Ni', 'Ca', 'Li']
B_ELEMENTS = ['Al', 'Cr', 'Ga', 'Ti', 'Fe', 'Mn', 'Zr', 'V', 'Sn', 'In']
ELEMENTS = A_ELEMENTS + B_ELEMENTS
A_SCALING, B_SCALING = 1.0 / 7.0, 2.0 / 7.0

N_VACANCIES_MAX = 8
REACTORS = ('AR', 'FR')
GIBBS_KEYS = [f'gibbs_{r}_{v}' for r in REACTORS for v in range(N_VACANCIES_MAX)]

MASS_O = 15.999


class Dataset:
    """Container for one loaded database.

    Attributes
    ----------
    X : np.ndarray [N, 17]
        Per-sublattice cation fractions (A block sums to 1, B block sums to 1).
    Y : np.ndarray [N, 16]
        Gibbs formation energies, ordered [AR_0..AR_7, FR_0..FR_7] (eV).
    mass : np.ndarray [N]
        Supercell mass `mass_` (g/mol), the OTC denominator.
    otc_ref, dv_ref : np.ndarray [N]
        Stored `otc_quadratic_fit` and `vacancy_diff_quadratic_fit`.
    cycle : np.ndarray [N]
        Active-learning cycle each row was acquired in.
    """

    def __init__(self, X, Y, mass, otc_ref, dv_ref, cycle):
        self.X, self.Y = X, Y
        self.mass, self.otc_ref, self.dv_ref, self.cycle = mass, otc_ref, dv_ref, cycle

    def __len__(self):
        return len(self.X)

    def active_tasks(self, tol=1e-9):
        """Indices of Gibbs columns that actually vary.

        `gibbs_AR_0` and `gibbs_FR_0` are identically zero by construction (the
        v=0 reference), which makes the coregionalization matrix singular and
        drives the per-task noise onto its constraint boundary. They are dropped
        from the model and reinserted analytically by `restore_landscape`.
        """
        return np.flatnonzero(self.Y.std(axis=0) > tol)

    def restore_landscape(self, Y_active, keep_idx):
        """Reinsert the structurally-zero columns to rebuild the full 16-D landscape."""
        full = np.zeros((*Y_active.shape[:-1], self.Y.shape[1]), dtype=Y_active.dtype)
        full[..., keep_idx] = Y_active
        return full


def load_dataset(db_path, selection='filtered=False'):
    """Read an ASE database into a `Dataset`.

    Rows missing any of the 16 Gibbs energies, `mass_`, or the reference OTC are
    skipped. Compositions come from `row.count_atoms()` divided by the total
    atom count and then by the sublattice scaling, exactly as the main pipeline
    does, so a model trained here sees the same inputs it would in an AL run.
    """
    db = connect(db_path)
    required = GIBBS_KEYS + ['mass_', 'otc_quadratic_fit', 'vacancy_diff_quadratic_fit']

    el_index = {el: i for i, el in enumerate(ELEMENTS)}
    scale = np.array([A_SCALING] * len(A_ELEMENTS) + [B_SCALING] * len(B_ELEMENTS))

    X, Y, mass, otc_ref, dv_ref, cycle = [], [], [], [], [], []
    for row in db.select(selection):
        kv = row.key_value_pairs
        if not all(k in kv for k in required):
            continue

        counts = row.count_atoms()
        total = sum(counts.values())
        x = np.zeros(len(ELEMENTS))
        for el, n in counts.items():
            if el in el_index:
                x[el_index[el]] = n / total
        X.append(x / scale)

        Y.append([kv[k] for k in GIBBS_KEYS])
        mass.append(kv['mass_'])
        otc_ref.append(kv['otc_quadratic_fit'])
        dv_ref.append(kv['vacancy_diff_quadratic_fit'])
        cycle.append(kv.get('cycle', -1))

    return Dataset(
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        np.asarray(mass, dtype=np.float64),
        np.asarray(otc_ref, dtype=np.float64),
        np.asarray(dv_ref, dtype=np.float64),
        np.asarray(cycle, dtype=np.int64),
    )


class GroupedZScore:
    """Single shared mean/std across all Gibbs columns.

    Reproduces `DataManager._normalize_landscape_mode`: one scalar shift and one
    scalar scale for the whole landscape, so the relative offsets between
    vacancy levels (which is what the quadratic fit reads) are preserved. The
    per-task constant mean in the GP absorbs the remaining offsets.
    """

    def __init__(self):
        self.mean_ = self.std_ = None

    def fit(self, Y):
        self.mean_ = float(Y.mean())
        self.std_ = float(Y.std())
        if self.std_ <= 1e-8:
            self.std_ = 1.0
        return self

    def transform(self, Y):
        return (Y - self.mean_) / self.std_

    def inverse_transform(self, Y):
        return Y * self.std_ + self.mean_

    def inverse_scale(self, sigma):
        """Map a standard deviation back to eV (no shift)."""
        return sigma * self.std_
