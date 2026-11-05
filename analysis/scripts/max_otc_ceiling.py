"""Largest OTC ceiling C(x) allowed by the composition screen (Supplementary Section S6).

C(x) = 7 M_O / M_cell(x) in wt.%, with M_cell the mass of the vacancy-free 84-atom cell
(12 A + 24 B + 48 O). Compositions are visited in order of increasing cell mass, starting
from the lightest cell and moving one atom at a time to the next heavier element of its
sublattice, so the first ones that pass the screen have the highest ceilings. The screen
is that of the runs: at most MAX_CATIONS cation species, 0.77 <= tau <= 1.06 and charge
neutrality, both from SpinelTolerance of the metal package. For comparison, the script
also reports the highest C(x) in the initial dataset and among all acquisitions.

Needs the metal package of code.zip and its dependencies (for example in the container
of code.zip), with code/metal on PYTHONPATH. Reads the run databases. Its output is
max_otc_ceiling.txt.
"""
from contextlib import redirect_stdout
import heapq
import io
import os
import sys

from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.db import connect
with redirect_stdout(io.StringIO()):  # metal reports optional packages on import
    from metal.applications.crystals.stability_factors import SpinelTolerance

from paths import HEOC

TOP = int(os.environ.get('TOP', '10'))
MAX_POPS = int(os.environ.get('MAX_POPS', '2000000'))
MAX_CATIONS = 8
N_VAC_MAX = 8                  # vacancy counts 0 to 7
N_A, N_B, N_O = 12, 24, 48     # 84-atom cell
A_EL = ['Mg', 'Zn', 'Cu', 'Co', 'Ni', 'Ca', 'Li']
B_EL = ['Al', 'Cr', 'Ga', 'Ti', 'Fe', 'Mn', 'Zr', 'V', 'Sn', 'In']
ARMS = ['gp', 'gp_clip', 'mogp', 'dgp', 'de', 'de_clip', 'dde', 'de_land', 'mc', 'mc_clip']
SEEDS = range(1, 6)

mass = {el: atomic_masses[atomic_numbers[el]] for el in A_EL + B_EL + ['O']}
A_EL.sort(key=mass.get)
B_EL.sort(key=mass.get)


def ceiling(cation_mass):
    """C(x) in wt% from the summed cation mass of the 84-atom cell."""
    return 100 * (N_VAC_MAX - 1) * mass['O'] / (cation_mass + N_O * mass['O'])


def passes(a, b):
    if sum(n > 0 for n in a + b) > MAX_CATIONS:
        return None
    symbols, tags = [], []
    for counts, els, tag in ((a, A_EL, 'a'), (b, B_EL, 'b')):
        for n, el in zip(counts, els):
            symbols += [el] * n
            tags += [tag] * n
    symbols += ['O'] * N_O
    tags += ['x'] * N_O
    try:
        with redirect_stdout(io.StringIO()):  # SpinelTolerance prints its radii
            st = SpinelTolerance(Atoms(symbols), possible_a_elements=A_EL,
                                 possible_b_elements=B_EL, site_tags=tags)
            ok = 0.77 <= st.spinel_tolerance <= 1.06 and st.charge_neutrality
            return st.spinel_tolerance if ok else None
    except Exception:
        return None


def cation_mass(a, b):
    return (sum(n * mass[el] for n, el in zip(a, A_EL))
            + sum(n * mass[el] for n, el in zip(b, B_EL)))


def label(a, b):
    return ' '.join(f'{el}{n}' for n, el in zip(a + b, A_EL + B_EL) if n)


def moves(counts):
    """One atom to the next heavier element: every composition is reachable
    from the all-lightest one and each move strictly raises the mass."""
    for i in range(len(counts) - 1):
        if counts[i]:
            c = list(counts)
            c[i] -= 1
            c[i + 1] += 1
            yield tuple(c)


# ------------------------------------------------------ best-first search
start = ((N_A,) + (0,) * (len(A_EL) - 1), (N_B,) + (0,) * (len(B_EL) - 1))
heap = [(cation_mass(*start), start)]
seen = {start}
found, pops = [], 0
while heap and len(found) < TOP and pops < MAX_POPS:
    m, (a, b) = heapq.heappop(heap)
    pops += 1
    tau = passes(a, b)
    if tau is not None:
        found.append((ceiling(m), tau, label(a, b)))
    for nxt in [(x, b) for x in moves(a)] + [(a, x) for x in moves(b)]:
        if nxt not in seen:
            seen.add(nxt)
            heapq.heappush(heap, (cation_mass(*nxt), nxt))

print(f'visited {pops} compositions in order of cell mass')
if len(found) < TOP:
    print(f'WARNING: stopped after {pops} pops with only {len(found)} found (MAX_POPS)')
print(f'lightest-cell (unscreened) C(x) = {ceiling(cation_mass(*start)):.3f} wt%  [{label(*start)}]')
print(f'\nTop {len(found)} C(x) passing the screen (<= {MAX_CATIONS} cations, tau, neutrality):')
for c, tau, lab in found:
    print(f'  {c:7.3f} wt%   tau={tau:.3f}   {lab}')


# ------------------------------------------- initial dataset and acquisitions
def db_ceilings(path, selection=None):
    out = {}
    for row in connect(path).select(selection):
        sym = row.symbols
        key = tuple(sorted((el, sym.count(el)) for el in set(sym) if el != 'O'))
        if sum(n for _, n in key) != N_A + N_B:
            print(f'{path} row {row.id}: not an 84-atom cell, skipped', file=sys.stderr)
            continue
        out[key] = ceiling(sum(mass[el] * n for el, n in key))
    return out


base = db_ceilings(f'{HEOC}/spinel_rev_de_s1/training.db', 'cycle=0')
k = max(base, key=base.get)
print(f'\ninitial dataset ({len(base)} compositions): max C(x) = {base[k]:.3f} wt%  {k}')

acq = {}
for arm in ARMS:
    for s in SEEDS:
        path = f'{HEOC}/spinel_rev_{arm}_s{s}/training.db'
        if not os.path.exists(path):
            print(f'missing {path}', file=sys.stderr)
            continue
        for key, c in db_ceilings(path).items():
            if key not in base:
                acq[key] = c
k = max(acq, key=acq.get)
print(f'acquisitions, {len(ARMS)} strategies x {len(SEEDS)} runs '
      f'({len(acq)} unique compositions): max C(x) = {acq[k]:.3f} wt%  {k}')
