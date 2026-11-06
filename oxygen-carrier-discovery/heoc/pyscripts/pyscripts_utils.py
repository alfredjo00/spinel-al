"""Shared utilities for the heoc pyscripts pipeline.

Centralises boilerplate that was duplicated across many scripts:
path resolution, metal import setup, cycle detection, crystal-space
loading, SLURM process count, and working-directory management.
"""

import os
import sys
from contextlib import contextmanager
from glob import glob

# ---------------------------------------------------------------------------
# Path constants (relative to *this* file, i.e. heoc/pyscripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)          # heoc/


# ---------------------------------------------------------------------------
# Metal import setup
# ---------------------------------------------------------------------------
def setup_metal_path():
    """Add the ``metal/`` package to ``sys.path`` so it can be imported.

    The metal directory is located three levels above the pyscripts folder
    (i.e. ``oxygen-carrier-discovery/../../metal``).
    """
    metal_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'metal'))
    if metal_path not in sys.path:
        sys.path.append(metal_path)


# ---------------------------------------------------------------------------
# Working-directory context manager
# ---------------------------------------------------------------------------
@contextmanager
def working_directory(path):
    """Temporarily change the working directory, restoring it on exit."""
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------
def get_latest_cycle(directory, base_dir=None):
    """Return the latest cycle number from ``cycle*`` folders.

    Parameters
    ----------
    directory : str
        Data directory name (e.g. ``'spinel'``).
    base_dir : str, optional
        Base directory containing *directory*.  Defaults to :data:`PARENT_DIR`.

    Returns
    -------
    int
    """
    if base_dir is None:
        base_dir = PARENT_DIR
    folders = glob(f'{base_dir}/{directory}/cycle*')
    return sorted(int(os.path.basename(f).replace('cycle', ''))
                  for f in folders)[-1]


# ---------------------------------------------------------------------------
# Crystal-space loading
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SLURM / process count
# ---------------------------------------------------------------------------
def get_process_count(args_processes=None):
    """Resolve the number of worker processes.

    Priority: *args_processes* > ``SLURM_NTASKS`` env var > 1.

    Parameters
    ----------
    args_processes : int or None
        Value from argparse (``--processes``).

    Returns
    -------
    int
    """
    if args_processes is not None:
        return args_processes
    return int(os.getenv('SLURM_NTASKS', 1))


# ---------------------------------------------------------------------------
# Proxy-element detection
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sublattice info convenience
# ---------------------------------------------------------------------------