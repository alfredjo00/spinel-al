"""Locations shared by the analysis scripts.

The scripts write their outputs to analysis/derived_data, analysis/tables and
analysis/figures, and later scripts read earlier outputs from there. The run databases
and run records are read from HEOC, by default oxygen-carrier-discovery/heoc, where
restore_original_layout.py links them in the directory layout of the runs. Set the
environment variable HEOC to use another location. SQS_VALIDATION is the unzipped
sqs_validation.zip of the Zenodo record, by default data/sqs_validation.
"""
import os
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
DATA = ANALYSIS / 'derived_data'
TABLES = ANALYSIS / 'tables'
FIGURES = ANALYSIS / 'figures'
REPO = ANALYSIS.parent
HEOC = Path(os.environ.get('HEOC', REPO / 'oxygen-carrier-discovery' / 'heoc'))
SQS_VALIDATION = Path(os.environ.get('SQS_VALIDATION', REPO / 'data' / 'sqs_validation'))

for _directory in (DATA, TABLES, FIGURES):
    _directory.mkdir(parents=True, exist_ok=True)
