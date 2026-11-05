"""Locations shared by the analysis scripts.

The scripts read and write the files of analysis.zip next to this folder. HEOC holds the
run databases in their original layout, created by restore_original_layout.py, by
default in heoc/ next to analysis/. Set the environment variable HEOC to use another
location. SQS_VALIDATION is the unzipped sqs_validation.zip.
"""
import os
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
DATA = ANALYSIS / 'derived_data'
TABLES = ANALYSIS / 'tables'
FIGURES = ANALYSIS / 'figures'
HEOC = Path(os.environ.get('HEOC', ANALYSIS.parent / 'heoc'))
SQS_VALIDATION = ANALYSIS.parent / 'sqs_validation'
