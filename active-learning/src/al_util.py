"""
Utility functions for active learning pipeline.
Handles checkpointing, progress tracking, logging, and data loading.
"""

import json
import logging
import os
import random
from glob import glob
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader

from ANN_functions_new import FeatureDataset
from al_functions import create_df, normalize_and_scale, normalize_and_scale_GP


# Stage constants for progress tracking
STAGE_TRAINING_AND_SAMPLING = "training_and_sampling"
STAGE_SAMPLE_DB = "sample_db"
STAGE_SQS = "sqs"
STAGE_PHONON = "phonon"
STAGE_PHASE_STABILITY = "phase_stability"
STAGE_TRAINING_DB = "training_db"
STAGE_COMPARE_PREDICTIONS = "compare_predictions"

# Ordered list of stages for validation
STAGE_ORDER = [
    STAGE_TRAINING_AND_SAMPLING,
    STAGE_SAMPLE_DB,
    STAGE_SQS,
    STAGE_PHONON,
    STAGE_PHASE_STABILITY,
    STAGE_TRAINING_DB,
    STAGE_COMPARE_PREDICTIONS
]


def save_rng_state(cycle, checkpoint_dir):
    """Snapshot every RNG stream at the end of a completed cycle.

    Without this a resume calls same_seeds() and restarts the stream from the
    seed, so cycle N of a resumed run draws different random numbers than cycle
    N of a straight-through run -- different WAE shuffle, different GMM
    candidate pool, different Thompson draws. The run stays valid but it is no
    longer the same realisation, which breaks matched-seed replicates and makes
    any job that gets preempted or hits walltime silently incomparable to one
    that did not.

    Saved at cycle end (not at the WAE checkpoint) because that is the point a
    resume actually restarts from: initial_cycle = latest_cycle + 1.

    Parameters
    ----------
    cycle : int
        The cycle that just finished.
    checkpoint_dir : str
        Same directory used for the WAE checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'cycle': cycle,
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()

    path = os.path.join(checkpoint_dir, f'rng_state_cycle_{cycle}.pt')
    torch.save(state, path)
    logging.info(f'RNG state saved to {path}')


def load_rng_state(cycle, checkpoint_dir, logger=None):
    """Restore the RNG streams saved at the end of `cycle`.

    Must be called *after* same_seeds(), which it deliberately overrides.

    Returns
    -------
    bool
        True if a state file existed and was restored.
    """
    path = os.path.join(checkpoint_dir, f'rng_state_cycle_{cycle}.pt')
    log = logger.info if logger else logging.info
    if not os.path.exists(path):
        (logger.warning if logger else logging.warning)(
            f'No RNG state for cycle {cycle} ({path}). Falling back to the '
            f'seed, so this resume is a valid but different realisation than '
            f'an uninterrupted run.')
        return False

    state = torch.load(path, map_location='cpu', weights_only=False)
    torch.set_rng_state(state['torch'].to(torch.uint8).cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if 'cuda' in state and torch.cuda.is_available():
        saved, have = len(state['cuda']), torch.cuda.device_count()
        if saved == have:
            torch.cuda.set_rng_state_all([s.to(torch.uint8).cpu() for s in state['cuda']])
        else:
            (logger.warning if logger else logging.warning)(
                f'Checkpoint holds {saved} CUDA RNG states but this node has '
                f'{have} devices; skipping CUDA RNG restore.')
    log(f'RNG state restored from end of cycle {cycle}')
    return True


def save_checkpoint(model, cycle, checkpoint_dir, optimizer=None, iteration=None):
    """
    Save model checkpoint including WAE state and optimizer.

    Parameters
    ----------
    model : torch.nn.Module
        The WAE model to save
    cycle : int
        Current AL cycle number (used for checkpoint filename)
    checkpoint_dir : str
        Directory to save checkpoints
    optimizer : torch.optim.Optimizer, optional
        Optimizer state to save
    iteration : int, optional
        Deprecated, ignored. Kept for backwards compatibility.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_cycle_{cycle}.pt')

    checkpoint = {
        'cycle': cycle,
        'model_state_dict': model.state_dict(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, checkpoint_path)
    logging.info(f'Checkpoint saved to {checkpoint_path}')


def load_checkpoint(checkpoint_path, model, device, optimizer=None, legacy=False):
    """
    Load model checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file
    model : torch.nn.Module
        The WAE model to load state into
    legacy : bool
        If True, this is a legacy checkpoint_iter_*.pt file
    device : torch.device
        Device to load tensors to
    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into

    Returns
    -------
    int
        The cycle number from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load single model
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Support both new (cycle) and legacy (iteration) checkpoint formats
    cycle = checkpoint.get('cycle', checkpoint.get('iteration'))
    logging.info(f'Checkpoint loaded from {checkpoint_path}, cycle {cycle}')
    return cycle


def save_progress_state(log_path, cycle, stage):
    """
    Save the current progress state for a cycle.

    Parameters
    ----------
    log_path : str
        Path to log directory
    cycle : int
        Current cycle number
    stage : str
        Current stage constant (e.g., STAGE_TRAINING_AND_SAMPLING)
    """
    os.makedirs(log_path, exist_ok=True)
    progress_file = os.path.join(log_path, 'progress.json')

    progress_data = {
        'cycle': cycle,
        'stage': stage,
        'stage_index': STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1
    }

    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

    logging.info(f'Progress saved: cycle {cycle}, stage {stage}')


def load_progress_state(log_path, cycle):
    """
    Load the progress state for a cycle.

    Parameters
    ----------
    log_path : str
        Path to log directory
    cycle : int
        Cycle number to load progress for

    Returns
    -------
    str or None
        The stage constant, or None if no progress file exists or cycle doesn't match
    """
    progress_file = os.path.join(log_path, 'progress.json')

    if not os.path.exists(progress_file):
        return None

    with open(progress_file, 'r') as f:
        progress_data = json.load(f)

    # Only return the stage if the cycle matches
    if progress_data.get('cycle') == cycle:
        return progress_data.get('stage')
    else:
        return None


def should_skip_stage(stage, current_progress):
    """
    Determine if a stage should be skipped based on current progress.

    Parameters
    ----------
    stage : str
        Stage to check (e.g., STAGE_SQS)
    current_progress : str or None
        Current progress state

    Returns
    -------
    bool
        True if the stage should be skipped (already completed)
    """
    if current_progress is None:
        return False

    if stage not in STAGE_ORDER or current_progress not in STAGE_ORDER:
        return False

    stage_idx = STAGE_ORDER.index(stage)
    progress_idx = STAGE_ORDER.index(current_progress)

    # Skip if current progress is at or past this stage
    return progress_idx >= stage_idx


def setup_logging(log_path):
    """
    Set up logging configuration.

    Parameters
    ----------
    log_path : str
        Path to directory where logs should be saved

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, 'active_learning.log')

    # Create logger
    logger = logging.getLogger('active_learning')
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Route warnings.warn() into the same log. Libraries called here report
    # recoverable trouble this way - notably Materials Project rate limiting
    # during the phase stability stage - and by default that goes to stderr
    # only, so it never appears in the log and the retries stay invisible.
    # captureWarnings targets the 'py.warnings' logger, which propagates to
    # root; root has no handlers here, so attach them explicitly.
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.handlers.clear()
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.addHandler(file_handler)
    warnings_logger.addHandler(console_handler)
    logging.captureWarnings(True)

    logger.info(f'Logging initialized. Log file: {log_file}')
    return logger




def validate_cycle_integrity(folder_name, logger):
    """
    Validate that cycle folders are sequential without gaps.

    Parameters
    ----------
    folder_name : str
        Base folder name pattern (e.g., '/path/to/cycle')
    logger : logging.Logger
        Logger instance
    """
    cycle_folders = sorted(glob(f'{folder_name}*'))

    if not cycle_folders:
        logger.info('No existing cycles found. Starting fresh.')
        return

    cycle_numbers = sorted([int(folder.replace(folder_name, ''))
                           for folder in cycle_folders])

    # Check for gaps
    expected = list(range(cycle_numbers[0], cycle_numbers[-1] + 1))
    if cycle_numbers != expected:
        missing = set(expected) - set(cycle_numbers)
        logger.warning(f'Found gaps in cycle sequence. Missing cycles: {sorted(missing)}')
    else:
        logger.info(f'Found {len(cycle_numbers)} sequential cycles: {cycle_numbers}')


def auto_detect_and_load_checkpoint(log_path, checkpoint_dir, model, device, logger, folder_name):
    """
    Auto-detect if a checkpoint exists and load it, determining the starting cycle.

    Parameters
    ----------
    log_path : str
        Path to log directory
    checkpoint_dir : str
        Directory containing checkpoints
    model : torch.nn.Module
        WAE model to load state into
    device : torch.device
        Device for computation
    logger : logging.Logger
        Logger instance
    folder_name : str
        Base folder name for cycles (e.g., 'cycle')

    Returns
    -------
    tuple
        (start_cycle, initial_cycle, initial_progress_state)
        start_cycle is used as the loop start index (same as initial_cycle).
    """
    initial_cycle = 1
    initial_progress_state = None

    # Check for existing checkpoint (support both new and legacy filenames)
    if os.path.exists(checkpoint_dir):
        checkpoint_files = (
            glob(os.path.join(checkpoint_dir, 'checkpoint_cycle_*.pt')) +
            glob(os.path.join(checkpoint_dir, 'checkpoint_iter_*.pt'))
        )

        if checkpoint_files:
            # Extract number from filename for sorting
            def _extract_num(path):
                base = os.path.basename(path)
                # Works for both checkpoint_cycle_N.pt and checkpoint_iter_N.pt
                return int(base.split('.pt')[0].rsplit('_', 1)[1])

            checkpoint_files = sorted(checkpoint_files, key=_extract_num)
            latest_checkpoint = checkpoint_files[-1]
            logger.info(f'Found checkpoint: {latest_checkpoint}')

            load_checkpoint(latest_checkpoint, model, device)
        else:
            logger.info('No checkpoints found. Starting fresh.')
    else:
        logger.info('No checkpoint directory found. Starting fresh.')

    # Determine initial cycle and progress state from cycle folders
    cycle_folders = glob(f'{folder_name}*')

    if len(cycle_folders) <= 1:
        initial_cycle = 1
        logger.info('No active learning cycles found. Starting with cycle 1.')
    else:
        latest_cycle = sorted([int(folder.replace(folder_name, ''))
                              for folder in cycle_folders])[-1]

        progress_state_local = load_progress_state(log_path, latest_cycle)

        if progress_state_local is None:
            # Check if a checkpoint exists for this cycle — if so, WAE was
            # trained but the stage never completed (crash during sampling).
            cycle_ckpt = os.path.join(checkpoint_dir,
                                      f'checkpoint_cycle_{latest_cycle}.pt')
            if os.path.exists(cycle_ckpt):
                initial_cycle = latest_cycle
                logger.info(f'No progress for cycle {latest_cycle} but '
                            f'checkpoint exists — resuming from start of stage')
            else:
                initial_cycle = latest_cycle + 1
                logger.info(f'No progress found for cycle {latest_cycle}. Starting new cycle {initial_cycle}')
        elif progress_state_local == STAGE_COMPARE_PREDICTIONS:
            initial_cycle = latest_cycle + 1
            logger.info(f'Cycle {latest_cycle} complete. Starting new cycle {initial_cycle}')
        else:
            initial_cycle = latest_cycle
            initial_progress_state = progress_state_local
            logger.info(f'Resuming cycle {initial_cycle} from stage: {progress_state_local}')

    return initial_cycle, initial_cycle, initial_progress_state

def log_model_info(model, params, logger, is_ensemble=False):
    """
    Log detailed model architecture and parameter information.

    Parameters
    ----------
    model : torch.nn.Module
        The model to log information about
    params : dict
        Training parameters dictionary
    logger : logging.Logger
        Logger instance
    is_ensemble : bool
        Whether this is part of an ensemble (affects messaging)
    """
    logger.info('='*80)
    logger.info('MODEL CONFIGURATION')
    logger.info('='*80)
    logger.info('Training parameters:')
    for key, value in params.items():
        logger.info(f'  {key}: {value}')

    logger.info('\nModel architecture (first ensemble member):' if is_ensemble else '\nModel architecture:')
    logger.info('-'*80)
    logger.info(str(model))

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('-'*80)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    logger.info('='*80)


def log_wae_loss(wae_loss_history, cycle, log_path):
    # Check if validation metrics are present (8 values vs 4)
    has_val = len(wae_loss_history[-1]) == 8
    if has_val:
        last_loss = {
            'cycle': cycle,
            'avg_loss': wae_loss_history[-1][0], 'avg_recon': wae_loss_history[-1][1],
            'avg_MMD': wae_loss_history[-1][2], 'avg_clas': wae_loss_history[-1][3],
            'avg_val_loss': wae_loss_history[-1][4], 'avg_val_recon': wae_loss_history[-1][5],
            'avg_val_MMD': wae_loss_history[-1][6], 'avg_val_clas': wae_loss_history[-1][7]
        }
    else:
        last_loss = {
            'cycle': cycle,
            'avg_loss': wae_loss_history[-1][0], 'avg_recon': wae_loss_history[-1][1],
            'avg_MMD': wae_loss_history[-1][2], 'avg_clas': wae_loss_history[-1][3]
        }

    df_loss = pd.DataFrame([last_loss])
    os.makedirs(log_path, exist_ok=True)
    if not os.path.exists(f'{log_path}/wae_loss.csv'):
        df_loss.to_csv(f'{log_path}/wae_loss.csv', index=False)
    else:
        df_loss.to_csv(f'{log_path}/wae_loss.csv', mode='a', header=False, index=False)
