"""SLURM fan-out for phonon calculations.

Submits parallel sbatch jobs for phonon calculations across multiple GPUs,
runs a chunk locally on the main GPU, and handles fallback for stuck jobs.
"""

import os
import re
import subprocess
import time

from perform_phonon_calculations import perform_phonon_calculations


def submit_phonon_slurm_jobs(config, cycle, logger):
    """Run phonon calculations on the main GPU and fan out to additional SLURM jobs.

    The work is split into (phonon_slurm_jobs + 1) chunks. The main process
    runs chunk 0 on its own GPU while the sbatch jobs handle chunks 1..N.
    After the main chunk finishes, it polls squeue until all sbatch jobs complete.

    Parameters
    ----------
    config : Config
        Configuration object with SLURM settings.
    cycle : int
        Current active learning cycle number.
    logger : logging.Logger
        Logger instance.
    """
    import glob as _glob
    import math as _math
    pyscripts_dir = os.path.join(config.project_root,
                                 'oxygen-carrier-discovery', 'heoc', 'pyscripts')
    phonon_script = os.path.join(pyscripts_dir, 'perform_phonon_calculations.py')
    src_dir = os.path.dirname(os.path.abspath(__file__))

    # Count actual files so we never spawn more jobs than files
    parent_dir = os.path.join(config.project_root,
                              'oxygen-carrier-discovery', 'heoc')
    n_files = len(_glob.glob(
        os.path.join(parent_dir, config.method, f'cycle{cycle}', 'sqs_*.db')))
    # main process handles 1 chunk; sbatch jobs get the rest (at most n_files - 1)
    n_sbatch_jobs = min(config.phonon_slurm_jobs, max(n_files - 1, 0))
    total_chunks = n_sbatch_jobs + 1

    # The worker splits with ceil(n_files / divide_by), so asking for more
    # chunks than that split actually fills leaves empty trailing chunks --
    # e.g. 8 files over 5 chunks gives size 2, covering only 4 chunks, and
    # chunk 4 slices [8:10] -> []. An empty chunk reaches the worker and dies
    # with 'No files have been provided'. Re-derive the count from the same
    # ceil arithmetic so every chunk we hand out is non-empty.
    if n_files:
        chunk_size = _math.ceil(n_files / total_chunks)
        total_chunks = _math.ceil(n_files / chunk_size)
        n_sbatch_jobs = total_chunks - 1
    logger.info(f'{n_files} db files found, using {total_chunks} chunks '
                f'({n_sbatch_jobs} sbatch + 1 local)')

    # Submit sbatch jobs for chunks 1..N
    job_ids = []
    for i in range(1, total_chunks):
        relax_flag = f' --n_relax_threads {config.relax_threads}' if config.relax_threads else ''
        batch_flag = f' --n_batch {config.slurm_phonon_batch}' if config.slurm_phonon_batch else ''
        python_cmd = (
            f'python -u {phonon_script}'
            f' --directory {config.method}'
            f' --cycle {cycle}'
            f' --divide_by {total_chunks}'
            f' --index {i}'
            f' --use_gpus'
            f'{relax_flag}'
            f'{batch_flag}'
        )

        if config.slurm_container:
            run_cmd = (
                f'apptainer exec --nv {config.slurm_container}'
                f' {python_cmd}'
            )
        else:
            run_cmd = f'{python_cmd}'

        job_name = f'phonon_c{cycle}_chunk{i}'
        log_file = os.path.join(config.log_path, f'phonon_c{cycle}_chunk{i}.log')

        # Keep phonon jobs on the cheap 40 GB nodes unless told otherwise; the
        # 80 GB 'fat' partition bills 2x and these chunks never need the memory.
        constraint = getattr(config, 'slurm_constraint', 'thin')
        constraint_line = f'#SBATCH -C {constraint}\n' if constraint else ''

        sbatch_script = f"""#!/bin/bash
#SBATCH -A {config.slurm_account}
#SBATCH --gpus=1
{constraint_line}#SBATCH -t {config.slurm_time}
#SBATCH -J {job_name}
#SBATCH -o {log_file}

cd {src_dir}

echo "Phonon job chunk {i}/{total_chunks} started: $(date)"
{run_cmd}
echo "Phonon job chunk {i}/{total_chunks} ended: $(date)"
"""

        proc = subprocess.run(
            ['sbatch'], input=sbatch_script, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'sbatch failed for chunk {i}: {proc.stderr}')

        match = re.search(r'Submitted batch job (\d+)', proc.stdout)
        if not match:
            raise RuntimeError(f'Could not parse job ID from: {proc.stdout}')
        job_id = match.group(1)
        job_ids.append(job_id)
        logger.info(f'Submitted phonon chunk {i}/{total_chunks} as SLURM job {job_id}')

    # Run chunk 0 locally on the main GPU (threaded relaxations run inside if configured)
    logger.info(f'Running phonon chunk 0/{total_chunks} locally on main GPU...')
    perform_phonon_calculations(
        processes=1, directory=f'{config.method}', use_gpus=True,
        logging=logger, batch_size=config.phonon_batch_size,
        divide_by=total_chunks, index=0,
        n_relax_threads=config.relax_threads or 0)
    logger.info(f'Local phonon chunk 0/{total_chunks} complete.')

    # Wait for sbatch jobs, with fallback for stuck PENDING jobs
    _wait_for_slurm_jobs(config, cycle, job_ids, total_chunks, logger)


def _wait_for_slurm_jobs(config, cycle, job_ids, total_chunks, logger):
    """Poll sbatch jobs; steal PENDING ones and run locally to keep main GPU busy."""
    job_chunk_map = {jid: chunk_i for jid, chunk_i in zip(job_ids, range(1, total_chunks))}
    stolen_chunks = []  # tracks which chunks were cancelled (for reporting / sacct)

    n_sbatch_jobs = len(job_ids)
    logger.info(f'Monitoring {n_sbatch_jobs} remote phonon SLURM jobs...')

    while job_chunk_map:
        remaining_ids = list(job_chunk_map.keys())
        proc = subprocess.run(
            ['squeue', '--jobs=' + ','.join(remaining_ids),
             '--noheader', '--format=%i %T'],
            capture_output=True, text=True)
        active_lines = [l for l in proc.stdout.strip().split('\n') if l.strip()]

        if not active_lines:
            break

        active_states = {l.split()[0]: l.split()[1] for l in active_lines}

        # Remove finished jobs (no longer in squeue) from tracking
        finished = [jid for jid in remaining_ids if jid not in active_states]
        for jid in finished:
            job_chunk_map.pop(jid, None)

        # Steal a PENDING job and run it locally — keeps the main GPU busy
        pending_jids = [jid for jid in remaining_ids
                        if jid in job_chunk_map and active_states.get(jid) == 'PENDING']
        if pending_jids:
            jid = pending_jids[0]
            chunk_i = job_chunk_map.pop(jid)
            subprocess.run(['scancel', jid], capture_output=True)
            logger.info(f'Stealing PENDING chunk {chunk_i}/{total_chunks} '
                        f'(job {jid}) — running locally.')
            stolen_chunks.append(chunk_i)
            try:
                perform_phonon_calculations(
                    processes=1, directory=f'{config.method}', use_gpus=True,
                    logging=logger, batch_size=config.phonon_batch_size,
                    divide_by=total_chunks, index=chunk_i,
                    n_relax_threads=config.relax_threads or 0)
            except Exception:
                # A stolen chunk runs in the main process, so its failure would
                # otherwise kill the whole run and orphan the sbatch jobs we
                # already queued. Cancel them before propagating.
                logger.error(f'Stolen chunk {chunk_i}/{total_chunks} failed — '
                             f'cancelling {len(job_chunk_map)} queued phonon job(s).')
                for orphan in job_chunk_map:
                    subprocess.run(['scancel', orphan], capture_output=True)
                raise
            logger.info(f'Local chunk {chunk_i}/{total_chunks} complete.')
            continue  # re-poll immediately, maybe steal another

        # All remaining jobs are RUNNING — just wait
        if job_chunk_map:
            running = sum(1 for st in active_states.values() if st == 'RUNNING')
            logger.info(f'  {running} running — '
                        + ', '.join(f'{jid}({active_states.get(jid, "done")})'
                                    for jid in job_chunk_map))
        time.sleep(config.slurm_poll_interval)

    # Check completed sbatch jobs for failures via sacct (skip stolen jobs)
    cancelled_ids = set()
    for jid, chunk_i in zip(job_ids, range(1, total_chunks)):
        if chunk_i in stolen_chunks:
            cancelled_ids.add(jid)
    sacct_ids = [jid for jid in job_ids if jid not in cancelled_ids]
    if sacct_ids:
        try:
            proc = subprocess.run(
                ['sacct', '-j', ','.join(sacct_ids), '--format=JobID,State,ExitCode',
                 '--noheader', '--parsable2'],
                capture_output=True, text=True)
            failed = []
            for line in proc.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split('|')
                if len(parts) >= 2:
                    job_id_field, state = parts[0], parts[1]
                    if '.' not in job_id_field and state not in ('COMPLETED', 'CANCELLED', ''):
                        failed.append(f'{job_id_field}: {state}')
            if failed:
                raise RuntimeError(
                    f'Phonon SLURM jobs failed:\n' + '\n'.join(failed) +
                    f'\nCheck logs in {config.log_path}/phonon_c{cycle}_chunk*.log')
        except FileNotFoundError:
            logger.warning('sacct not available — skipping post-job failure check. '
                           f'Verify logs manually: {config.log_path}/phonon_c{cycle}_chunk*.log')

    logger.info(f'All {total_chunks} phonon chunks completed successfully'
                + (f' ({len(stolen_chunks)} stolen from queue and ran locally).'
                   if stolen_chunks else '.'))
