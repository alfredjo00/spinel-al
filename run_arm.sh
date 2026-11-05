#!/usr/bin/env bash
# Run one active learning strategy (arm) with the settings used in the article.
#
#   ./run_arm.sh <arm> <seed> [cycles]
#
# arm     de, de_clip, de_land, dde, gp, gp_clip, mogp, dgp, mc, mc_clip
# seed    1-5 reproduce runs 1-5 of the article
# cycles  acquisition cycles, default 20 as in the article
#
# Environment variables:
#   AL_RUNNER               apptainer (default) or docker
#   AL_IMAGE                Apptainer image, default container/pytorch_al_img.sif,
#                           or Docker image tag, default spinel-al
#   AL_STOP_AFTER_SAMPLING  if set, stop after the candidates of the first cycle
#                           are selected (no CHGNet/Phonopy, no Materials Project)
#   PMG_MAPI_KEY            Materials Project API key, needed for a full cycle
#
# The run directory is code/oxygen-carrier-discovery/heoc/spinel_rev_<arm>_s<seed>/. Running
# the same command again resumes an interrupted run.
set -euo pipefail

arm=${1:?arm}; seed=${2:?seed}; cycles=${3:-20}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code=$here/code
heoc=$code/oxygen-carrier-discovery/heoc

# Shared settings of all arms (Methods and SI). --cycles counts cycle 0.
common="--cycles $((cycles + 1)) -b 3 -t 10 --n-samples-total 25000
        --max-element-count 8 --use-validation --val-split 0.15
        --val-split-mode random --round-compositions --quadratic-fit-otc
        --bootstrap-fraction none"

case $arm in
    de)      flags="--model ensemble --ensemble-size 16" ;;
    de_clip) flags="--model ensemble --ensemble-size 16 --clip-otc-physical" ;;
    de_land) flags="--model ensemble --ensemble-size 16 --predict-landscape --quadratic-fit-landscape-otc" ;;
    dde)     flags="--model delta_de --ensemble-size 16 --delta-de-epochs 200 --delta-de-lr 1e-4 --predict-landscape --quadratic-fit-landscape-otc" ;;
    gp)      flags="--model gp" ;;
    gp_clip) flags="--model gp --clip-otc-physical" ;;
    mogp)    flags="--model mogp --ensemble-size 16 --mogp-rank 1 --predict-landscape --quadratic-fit-landscape-otc" ;;
    dgp)     flags="--model delta_gp --ensemble-size 16 --delta-gp-iters 1000 --delta-gp-patience 100 --predict-landscape --quadratic-fit-landscape-otc" ;;
    mc)      flags="--model mc" ;;
    mc_clip) flags="--model mc --clip-otc-physical" ;;
    *) echo "unknown arm: $arm" >&2; exit 1 ;;
esac

method=spinel_rev_${arm}_s${seed}
if [ ! -e "$heoc/$method" ]; then
    # cycle0/ marks the initial dataset as evaluated. Without it the pipeline
    # treats the run as cycle 0 and builds SQS structures for all of training.db.
    mkdir -p "$heoc/$method/cycle0"
    cp "$here"/data/initial_dataset/{training.db,crystal_space.spa,prototypes.db} "$heoc/$method/"
fi

cmd="python -u active_learning_HEO.py --method $method --auto-resume --seed $seed $flags $common"
envs=(AL_STOP_AFTER_SAMPLING="${AL_STOP_AFTER_SAMPLING:-}" PMG_MAPI_KEY="${PMG_MAPI_KEY:-}")

case ${AL_RUNNER:-apptainer} in
    apptainer)
        env "${envs[@]/#/APPTAINERENV_}" \
            apptainer exec --nv --bind "$here" --pwd "$code/active-learning/src" \
            "${AL_IMAGE:-$here/container/pytorch_al_img.sif}" $cmd ;;
    docker)
        docker run --rm --gpus all --user "$(id -u):$(id -g)" \
            -e HOME=/tmp -e "${envs[0]}" -e "${envs[1]}" \
            -v "$here:$here" -w "$code/active-learning/src" \
            "${AL_IMAGE:-spinel-al}" $cmd ;;
    *) echo "AL_RUNNER must be apptainer or docker" >&2; exit 1 ;;
esac
