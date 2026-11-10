#!/bin/bash
# Runs the experiments of the paper. Every command is resumable: running it again only does what is missing.
#   ./run_all.sh fig3            # or fig5, delta, light, all
#   JOBS=32 ./run_all.sh fig5    # number of parallel runs (default: all cores)
# Never run the same (instance, algorithm) twice at the same time, both would write the same result file.

set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
JOBS=${JOBS:-$($PYTHON -c "import os; print(os.cpu_count())")}
ALGORITHMS="STS-LB ASTS SGTS MuSTS STS-Plain"
mkdir -p logs

run_simulations () {  # instances, runs, extra arguments, log name, algorithms (default $ALGORITHMS)
    echo "$(date '+%F %T') instances $1"
    $PYTHON -u simulate.py --algorithm ${5:-$ALGORITHMS} --instance_index $1 --runs $2 --jobs $JOBS $3 >> logs/sim_$4.log 2>&1
}

wants () {  # is $1 one of the targets given on the command line
    for target in $TARGETS; do
        if [ "$target" = "$1" ] || [ "$target" = "all" ]; then return 0; fi
    done
    return 1
}
TARGETS="$*"

if wants light; then
    # a few minutes each, also fine on a laptop
    $PYTHON -u bound_tightness.py --jobs $JOBS > logs/bound_tightness.log 2>&1
    $PYTHON -u glr_trace.py --instance_index 1 --log_period 20 --jobs $JOBS > logs/glr_trace_1.log 2>&1
    $PYTHON -u glr_trace.py --instance_index 2 --log_period 50 --jobs $JOBS > logs/glr_trace_2.log 2>&1
    $PYTHON -u glr_trace.py --instance_index 3 --log_period 50 --reference_starts 100 --jobs $JOBS > logs/glr_trace_3.log 2>&1
    $PYTHON -u glr_trace.py --instance_index 9 --log_period 20 --jobs $JOBS > logs/glr_trace_9.log 2>&1
    $PYTHON -u heatmaps.py --instance_index 6 --div 61 --jobs $JOBS > logs/heatmaps_6.log 2>&1
fi
if wants fig3; then
    run_simulations "4 5" 100 "" fig3 "$ALGORITHMS STS-LB-Naive"
fi
if wants delta; then
    run_simulations "7 8" 200 "--delta 0.1 --out_dir results/delta_correctness/" delta
fi
if wants fig5; then
    run_simulations "32 33 34 62 63 64 92 93 94 122 123 124" 20 "" fig5 "$ALGORITHMS STS-LB-Naive"
fi
$PYTHON plot.py all > logs/plot.log 2>&1
echo "$(date '+%F %T') done"
