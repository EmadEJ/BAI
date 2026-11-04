# Best Arm Identification with Unknown Post-Action Context Distribution

Code of the experiments of the paper. Every experiment stores its raw results in `results/`, and `plot.py` draws
every figure and table from them, so figures can be changed without rerunning anything.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## The code

| file | content |
|---|---|
| `lowerbound.py` | the closed-form lower bounds `L_bound` (Theorem 1 of the bound report) and `L_naive` (Proposition 6), the proven factors (Theorems 2-3), and the convex sampling rule `optimal_w` |
| `optimization.py` | the non-convex GLR: `plain_GLR` (local solver from a few starting points), `exact_GLR_k2` (closed form for k = 2), `reference_GLR` (exact for k = 2, 20 random starting points otherwise, our "true value"), and the w-optimization `optimize_bfgs` |
| `algorithms/` | `ASTS` (GTS-A), `MuSTS` (GTS-mu), `STS` (GTS-Unknown, `glr="lb"` or `glr="plain"`), `SGTS` (GTS-SG); `TS.py` has the tracking rules |
| `environment.py` | the bandit and the run loop (`Environment.run`) |
| `utils.py` | `C_exp` of [KK21] and `W_bar` of [TD25a] used in the thresholds |
| `simulate.py` | many runs of one algorithm on one instance (figures 3, 5 and the delta-correctness check) |
| `glr_trace.py` | figure 2 (instances 13 and 14, delta = 1e-6), and the same on instance 15 (k = 4) |
| `heatmaps.py` | figure 4 |
| `bound_tightness.py` | the table and figure comparing the bounds with the true GLR |
| `plot.py` | all figures and tables from the stored results |
| `run_all.sh` | every experiment of the paper |

Algorithm names used on the command line and in the result files:

| name | paper | stopping | sampling |
|---|---|---|---|
| `ASTS` | GTS-A | exact GLR (A known) | exact (convex) |
| `MuSTS` | GTS-mu | exact GLR (mu known) | Appendix F.1 |
| `STS-LB` | GTS-Unknown (LB) | `min_i L_i(t) > beta(t, delta)` | maximizer of `w -> min_i L_i(w, A w)` (convex) |
| `STS-Plain` | GTS-Unknown (solver) | plain solver of the GLR (an upper bound, not provably delta-correct); the closed form for k = 2 | the same GLR inside BFGS |
| `SGTS` | GTS-SG | sub-Gaussian GLR, threshold of [TD25a] | exact (convex) |

All algorithms recompute the optimal `w` only every `10^(d-2)` rounds once `t` has `d` digits (`--no_fast` turns this
off). `STS-Plain` also checks its (expensive) stopping rule only once every `t/100` rounds, which can only delay
stopping by at most 1%. `STS-LB` checks its closed-form stopping rule at every round.

## Running the experiments

`run_all.sh` has every experiment, and every command can be stopped and run again: `simulate.py` keeps one entry per
run (seed = run index) in its result file and only runs the missing seeds.

```bash
./run_all.sh light          # figures 2 and 4 and the bound tightness experiment, minutes on a laptop
JOBS=32 ./run_all.sh fig3   # figure 3, 100 runs per algorithm
JOBS=32 ./run_all.sh delta  # delta-correctness check, delta = 0.1, 200 runs per algorithm
JOBS=32 ./run_all.sh fig5   # figure 5, 20 runs per algorithm
python plot.py all          # figures/ and figures/tables.md
```

Single experiments:

```bash
python simulate.py --algorithm STS-LB --instance_index 11 --runs 100 --jobs 8
python simulate.py --algorithm ASTS SGTS --instance_index 32 33 --runs 20 --jobs 8
python simulate.py --algorithm ASTS --instance_index 21 --runs 200 --delta 0.1 --out_dir results/delta_correctness/ --jobs 8
python glr_trace.py --instance_index 14 --log_period 50 --jobs 8
python heatmaps.py --instance_index 10 --div 61 --jobs 8
python bound_tightness.py --jobs 8
python optimization.py -n 2 -k 3 --fixed_w --name plain   # checks the GLR solver against the grid search of testsets/
```

`simulate.py` also takes several algorithms and instances (`--algorithm STS-LB ASTS --instance_index 11 12`), all
their missing runs then share one pool, the slowest algorithm first. Do not run the same (instance, algorithm) in two
commands at the same time: both would write the same result file. To stop the runs, use Ctrl-C (or
`pkill -f simulate.py`; on macOS the workers are `multiprocessing.spawn` processes and need to be killed too).

### Figure 2 on another instance

1. Write `instances/instance16.json` (any unused number), with `"A"` given row by row: one row per arm, with the k
   probabilities of the contexts (the transpose of the paper's k x n matrix). All entries must be positive, the rows sum
   to 1, `"mus"` is in [0, 1] and the best arm is unique. `"confidence"` is delta.
2. Run the trace, then draw it (the paper figure `fig2_glr_trace` is not touched):

```bash
python glr_trace.py --instance_index 16 --log_period 20 --jobs 8
python plot.py fig2 16          # figures/fig2_glr_trace_16.pdf, or "fig2 16 17" for two panels
```

`glr_trace.py` prints when each statistic crosses each threshold. The run continues until both lower bounds crossed the
naive threshold, so a harder instance (smaller gap, smaller delta) runs longer; keep about 1000 to 2000 logged rounds
with `--log_period` (the run length divided by 1500). `--delta` overrides the delta of the instance, `--seed` changes
the sample path, and `--reference_starts` (20 by default) is the number of random starting points of the true GLR for
k >= 3 (for k = 2 it is the exact closed form). With 8 cores it takes about 1.5 min for instance 13 (k = 2), 1 min for
instance 14 (k = 3) and 5 min for instance 15 (k = 4, with `--reference_starts 100`).

### Expected runtimes

Measured on one core of a laptop (Apple M5, all cores busy), a server core can be up to 2x slower. `figures/tables.md`
has the measured time of every stored run.

| algorithm | time per run | |
|---|---|---|
| `ASTS`, `SGTS`, `STS-LB` | 1 s to 1.5 min | negligible |
| `MuSTS` | 0.5 to 8 min | |
| `STS-Plain`, k = 2 | seconds (closed-form GLR) | |
| `STS-Plain`, k >= 3 | 35 min (instance 33) to about 4 h (instance 124) | 97% of it is the optimization of `w` |

| experiment | CPU-hours (`STS-Plain` / everything else) |
|---|---|
| `fig3` (100 runs) | below 1 / about 4 |
| `delta` (200 runs, 2 instances) | below 1 / about 3 |
| `fig5` (20 runs) | about 380 / about 15 |

## Results

| directory | content |
|---|---|
| `results/simulation/instance{i}_{algorithm}.json` | one entry per run: seed, stopping time `T`, recommended arm, correctness, runtime (total, sampling, stopping) |
| `results/delta_correctness/` | the same for the runs with delta = 0.1 |
| `results/glr_trace/instance{i}.json` | figure 2: every statistic along one run |
| `results/heatmaps/instance{i}.json` | figure 4: `F(w)` on a grid of the simplex for the four settings |
| `results/bound_tightness/` | `pairs.json` (Lambda_i, L_i, L_naive_i and the proven factor of every pair) and `characteristic_times.json` (T*, U, U_naive) |
