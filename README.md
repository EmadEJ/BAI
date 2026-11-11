# Best Arm Identification with Unknown Post-Action Context Distribution

Code of the experiments of the paper. Every experiment stores its raw results in `results/`, and `plot.py` draws
every figure and table from them, so figures can be changed without rerunning anything.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Instances

`instances/instance{i}.json` holds `n`, `k`, `confidence` (delta), `mus` and `A`. `A` is given row by row: one row per
arm, with the probabilities of the k contexts (the transpose of the k x n matrix of the paper).

| index | n, k, delta | used for |
|---|---|---|
| 0 | 3, 2, 1e-2 | a small sample instance, the default of the scripts (every algorithm stops within a minute) |
| 1 | 3, 2, 1e-6 | figure 2, left (k = 2) |
| 2 | 3, 3, 1e-6 | figure 2, middle (k = 3) |
| 3 | 3, 4, 1e-6 | figure 2, right (k = 4) |
| 4 | 3, 2, 1e-3 | eq. (16), figure 3 |
| 5 | 3, 2, 1e-3 | eq. (17), figure 3 |
| 6 | 3, 3, 1e-3 | eq. (42), figure 4 |
| 7, 8 | 3, 2, 1e-3 | delta-correctness check (run with delta = 0.1), easier versions of instances 4 and 5 |
| 10 n + k | n, k, 1e-3 | figure 5: 32, 33, 34, 62, 63, 64, 92, 93, 94, 122, 123, 124 |

The values of instances 0 to 8 have at most 2 decimals. Instances 1, 2, 4, 5 and 6 are also the main instances of the
bound tightness table. The figure 5 instances have the rows of `A` drawn by `utils.context_initiator(n, k, "random")`
and `mus` uniform in [0, 1] (instance 124 with `np.random.seed(2)`), and `io_utils.add_instance_to_json` writes a new
instance file.

## The code

| file | content |
|---|---|
| `lowerbound.py` | the closed-form lower bounds `L_bound` (`L_i^new`, Theorem 1 of the bound report `new_simple_bound.pdf`) and `L_naive` (the Pinsker bound, eq. (26)), the proven factors of `L_bound` (Theorems 2-3), and the convex sampling rule `optimal_w` |
| `optimization.py` | the non-convex GLR: `plain_GLR` (local solver from a few starting points), `exact_GLR_k2` (closed form for k = 2), `reference_GLR` (exact for k = 2, 20 random starting points otherwise, our "true value"), and the w-optimization `optimize_bfgs` |
| `algorithms/` | `ASTS` (GTS-A), `MuSTS` (GTS-mu), `STS` (GTS-Unknown, `glr="lb"`, `glr="naive"` or `glr="plain"`), `SGTS` (GTS-SG); `TS.py` has the tracking rules |
| `environment.py` | the bandit and the run loop (`Environment.run`) |
| `utils.py` | `C_exp` of [KK21] and `W_bar` of [TD25a] used in the thresholds |
| `simulate.py` | many runs of one algorithm on one instance (figures 3, 5 and the delta-correctness check) |
| `glr_trace.py` | figure 2 (instances 1, 2 and 3) |
| `heatmaps.py` | figure 4 |
| `bound_tightness.py` | the table and figure comparing the bounds with the true GLR |
| `plot.py` | all figures and tables from the stored results |
| `run_all.sh` | every experiment of the paper |

Algorithm names used on the command line and in the result files:

| name | paper | stopping | sampling |
|---|---|---|---|
| `ASTS` | GTS-A | exact GLR (A known) | exact (convex) |
| `MuSTS` | GTS-mu | exact GLR (mu known) | Appendix F.1 |
| `STS-LB` | GTS-Unknown (ours) | `min_i L_i^new(t) > beta(t, delta)`; for k = 2 the exact GLR (closed form) | maximizer of `w -> min_i L_i^new(w, A w)` (convex); for k = 2 the exact GLR inside BFGS |
| `STS-LB-Naive` | GTS-Unknown (Naive) | `min_i L_naive_i(t) > naive_beta(t, delta)`, for every k | maximizer of `w -> min_i L_naive_i(w, A w)` (convex) |
| `STS-Plain` | GTS-Unknown (solver) | plain solver of the GLR (an upper bound, not provably delta-correct); the closed form for k = 2 | the same GLR inside BFGS; not in the figures or in `run_all.sh` |
| `SGTS` | GTS-SG | sub-Gaussian GLR, threshold of [TD25a] | exact (convex) |

All algorithms recompute the optimal `w` only every `10^(d-2)` rounds once `t` has `d` digits (`--no_fast` turns this
off). `STS-Plain` also checks its (expensive) stopping rule only once every `t/100` rounds, which can only delay
stopping by at most 1%. `STS-LB` and `STS-LB-Naive` check their (closed-form) stopping rule at every round.

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

Single experiments (without arguments, `simulate.py`, `glr_trace.py` and `heatmaps.py` run on the sample instance 0):

```bash
python simulate.py                                                  # one run of GTS-Unknown (ours) on instance 0
python simulate.py --algorithm STS-LB --instance_index 4 --runs 100 --jobs 8
python simulate.py --algorithm ASTS SGTS --instance_index 32 33 --runs 20 --jobs 8
python simulate.py --algorithm ASTS --instance_index 7 --runs 200 --delta 0.1 --out_dir results/delta_correctness/ --jobs 8
python glr_trace.py --instance_index 2 --log_period 50 --jobs 8
python heatmaps.py --instance_index 6 --div 61 --jobs 8
python bound_tightness.py --jobs 8
python optimization.py -n 2 -k 3 --fixed_w --name plain   # checks the GLR solver against the grid search of testsets/
```

In `testsets/fixed_w_n=2_k=3.json`, 4 of the 100 grid search solutions have rewards outside [0, 1] (they were made
before the grid was clipped), so the solver, which keeps the rewards in [0, 1], is reported above them on those 4.

`simulate.py` also takes several algorithms and instances (`--algorithm STS-LB ASTS --instance_index 4 5`), all
their missing runs then share one pool, the slowest algorithm first. Do not run the same (instance, algorithm) in two
commands at the same time: both would write the same result file. To stop the runs, use Ctrl-C (or
`pkill -f simulate.py`; on macOS the workers are `multiprocessing.spawn` processes and need to be killed too).

### Figure 2 on another instance

1. Write `instances/instance10.json` (any unused number, see [Instances](#instances) for the format). All entries of `A`
   must be positive, its rows sum to 1, `"mus"` is in [0, 1] and the best arm is unique.
2. Run the trace, then draw it (the paper figure `fig2_glr_trace` is not touched):

```bash
python glr_trace.py --instance_index 10 --log_period 20 --jobs 8
python plot.py fig2 10          # figures/fig2_glr_trace_10.pdf and its log scale version fig2_glr_trace_10_log.pdf
```

`glr_trace.py` prints when each statistic crosses each threshold. The run continues until both lower bounds crossed the
naive threshold, so a harder instance (smaller gap, smaller delta) runs longer; keep about 1000 to 2000 logged rounds
with `--log_period` (the run length divided by 1500). `--delta` overrides the delta of the instance, `--seed` changes
the sample path, and `--reference_starts` (20 by default) is the number of random starting points of the true GLR for
k >= 3 (for k = 2 it is the exact closed form). On the plot the vertical lines are the stopping times of the methods:
the true GLR and `L` at our threshold, `L_naive` at the naive threshold. With 8 cores it takes about 1.5 min for instance 1 (k = 2), 1 min for
instance 2 (k = 3) and 5 min for instance 3 (k = 4, with `--reference_starts 100`).

### Expected runtimes

Measured on one core of a laptop (Apple M5, all cores busy), a server core can be up to 2x slower. `figures/tables.md`
has the measured time of every stored run.

| algorithm | time per run | |
|---|---|---|
| `ASTS`, `SGTS`, `STS-LB`, `STS-LB-Naive` | 1 s to 1.5 min (seconds for `STS-LB` with k = 2) | negligible |
| `MuSTS` | 0.5 to 8 min | |
| `STS-Plain`, k = 2 | seconds (closed-form GLR) | |
| `STS-Plain`, k >= 3 | 35 min (instance 33) to about 4 h (instance 124) | 97% of it is the optimization of `w` |

| experiment | CPU-hours (`run_all.sh`, without `STS-Plain`) | `STS-Plain` alone |
|---|---|---|
| `fig3` (100 runs) | about 4 | below 1 |
| `delta` (200 runs, 2 instances) | about 3 | below 1 |
| `fig5` (20 runs) | about 25 (17 of it `MuSTS`) | about 490 |

## Results

| directory | content |
|---|---|
| `results/simulation/instance{i}_{algorithm}.json` | one entry per run: seed, stopping time `T`, recommended arm, correctness, runtime (total, sampling, stopping) |
| `results/delta_correctness/` | the same for the runs with delta = 0.1 |
| `results/glr_trace/instance{i}.json` | figure 2: every statistic along one run |
| `results/heatmaps/instance{i}.json` | figure 4: `F(w)` on a grid of the simplex for the four settings |
| `results/bound_tightness/` | `pairs.json` (Lambda_i, L_i, L_naive_i and the proven factor of every pair) and `characteristic_times.json` (T*, U, U_naive) |
