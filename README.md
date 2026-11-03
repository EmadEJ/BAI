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
| `optimization.py` | the non-convex GLR: `plain_GLR` (local solver from a few starting points), `exact_GLR_k2` (closed form for k = 2), `reference_GLR` (exact for k = 2, 40 random starting points otherwise, our "true value"), and the w-optimization `optimize_bfgs` |
| `algorithms/` | `ASTS` (GTS-A), `MuSTS` (GTS-mu), `STS` (GTS-Unknown, `glr="lb"` or `glr="plain"`), `SGTS` (GTS-SG); `TS.py` has the tracking rules |
| `environment.py` | the bandit and the run loop (`Environment.run`) |
| `utils.py` | `C_exp` of [KK21] and `W_bar` of [TD25a] used in the thresholds |
| `simulate.py` | many runs of one algorithm on one instance (figures 3, 5 and the delta-correctness check) |
| `glr_trace.py` | figure 2 |
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
| `STS-Plain` | GTS-Unknown (solver) | plain solver of the GLR (an upper bound, not provably delta-correct) | plain solver inside BFGS |
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
python simulate.py --algorithm ASTS --instance_index 21 --runs 200 --delta 0.1 --out_dir results/delta_correctness/ --jobs 8
python glr_trace.py --instance_index 10 --log_period 50 --jobs 8
python heatmaps.py --instance_index 10 --div 61 --jobs 8
python bound_tightness.py --jobs 8
python optimization.py -n 2 -k 3 --fixed_w --name plain   # checks the GLR solver against the grid search of testsets/
```

Do not run the same (instance, algorithm) twice at the same time: both processes would write the same result file.

### On the server

```bash
git pull
source .venv/bin/activate
tmux new -s bai                      # or nohup, so that the runs survive the ssh session
JOBS=$(nproc) ./run_all.sh fig3 fig5 delta
# when it is done (logs/ has the progress of every run)
python plot.py all
git add results figures
git commit -m "server runs"
git push
```

The runs are parallelized over processes (`--jobs`), and `simulate.py` sets `OMP_NUM_THREADS=1` so that the
processes do not compete for threads.

## Results

| directory | content |
|---|---|
| `results/simulation/instance{i}_{algorithm}.json` | one entry per run: seed, stopping time `T`, recommended arm, correctness, runtime (total, sampling, stopping) |
| `results/delta_correctness/` | the same for the runs with delta = 0.1 |
| `results/glr_trace/instance{i}.json` | figure 2: every statistic along one run |
| `results/heatmaps/instance{i}.json` | figure 4: `F(w)` on a grid of the simplex for the four settings |
| `results/bound_tightness/` | `pairs.json` (Lambda_i, L_i, L_naive_i and the proven factor of every pair) and `characteristic_times.json` (T*, U, U_naive) |
