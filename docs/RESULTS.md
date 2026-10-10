# Paper-reported results

These are the results reported in the [H-JEPA paper](https://arxiv.org/abs/2610.06805),
transcribed from the original repository README. They are reference values for reproduction,
not measurements from a new validation run.

## Depth comparison (Figure 6)

[Reproduction instructions](../README.md#35-depth-comparison-figure-6).

Planning success rate (%), mean ± standard error over model seeds 42, 43, 44,
with the planner seed paired to the model seed. Each environment uses 50 fixed start/goal tasks.
LeWM uses flat planning; H-JEPA and HWM use the planner matching their model depth.
The numbers in the column headings indicate model depth.

| | LeWM | H-JEPA 2 | H-JEPA 3 | H-JEPA 4 | HWM 2 | HWM 3 | HWM 4 |
|---|---|---|---|---|---|---|---|
| Visual AntMaze | 18.0 ± 3.5 | 39.3 ± 3.7 | 73.3 ± 3.5 | 67.3 ± 2.4 | 34.0 ± 1.2 | 38.7 ± 3.5 | 14.0 ± 2.0 |
| FourRoom Distractors | 40.7 ± 6.8 | 81.3 ± 6.7 | 96.0 ± 1.1 | 88.0 ± 2.3 | 75.3 ± 3.3 | 99.3 ± 0.7 | 92.7 ± 4.4 |
| OGBench Cube | 32.0 ± 6.1 | 47.3 ± 2.7 | 60.0 ± 5.0 | 62.7 ± 2.4 | 46.7 ± 2.9 | 53.3 ± 5.2 | 34.0 ± 3.1 |
| Push-T | 40.0 ± 2.3 | 45.3 ± 1.3 | 17.3 ± 3.5 | 0.7 ± 0.7 | 42.0 ± 6.4 | 12.0 ± 2.0 | 0.0 ± 0.0 |


## Cost ladder (Tables 1 and 9)

[Reproduction instructions](../README.md#36-cost-ladder-tables-1-and-9).

Planning success rate (%), mean ± standard error over model seeds 42, 43, 44,
with paired planner seeds and the same 50 fixed tasks per environment.
All entries use level-1 planning. Native L1 measures cost in the level-1 latent space;
L2/L3/L4 cost projects predictions and the goal into the corresponding upper latent space.
Blank cells indicate a cost level above the model's depth.

| | Model | Native L1 | L2 cost | L3 cost | L4 cost |
|---|---|---|---|---|---|
| Visual AntMaze | 2 levels | 23.3 ± 3.3 | 31.3 ± 2.7 | | |
| | 3 levels | 16.7 ± 1.8 | 20.7 ± 4.8 | 22.0 ± 2.3 | |
| | 4 levels | 10.0 ± 0.0 | 20.7 ± 1.8 | 21.3 ± 4.7 | 26.7 ± 2.9 |
| FourRoom Distractors | 2 levels | 9.3 ± 0.7 | 38.7 ± 6.6 | | |
| | 3 levels | 43.3 ± 8.7 | 65.3 ± 5.5 | 71.3 ± 4.1 | |
| | 4 levels | 68.7 ± 4.7 | 72.0 ± 3.1 | 66.0 ± 3.1 | 70.7 ± 2.7 |
| OGBench Cube | 2 levels | 24.0 ± 1.2 | 13.3 ± 0.7 | | |
| | 3 levels | 24.7 ± 1.3 | 28.0 ± 4.2 | 14.0 ± 3.5 | |
| | 4 levels | 26.7 ± 2.7 | 25.3 ± 2.4 | 12.0 ± 2.0 | 12.0 ± 2.3 |
| Push-T | 2 levels | 36.7 ± 1.8 | 38.0 ± 3.1 | | |
| | 3 levels | 30.7 ± 1.8 | 28.0 ± 4.0 | 24.7 ± 4.1 | |
| | 4 levels | 1.3 ± 0.7 | 1.3 ± 0.7 | 0.7 ± 0.7 | 1.3 ± 1.3 |

## DROID planning (Figure 9b)

[Reproduction instructions](../README.md#44-planning-evaluation).

Open-loop planning on 16 DROID evaluation clips at 5 fps, using the released epoch-100 checkpoints.
Fréchet fidelity (%) is reported as mean ± standard error over training seeds 1, 1000, 10000;
each training seed's result is first averaged over planner seeds 1, 2, 3.

| Model | Fréchet fidelity (%) |
|---|---|
| LeWM + IDM | 33.57 ± 1.52 |
| HWM | 35.65 ± 2.22 |
| H-JEPA | 38.42 ± 0.94 |

Fréchet fidelity is `1 − F(planned, expert) / F(zero motion, expert)`, where `F` is the discrete
Fréchet distance between cumulative xyz end-effector paths. The metric is computed per clip and
averaged over the 16 clips. Zero motion scores 0%, an exact path match scores 100%, and negative
values indicate worse fidelity than zero motion. This is an offline path metric, not closed-loop task success.
The `frechet/skill_mean` column in `eval.csv` stores the unscaled value; multiply by 100 for this table.

This table covers the three models evaluated by `scripts/eval_droid.sh`; it does not include
the no-IDM LeWM bar or the compute sweep in Figure 9(c).
