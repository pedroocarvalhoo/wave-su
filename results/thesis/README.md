# Explore the thesis results

Browse the graphs first, then open the corresponding CSVs for a closer look.
These are the results used in [the WAVE thesis](../../thesis.pdf).

[Back to WAVE](../../README.md) · [Run an experiment](../../docs/README.md) · [Input datasets](../../datasets/README.md)

## Start with the workload overview

![Operational carbon and P95 waiting-time overview for the 500-job and 1,250-job thesis campaigns.](plots/workloads/overview.png)

The overview compares FIFO, Priority, EFT and Puzzle within each workload size.
Open a policy's detailed plot to follow its carbon, power and energy-budget
traces over time.

## Pick a comparison

| Results | Graphs (PNG) | Data (CSV) |
|---|---|---|
| Main supply-profile comparisons | [plots/main/](plots/main/) | [data/main/](data/main/) |
| Carbon-threshold sensitivity | [carbon_threshold.png](plots/carbon_threshold.png) | [data/carbon_threshold/](data/carbon_threshold/) |
| Energy policies | [main plots](plots/main/) | [data/energy_policies/](data/energy_policies/) |
| 500-job workloads | [plots/workloads/500/](plots/workloads/500/) | [data/workloads/500/](data/workloads/500/) |
| 1,250-job workloads | [plots/workloads/1250/](plots/workloads/1250/) | [data/workloads/1250/](data/workloads/1250/) |
| Battery and fixed-frequency controls | [plots/controls/](plots/controls/) | [data/controls/](data/controls/) |
| Summary tables | | [data/summary/](data/summary/) |

The main folders use `grid`, `solar` and `solar_wind`, followed by the scheduling
policy: for example, [`main/solar_wind_puzzle/`](plots/main/solar_wind_puzzle/).
The 500-job and 1,250-job campaigns have their own folders and comparison groups.

## A detailed run

![The 500-job Puzzle run, with cumulative operational carbon, budget-source composition and actor-power traces.](plots/workloads/500/puzzle.png)

[Open the image](plots/workloads/500/puzzle.png) · [Inspect this run's CSVs](data/workloads/500/puzzle/)

## Read the CSVs

| File | Contents |
|---|---|
| `energy_simulation_result.csv` | Monitor time series for the microgrid and energy budgets |
| `jobs_metrics_<cluster>.csv` | Per-job completion, waiting and execution metrics |
| `carbon_metrics.csv` | Derived energy and carbon metrics, where exported |
| `data/summary/*.csv` | Tables used for the reported comparisons |

A run shared by several comparisons is stored once. The CSVs retain their
original values and columns; the graph folders contain the evaluation and
Appendix A images exported as PNG.

The [scheduling-policy illustration](plots/scheduling_policy_demo.png) is a
didactic example. New simulations write to the local `outputs/` directory.
