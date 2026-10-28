# Results, endpoints and limitations

`data/nine-v1-repeats.csv` holds 27 baseline/candidate duration pairs, three per
case. For each case, take the median of the paired ratios TB/TC; take the
geometric mean across nine cases. `scripts/verify_results.py` checks the score,
absolute marginal medians, group summaries and archived gate flags.

- Frozen development score: **3.7490935717621388**.
- Workload: Si, Cu-Ni, MgO; small, medium, large; fixed model and conditions.
- Scope: full scored intervals, including required output and sealing.
- These repeats are not ABBA. A median of ratios need not equal a ratio of medians.
- Throughput improvement is `TB/TC - 1`; time reduction is `1 - TC/TB`.
- `verified=false`; independent post-timer audit `not_run`, zero checked intervals.
- 36 physical-stage flags are reported as passing. Error arrays are incomplete.

`data/correctness-coverage.json` describes recovered protocol coverage.
`numeric-tolerances.json` and `replay-thresholds.csv` contain acceptance limits,
not observed errors. The recovered protocol describes 5000-step (5 ps) checks;
this is not a claim of long-timescale scientific stability.

Local intervention rows use their own baselines and measurement scopes. Do not
multiply them into the nine-case score. Profiling windows, microbenchmarks,
static resource counts and negative results do not become complete-MD timings.
No unified causal ablation or cross-platform generalization is claimed.

The public analysis inputs let a reader recompute the reported arithmetic. They
do not substitute for executable reproduction with the fixed model, evaluator,
SDK, inputs, correctly loaded runtime and full physical checks.
