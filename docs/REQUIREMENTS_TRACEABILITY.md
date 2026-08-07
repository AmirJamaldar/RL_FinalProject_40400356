# Requirement traceability

This table maps each project requirement to its implementation and the evidence available in the package. The implementation items have been verified from the code, tests, and saved experiment outputs. The GitHub publication and final PDF are listed as submission checks because they can only be confirmed after the final repository is assembled.

| Requirement | Implementation | Verification/evidence | Status |
|---|---|---|---|
| Student ID, base seed, map size | `experiments/configs/*.json`, `environments/generator.py` | `test_student_specific_seed_and_size`, `map_validation.json` | Verified |
| 15-18 square map; >=15% walls; >=5 penalties | `generate_source_map`, `validate_map` | map tests and `map_validation.json` | Verified |
| Start -> key -> locked door -> goal | structural barrier and augmented-state BFS in `generator.py` | `test_hard_map_requirements_and_bfs` | Verified |
| Stochastic action 0.8 intended, 0.1 each perpendicular | `DynamicMazeEnv.transition_distribution/step` | exact-probability test | Verified |
| Markov state includes key status | state `(row, col, has_key)` throughout | history-independence test | Verified |
| Extra mechanic affects decisions and GUI | bidirectional teleporter in transition and renderer | teleporter transition test; GUI renderer | Verified |
| Sparse and shaped rewards | `RewardSpec`, potential-based shaping | sparse/shaped VI agreement = 100% in `reward_design_verification.json` | Verified |
| Prevent shaping/reward exploits | zero repeatable door bonus; bounded shaping scale | reward design verification and VI invariance | Verified |
| All required events logged | environment JSONL logger and witness generation | `required_event_witnesses.jsonl`; coverage test | Verified |
| Episode cap = 3 x traversable cells | environment/config construction | shape/limit test; map validation | Verified |
| Termination distinct from time truncation | five-value `step` result | truncation test | Verified |
| Value Iteration from scratch | `agents/value_iteration.py` | Bellman residual and convergence tests | Verified |
| VI heatmap, policy, iterations, time | experiment raw data/models and analysis | `vi_gamma_sensitivity.csv`, figures | Verified |
| At least three gamma values | 0.80, 0.95, 0.99 | gamma CSV/figure | Verified |
| Q-Learning off-policy update | `agents/q_learning.py` | exact numeric update test and `q_update_trace.csv` | Verified |
| Linear and exponential epsilon decay | `EpsilonSchedule` and experiment matrix | schedule tests, raw curves/figure | Verified |
| Per-episode Q metrics/events | Q trainer | `training_episodes.csv` | Verified |
| SARSA(lambda), on-policy traces | `agents/sarsa_lambda.py` | lambda tests and `sarsa_trace.csv` | Verified |
| lambda = 0, 0.3, 0.7, 0.9 | quick/full configs and matrix | raw data and sensitivity figure | Verified |
| Replacing versus accumulating trace choice | configurable, replacing default | tests and executed config | Verified |
| Same-map three-algorithm comparison | source/shaped primary runs | run/aggregate summaries and comparison figure | Verified |
| Time, samples, stability, memory, path quality, sensitivity | summaries and analysis; SARSA memory includes its eligibility workspace | `aggregate_summary.csv`, memory unit test, figures | Verified |
| Model-free policy agreement with VI | `policy_agreement` | summaries and mismatch figure | Verified |
| At least three local mismatch examples | `mismatch_examples` | six records in `policy_mismatch_examples.json` | Verified |
| Similar target changes 15-20% obstacles | transfer generator | 17.8% of all source walls; tests/validation | Verified |
| Different target changes >=35%, relocates key, adds penalties | transfer generator | 40.3% of all source walls; tests/validation | Verified |
| Target BFS validity | transfer generator retry + validation | map tests and validation JSON | Verified |
| Scratch/full/scaled/selective transfer | `transfer_learning.py` | all modes tests and full 120-run matrix | Verified |
| beta = 0.25, 0.50, 0.75 | configs/initializer | numeric transfer tests and raw data | Verified |
| Initial, learning-speed, final transfer metrics | transfer experiment loop | transfer summary/curves | Verified |
| Negative transfer and correction | exact target-model regret witness | `negative_transfer_witnesses.json` | Verified |
| Animated GUI and controls | `gui/app.py`, `gui/renderer.py` | GUI state-machine and checkpoint tests; successful Windows desktop check of launch, Start/Pause/Resume, evaluation, and teleporter behavior | Verified |
| Evaluation loads/preserves real trained models | semantic checkpoint resolver in `gui/app.py` | algorithm/map/gamma/lambda/fingerprint tests; `verify_project.py` evaluates four packaged selections | Verified |
| Value/policy/visit/path/mismatch/transfer figures | `experiments/analysis.py` | 14 PNGs and `figure_manifest.json` | Verified |
| Raw CSVs, configs, Q tables, README, tests | project tree | full experiment and test suite | Verified |
| Reproducibility and no cherry-picking | isolated training RNGs, common evaluation randomness, all configured seeds saved | per-episode seeds, repeatability test, exact count gates, manifests and all-run CSVs | Verified |
| Public GitHub, repository name, `main` branch, and >=3 meaningful commits | final repository publication | confirm the public URL, repository name `RL_FinalProject_40400356`, default branch, and commit history | Final submission check |
| Final PDF report and analytical discussion | `report.pdf` in the project root | open the final PDF and check its sections, figures, tables, and AI-use disclosure against the specification | Final submission check |
