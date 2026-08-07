# Verification record

The full experiment profile is recorded in `results/raw_data/run_manifest.json`. This file is the machine-readable record of the completed run and is written only after the configured experiment matrix, model checks, and figure generation finish successfully.

## Acceptance checks

- map generation is deterministic for student ID 40400356;
- every map passes augmented-state BFS;
- Bellman updates converge below the configured max-norm threshold;
- sparse and potential-shaped rewards yield the same optimal policy;
- every tabular array and checkpoint is finite and shape-valid;
- evaluation uses epsilon zero and cannot mutate Q;
- fixed-seed training is exactly reproducible;
- all policies use common evaluation environment seeds;
- checkpoints match the selected map by semantic fingerprint;
- all four transfer modes and all three beta values execute;
- negative transfer is supported by exact target-model regret and post-training Q values;
- every figure is regenerated from saved raw data and model files;
- stale generated outputs and temporary files cannot survive a clean rerun.

## Full-profile results

The final full-profile run completed on 2026-07-22. The saved outputs were checked again with the following results:

- profile: `full_experiment` with 10 seeds;
- source summaries: 71 (1 Value Iteration, 30 Q-Learning, 40 SARSA);
- transfer runs: 120 (2 targets x 6 initializations x 10 seeds);
- persisted episode rows: 210,000 source training and 180,000 transfer training;
- models: 196 archives and 404 finite numeric arrays;
- figures: 14 valid PNGs, all visually inspected;
- sparse/shaped optimal-policy agreement: 100%;
- maximum persisted Bellman residual: `6.719602652083266e-11`;
- negative-transfer witnesses: 2, both corrected after target training;
- policy mismatch examples: 3 Q-Learning and 3 SARSA;
- unit/integration suite: 45/45 passed in the final workspace run;
- manifest/source SHA-256 (cross-platform and independent of local `.venv`):
  `0e10940d384b2b20ce6008a480e559adebf41c1bbb2dd4fa02b6139d6168ac88`.

Independent replay of the selected 3,000-episode Q-Learning and `SARSA(lambda=0.7)` runs reproduced their Q tables and visit tables exactly. The saved episode metrics were also reproduced after writing and reading the CSV files. Automated GUI tests covered the state transitions, checkpoint loading, and all four packaged evaluation selections.

The GUI was also run in a real Windows desktop session. Launch, Start, Pause, Resume, evaluation, and teleporter behavior were checked successfully. Reset-episode and rerun-from-scratch behavior are additionally covered by the GUI state-machine tests.

The verification commands are:

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
python verify_project.py
```
