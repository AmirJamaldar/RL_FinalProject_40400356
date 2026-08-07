# RL Final Project - Student 40400356

This repository contains a tabular reinforcement-learning implementation of the dynamic key-door-goal maze. For student number `40400356`, the second-to-last digit is `5`; therefore, the base random seed is `5` and the generated maze size is `16 x 16`.

A bidirectional teleporter pair is included as the additional environment mechanic. The teleporter changes the next state, so it is part of the transition model rather than a visual-only feature. It is also shown in the GUI. The agent state is `(row, column, has_key)`, which includes all information needed to determine the next-state distribution and reward.

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Tkinter is part of standard Python on Windows and macOS. On minimal Linux installations, install the OS package `python3-tk` if it is missing.

## Run

Launch the interactive GUI:

```bash
python main.py
```

Run a terminal-only demonstration:

```bash
python main.py --headless --algorithm vi --episodes 10
python main.py --headless --algorithm q --episodes 500
python main.py --headless --algorithm sarsa --episodes 500
```

Reproduce the delivered full 10-seed experiment matrix and all figures:

```bash
python experiments/run_experiments.py
```

The files included in `results/` were generated with this full profile. The shorter profile below is intended only for development and quick checks.

Run the shorter 3-seed development/validation profile:

```bash
python experiments/run_experiments.py --config experiments/configs/quick.json
```

Regenerate figures without retraining:

```bash
python experiments/run_experiments.py --analysis-only
```

## Verify

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
python verify_project.py
```

The automated tests cover map constraints and BFS validity, stochastic transition probabilities, reward and event semantics, termination versus truncation, teleporter behavior, Bellman convergence, exact Q updates, SARSA trace snapshots, reproducibility, evaluation isolation, GUI checkpoint handling, map fingerprints, transfer initialization, negative transfer, clean reruns, and command-line execution. `verify_project.py` also decodes the saved models and figures, checks the expected experiment counts and hashes, and performs a short evaluation of each checkpoint packaged for the GUI.

Verified snapshot included in this package:

- full 10-seed profile;
- 71 source summaries and 120 transfer runs;
- 210,000 source-training and 180,000 transfer-training episode rows;
- 196 model archives containing 404 finite numeric arrays;
- 14 decoded and visually inspected figures;
- 45 unit/integration tests;
- cross-platform source hash `0e10940d384b2b20ce6008a480e559adebf41c1bbb2dd4fa02b6139d6168ac88`.

The source hash covers the project source files and the JSON configuration and map files. Virtual environments, caches, version-control or IDE metadata, and generated files under `results/` are excluded. This keeps the verification result consistent across Windows, Linux, and macOS.

When the GUI switches from training to evaluation, it keeps the learned Q table and sets epsilon to exactly zero. If there is no model in memory, the program searches for a compatible saved Q-Learning or source-map SARSA checkpoint and checks its algorithm, map, parameters, and map fingerprint before loading it. A SARSA policy for a target map can be trained in the GUI and evaluated immediately afterward. If no compatible checkpoint is available, the GUI blocks evaluation and displays an error instead of using an unrelated model.

## Outputs

- `environments/maps/`: source, similar-target, and different-target maps in JSON format.
- `results/raw_data/`: episode logs, evaluation results, traces, validation records, summaries, and the run manifest.
- `results/models/`: Value Iteration outputs and the saved Q-Learning, SARSA, and transfer checkpoints.
- `results/figures/`: heatmaps, policies, visit counts, paths, parameter sweeps, policy disagreement, and transfer-learning plots.
- `docs/REQUIREMENTS_TRACEABILITY.md`: mapping between the project requirements, implementation, and verification evidence.

Each checkpoint stores a semantic fingerprint of its map. The same environment-noise seeds are used when comparing algorithms and transfer scenarios, so differences in the reported policies are not caused by different random draws during evaluation.

No prebuilt reinforcement-learning algorithm is used. The tabular updates are implemented with NumPy; pandas and Matplotlib are used only for saving, analyzing, and plotting the experiment results.
