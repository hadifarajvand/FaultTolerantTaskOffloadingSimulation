# Fault-Tolerant Task Offloading Simulation (v2)

This project simulates a Mobile Edge-Cloud Computing environment using Deep Reinforcement Learning (DDPG, TD3, A2C) for fault-tolerant task offloading. The codebase is modular, extensible, and supports modern experiment tracking and configuration management.

## Features
- Modular structure: `env/`, `agent/`, `train/`, `utils/`, `tests/`
- YAML-based configuration (see `config.yaml`)
- Supports multiple DRL agents: DDPG, TD3, A2C (PyTorch)
- Experiment orchestration: run all agents in sequence, logs separated by agent
- Unit and integration tests (`pytest`)
- API documentation with [MkDocs](https://www.mkdocs.org/)
- Performance profiling and support for parallel/distributed execution

## Setup

### Requirements
- Python 3.7+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  pip install mkdocs pytest torch
  ```

### Configuration
- All simulation parameters are in `config.yaml`.
- Edit this file to change scenario, server/task counts, RL hyperparameters, or agent type (`AGENT_TYPE`).

## How to Run the Workflow (Step-by-Step)

### 1. Generate Server and Task Parameters
```bash
python -m utils.generate_server_and_task_parameters
```

### 2. Run All DRL Agents (DDPG, TD3, A2C)
```bash
python -m train.ddpg_main
```
- This will run the simulation for each agent in sequence.
- Logs for each agent are saved in separate folders:
  - `logs/ddpg/training_log.csv`
  - `logs/td3/training_log.csv`
  - `logs/a2c/training_log.csv`
- You can analyze and compare results by inspecting these CSV files.

#### To run a single agent only:
- Edit `config.yaml` and set `AGENT_TYPE: ddpg` (or `td3`, `a2c`), then run:
  ```bash
  python -m train.ddpg_main
  ```

### 3. Run Unit and Integration Tests
```bash
PYTHONPATH=. pytest tests/
```
- This will test all agent instantiations and core logic.

### 4. Build/View Documentation
```bash
mkdocs serve
# or
mkdocs build
```
- Docs are in the `docs/` directory.

## Modular Structure
- `env/`: Environment classes (`EnvState`, `Server`, `Task`)
- `agent/`: RL agents (`ddpgModel`, `td3Model`, `a2cModel`, `Buffer`)
- `train/`: Training loop and experiment entry point
- `utils/`: Config loader, parameter management, logging
- `tests/`: Unit/integration tests

## Experiment Tracking
- All metrics and logs are saved as CSV files in the `logs/` directory, separated by agent type.

## Performance Profiling
- Use Python profilers (e.g., `cProfile`, `line_profiler`) for bottleneck analysis.
- For parallel/distributed runs, consider [Ray](https://docs.ray.io/en/latest/) or [joblib](https://joblib.readthedocs.io/).

## API Documentation
- All classes and functions have comprehensive docstrings.
- Browse the API docs with MkDocs (`mkdocs serve`).

## Example: Customizing a Scenario
Edit `config.yaml`:
```yaml
SCENARIO_TYPE: heterogeneous
NUM_EDGE_SERVERS: 8
NUM_CLOUD_SERVERS: 3
AGENT_TYPE: td3
# ...
```
Then rerun the parameter generation and simulation.

## Citing
If you use this simulator, please cite as described in `CITATION.cff`.

---
For more details, see the [documentation](docs/).

