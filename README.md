# Fault-Tolerant Task Offloading Simulation (v2)

This project simulates a Mobile Edge-Cloud Computing environment using Deep Reinforcement Learning (DDPG) for fault-tolerant task offloading. The codebase is modular, extensible, and supports modern experiment tracking and configuration management.

## Features
- Modular structure: `env/`, `agent/`, `train/`, `utils/`, `tests/`
- YAML-based configuration (see `config.yaml`)
- Custom DDPG implementation (easily extensible)
- Modern experiment tracking with [Weights & Biases (wandb)](https://wandb.ai/)
- Unit and integration tests (`pytest`)
- API documentation with [MkDocs](https://www.mkdocs.org/)
- Performance profiling and support for parallel/distributed execution

## Setup

### Requirements
- Python 3.7+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  pip install wandb mkdocs pytest
  ```

### Configuration
- All simulation parameters are in `config.yaml`.
- Edit this file to change scenario, server/task counts, RL hyperparameters, etc.

## Usage

### 1. Generate Parameters
```bash
python -m utils.generate_server_and_task_parameters
```

### 2. Run Simulation
```bash
python -m train.ddpg_main
```
- Results and metrics are logged to [wandb](https://wandb.ai/).

### 3. Run Tests
```bash
pytest tests/
```

### 4. Build/View Documentation
```bash
mkdocs serve
# or
mkdocs build
```
- Docs are in the `docs/` directory.

## Modular Structure
- `env/`: Environment classes (`EnvState`, `Server`, `Task`)
- `agent/`: RL agent (`ddpgModel`, `Buffer`)
- `train/`: Training loop and experiment entry point
- `utils/`: Config loader, parameter management, (deprecated) logging
- `tests/`: Unit/integration tests

## Experiment Tracking
- All metrics, hyperparameters, and plots are logged to [wandb](https://wandb.ai/).
- Set your wandb API key with `wandb login` before running experiments.

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
# ...
```
Then rerun the parameter generation and simulation.

## Citing
If you use this simulator, please cite as described in `CITATION.cff`.

---
For more details, see the [documentation](docs/).

