# Fault-Tolerant Task Offloading Simulation

Welcome to the documentation for the Fault-Tolerant Task Offloading Simulation project.

## Overview

This project simulates a Mobile Edge-Cloud Computing environment using Deep Reinforcement Learning (DDPG) for fault-tolerant task offloading. The codebase is modular and organized into the following main packages:

- **env/**: Environment components (servers, tasks, state management)
- **agent/**: RL agent implementations (DDPG, etc.)
- **train/**: Training and experiment orchestration
- **utils/**: Configuration, parameter management, and utilities
- **tests/**: Unit and integration tests

## Getting Started

- See the [README](../../README.md) for setup and usage instructions.
- Browse the API documentation for details on each module and class.

## Structure

- **env/**: Contains `EnvState`, `Server`, and `Task` classes.
- **agent/**: Contains the DDPG agent and buffer.
- **train/**: Contains the main training loop and experiment entry point.
- **utils/**: Contains configuration loader, parameter management, and (deprecated) logging utilities.
- **tests/**: Contains unit and integration tests for core logic.

---

For more details, see the navigation on the left.
