# Deep Q-Learning Agent for Traffic Signal Control

A **PyTorch-based Deep Q-Learning agent** that learns to control a single 4-way intersection in **SUMO**. The project includes a configurable training pipeline, CLI, and plotting utilities to focus on experimentation and evaluation.

- **Agent**: epsilon-greedy DQN with experience replay and configurable fully connected network.
- **Environment**: fixed SUMO intersection; state is 80 binary cells from discretized incoming lanes; 4 traffic-signal actions.
- **Outputs**: trained model, settings copy, and plots for rewards, delay, and queue lengths.

---

## Prerequisites

- [uv](https://astral.sh/uv) installed.
- Python 3.13 (managed via `uv`; see `pyproject.toml`).
- SUMO installed and accessible via `sumo` / `sumo-gui` on your PATH. Set `SUMO_HOME` accordingly. [Official SUMO installation guide](https://sumo.dlr.de/docs/Installing/index.htm).
- GPU is optional; CPU training is supported.

> **macOS users:** Certain versions of SUMO-GUI with XQuartz may crash. See:
>
> - [https://github.com/eclipse-sumo/sumo/issues/17272](https://github.com/eclipse-sumo/sumo/issues/17272)
> - [https://github.com/XQuartz/XQuartz/issues/446](https://github.com/XQuartz/XQuartz/issues/446)

---

## Installation

Clone the repository and install dependencies:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Check CLI availability:

```bash
tlcs --help
```

Deactivate when finished:

```bash
deactivate
```

Or run commands without activating:

```bash
uv run tlcs --help
```

---

## Training and Testing

Training and testing read YAML configuration from `settings/`. Both prompt before overwriting existing outputs.

**Train** (defaults to `settings/training_settings.yaml`, output to `model/`):

```bash
tlcs train
tlcs train --out-path model/run-01
```

**Test** a trained run (defaults to `settings/testing_settings.yaml`, output to `model/<run>/test/`):

```bash
tlcs test --model-path model/run-01 --test-name foo
```

Discover all options:

```bash
tlcs train --help
tlcs test --help
```

---

## Configuration

Settings live in `settings/`:

**training_settings.yaml**

- `gui`: run SUMO GUI (`true`) or headless (`false`)
- `total_episodes`, `max_steps`, `n_cars_generated`: episode count, length, and traffic volume
- `green_duration`, `yellow_duration`: phase durations in seconds
- `turn_chance`: probability that a vehicle turns instead of going straight
- `num_layers`, `width_layers`: neural network architecture
- `batch_size`, `learning_rate`, `training_epochs`: optimizer and replay batch parameters
- `memory_size_min`, `memory_size_max`: replay buffer warmup and capacity
- `gamma`: discount factor
- `sumocfg_file`: SUMO configuration path

**testing_settings.yaml**

- Mirrors simulation settings plus `episode_seed` for reproducibility

---

## Outputs

Each training run saves:

- `trained_model.pt`: PyTorch model
- `training_settings.yaml`: settings copy
- Plots: `plot_reward.png`, `plot_delay.png`, `plot_queue.png` + raw data
- Testing outputs in `test/` subfolder with reward and queue plots

---

## Project Structure

```
src/tlcs/
├─ cli.py           # CLI for training/testing
├─ main.py          # Training/testing orchestration
├─ agent.py         # Epsilon-greedy policy
├─ model.py         # Neural network definition
├─ memory.py        # Replay buffer
├─ env.py           # SUMO wrapper
├─ generator.py     # Episode route generation
├─ plots.py         # Plotting and data export
intersection/       # SUMO assets (network, config, routes)
settings/           # YAML configs for training/testing
```

---

## Methodology

### Scenario and State

- Single 4-way junction; each arm has 4 incoming lanes (750 m)
- Lanes grouped by movement (left vs. straight/right) → **8 lane groups**
- Each group discretized into **10 distance buckets** → **80 binary cells**
- State vector: `1` if at least one vehicle occupies a cell, else `0`

### Actions and Signals

Four fixed green phases (yellow added automatically on changes):

1. North-South straight/right
2. North-South left
3. East-West straight/right
4. East-West left

Durations controlled by `green_duration` and `yellow_duration`.

### Reward

Reward is the **change in cumulative waiting time**:

```
reward = previous_total_wait - current_total_wait
```

Reducing total wait yields positive reward.

### Traffic Generation

- One route file per episode generated on the fly
- Departure times follow Weibull distribution scaled to `[0, max_steps]`
- `turn_chance` determines turn vs. straight routes
- Seeds ensure reproducibility

### Policy and Learning Loop

- Epsilon-greedy exploration: epsilon decays linearly from 1 → 0
- Q-targets: `r + gamma * max_a' Q(next_state, a')`
- Experience replay with warmup (`memory_size_min`) and capacity (`memory_size_max`)
- Configurable MLP trained with MSE loss and Adam optimizer

**Episode Loop:**

1. Generate routes
2. Run SUMO until `max_steps`
3. Collect transitions
4. Push to replay buffer
5. Train network for `training_epochs` batches
6. Log and plot cumulative reward, delay, and queue lengths

---

## Tips

- Prefer headless mode (`gui: false`) for training; GUI for debugging/testing
- CLI prompts before overwriting output folders

---

## License

MIT – see `LICENSE`.
