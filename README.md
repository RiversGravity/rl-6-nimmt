# Beating *6 nimmt!* with reinforcement learning

Leverage reinforcement learning, Monte-Carlo search, and tournament-style self-play to build agents that master the classic card game *6 nimmt!* This repository packages an OpenAI Gym environment, a family of agents, and utilities for orchestrating large-scale experiments so you can explore meta-learning strategies and evaluate emergent play.

> ### Quick Start
> | Step | Command |
> | --- | --- |
> | Clone the project | `git clone git@github.com:johannbrehmer/rl-6nimmt.git` |
> | Create the Conda environment | `conda env create -f environment.yml` |
> | Activate tools | `conda activate rl` |
> | Validate the install | `pytest` |
> | Explore workflows | `jupyter notebook experiments/simple_tournament.ipynb` |

## Project Overview

### What game are we solving?
[6 nimmt!](https://boardgamegeek.com/boardgame/432/6-nimmt) is an award-winning card game that mixes hidden information, stochasticity, and timing. Each round players simultaneously reveal cards and collect "Hornochsen" penalty points whenever they complete a row. Skilled play hinges on bluffing, anticipatory planning, and adaptable heuristics—making it a rich benchmark for reinforcement learning.

### Core building blocks

| Module | Role in the system | Highlights |
| --- | --- | --- |
| `rl_6_nimmt/env.py` | Implements `SechsNimmtEnv`, an OpenAI Gym-compatible environment that deals cards, enforces move legality, and scores penalty rows. | Configurable deck size, row thresholds, and verbose rendering for debugging. |
| `rl_6_nimmt/agents/` | Houses neural and search-based agents including REINFORCE, ACER, DQN variants, Monte-Carlo search, and a human interface. | Shared `Agent` base class, PyTorch models, card-memory utilities, and drop-in registration through `AGENTS`. |
| `rl_6_nimmt/tournament.py` | Coordinates population-based training via the `Tournament` class with dynamic roster management, baseline evaluation, and ELO tracking. | Automates matchmaking, logs tournament/baseline scores, and supports evolutionary cloning with configurable metrics. |

### Why tournaments?
Population-based self-play helps agents adapt to a shifting meta-game. `Tournament.play_game` continuously pits active models against each other, promotes top performers, and benchmarks progress against static baselines. This closed loop encourages robustness and uncovers diverse strategies more reliably than single-agent training.

## Getting Started

### Prerequisites
- **Python:** Tested with Python 3.8+ (installed via [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)).
- **CUDA (optional):** GPU acceleration is supported for neural agents but not required.
- **System packages:** `git`, `make`, and build tooling needed for PyTorch (installed automatically through Conda on Linux/macOS).

> 💡 *Tip:* If you prefer [mamba](https://mamba.readthedocs.io/) you can substitute `mamba` for `conda` in the commands below for faster solves.

### Create and activate the environment
```bash
# Clone the repository
git clone git@github.com:johannbrehmer/rl-6nimmt.git
cd rl-6nimmt

# Reproduce the curated environment
conda env create -f environment.yml
conda activate rl
```

### Verify your installation
Before launching long-running training, smoke-test the setup:
```bash
# Run the unit suite (env dynamics, agent interfaces, utilities)
pytest

# Optionally simulate a quick random match-up
python - <<'PY'
from rl_6_nimmt.tournament import Tournament
from rl_6_nimmt.agents import DrunkHamster

arena = Tournament()
arena.add_player("random_a", DrunkHamster())
arena.add_player("random_b", DrunkHamster())
arena.play_game(num_players=2)
print("Scores:", arena.tournament_scores)
PY
```
Successful execution confirms Gym registration, PyTorch dependencies, and tournament plumbing are operational.

## Train Your Own Agents

Follow this recipe to launch self-play experiments that evolve increasingly strong agents:

1. **Initialize a tournament population.** Combine learning agents with baselines to seed the meta-game.
   ```python
   from rl_6_nimmt.tournament import Tournament
   from rl_6_nimmt.agents import PolicyMCSAgent, BatchedACERAgent, DrunkHamster

   tournament = Tournament(
       min_players=2,
       max_players=4,
       baseline_agents=[DrunkHamster()],
       baseline_num_games=5,
       elo_initial=1600,
       elo_k=24,
   )

   tournament.add_player("pmcs_seed", PolicyMCSAgent(mc_per_card=20))
   tournament.add_player("acer_seed", BatchedACERAgent())
   tournament.add_player("random_baseline", DrunkHamster())
   ```

2. **Play games and learn.** Use `Tournament.play_game` to drive simulated matches. Agents invoke their `learn` methods after every environment step.
   ```python
   for generation in range(20):
       for _ in range(200):
           tournament.play_game()  # scores + ELO are updated internally
       # Periodically print KPIs
       print(generation, tournament.elos["pmcs_seed"][-1])
   ```

3. **Evolve the roster.** Clone top performers and retire weak agents using built-in heuristics.
   ```python
   tournament.evolve(copies=(2, 1), max_players=6, max_per_descendant=2, metric="elo")
   ```

4. **Persist checkpoints.** Agents are standard PyTorch modules; save state dicts or entire objects for later evaluation.
   ```python
   from pathlib import Path
   import torch

   checkpoint_dir = Path("experiments/checkpoints")
   checkpoint_dir.mkdir(parents=True, exist_ok=True)
   torch.save(tournament.agents["pmcs_seed"].state_dict(), checkpoint_dir / "pmcs_seed.pt")
   ```

5. **Automate with notebooks or scripts.** The `experiments/simple_tournament.ipynb` notebook demonstrates an end-to-end population loop and plotting utilities. For scripted runs, adapt the snippet above or explore `experiments/debug_*.py` for agent-specific debugging.

> 📈 **Resource planning:** Expect 1–2 GB GPU memory per neural agent (policy gradients, DQN variants) and moderate CPU usage from Monte-Carlo rollouts. Control runtime with knobs like `mc_per_card`, `baseline_condition`, and the tournament loop length.

## Evaluate & Play

### Track competitive progress
- **ELO trends:** Access `tournament.elos[name]` to chart historical strength. `experiments/elo.png` and `elo.pdf` provide example plots that aggregate thousands of games.
- **Baseline matches:** Configure `baseline_agents` and `baseline_num_games` to evaluate every *n*th game (`baseline_condition`). Metrics accumulate in `baseline_scores`, `baseline_positions`, and `baseline_wins` for longitudinal comparisons.
- **Experiment artifacts:** Log your own runs under `experiments/` (e.g., `experiments/checkpoints/`, `experiments/metrics.csv`) to consolidate configs, seeds, and trained weights.

### Run human-vs-agent sessions
- Launch interactive matches by pairing the human interface with any trained agent:
  ```python
  import torch
  from rl_6_nimmt.play import GameSession
  from rl_6_nimmt.agents import Human, PolicyMCSAgent

  human = Human()
  champion = PolicyMCSAgent()
  champion.load_state_dict(torch.load("experiments/checkpoints/pmcs_champion.pt"))

  session = GameSession(human, champion)
  session.play_game(render=True)
  ```
- Customize the experience by tweaking environment parameters (e.g., `SechsNimmtEnv(num_players=5, threshold=5)`), logging verbosity, or replacing agents mid-session.
- Troubleshooting tips:
  - If rendering is quiet, ensure `logging` is configured (e.g., `logging.basicConfig(level=logging.INFO)`).
  - On headless servers, run without `render=True` and stream moves via logs.

## Configuration Cheat Sheet

| Category | Key knob | Effect |
| --- | --- | --- |
| Environment | `num_players`, `num_rows`, `threshold`, `include_summaries` | Adjusts game difficulty, observation richness, and combinatorics. |
| Agents | `mc_per_card`, `mc_max`, network `hidden_sizes`, exploration schedules | Balance search depth vs. compute, tune neural capacity, and control randomness. |
| Tournament | `baseline_condition`, `elo_k`, `copies`, `max_players` | Frequency of evaluation, rating sensitivity, evolutionary pressure, and population size. |

Keep detailed notes alongside artifacts in `experiments/`—meta-learning thrives on reproducibility and disciplined comparisons across seeds and hyperparameters.

## Contributors
Created by Johann Brehmer and Marcel Gutsche.
