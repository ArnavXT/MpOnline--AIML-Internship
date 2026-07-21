# Implementation Plan: RL Agent Training (CartPole & Lunar Lander)

Two separate, independent projects. Each ships as its own folder/zip with its
own dependencies, training script, evaluation script, and README.

## Overview

| | CartPole-v1 | LunarLander-v3 |
|---|---|---|
| **Difficulty** | Beginner | Intermediate |
| **State space** | 4 continuous values | 8 continuous values |
| **Action space** | 2 discrete (left/right) | 4 discrete (thrusters) |
| **Algorithm** | DQN, built from scratch in PyTorch | PPO, via Stable-Baselines3 |
| **Why this pairing** | Learn value-based RL fundamentals by writing DQN yourself | Apply the concepts at scale with a production-grade library — PPO-from-scratch is error-prone and slow to tune |
| **Est. training time** | 5–10 min on CPU | 15–30 min on CPU for a good result, less with a GPU |
| **Success criterion** | Avg reward ≥ 475 over 100 episodes | Avg reward ≥ ~200 over 100 episodes |
| **Verified** | Smoke-tested: training loop, model save/load, and evaluation all run without errors | Smoke-tested: training loop, model save/load, and evaluation all run without errors |

## Phase 1 — Environment Setup
1. Create a virtual environment per project (they have separate `requirements.txt`).
2. Install dependencies. LunarLander additionally needs the `swig` system package
   to build the box2d physics engine (`apt-get install swig` / `brew install swig`).
3. Sanity-check with `gym.make(...)` before training.

## Phase 2 — CartPole (custom DQN)
1. Replay buffer + Q-network + target network + epsilon-greedy policy (`dqn_agent.py`).
2. Training loop: sample batches, compute TD targets, backprop Huber loss,
   periodic hard target-network updates (`train.py`).
3. Logs episode reward and epsilon every 10 episodes; stops early once solved.
4. Saves `cartpole_dqn.pt` and a reward-curve PNG.
5. `evaluate.py` runs the greedy policy over N episodes and reports mean ± std.

## Phase 3 — Lunar Lander (PPO via Stable-Baselines3)
1. 8 vectorized environments (`make_vec_env`) for stable, fast rollout collection.
2. `EvalCallback` periodically evaluates and checkpoints the best policy.
3. PPO hyperparameters tuned for LunarLander (gamma=0.999, gae_lambda=0.98, ent_coef=0.01).
4. TensorBoard logging to `logs/` for live monitoring.
5. `evaluate.py` loads either the final model or the best checkpoint and reports mean ± std reward.

## Phase 4 — Evaluation & Comparison
- Both `evaluate.py` scripts follow the same pattern: load the saved policy,
  run N episodes with greedy/deterministic actions, print mean ± std reward.
- Compare episodes-to-solve, wall-clock time, and final reward across the two projects.

## Phase 5 — Stretch Goals (optional)
- CartPole: Double DQN, Dueling DQN, Prioritized Experience Replay.
- Lunar Lander: try `LunarLanderContinuous-v3` with PPO or SAC; hyperparameter
  sweep with Optuna; record video with `gymnasium.wrappers.RecordVideo`.

## Deliverables
- `cartpole_dqn.zip` — standalone CartPole project
- `lunarlander_ppo.zip` — standalone Lunar Lander project
