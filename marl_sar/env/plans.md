Below is a concrete, execution-oriented plan for **2 Tier 1 projects + 1 Tier 2 flagship**, built so that each Tier 1 directly feeds components into the Tier 2, and you still get **Unity visuals early**.

I’m assuming a sustainable cadence of **5–7 hours/week** (3 short evenings + one weekend block). If you do more, you compress the calendar; the structure stays the same.

---

## Portfolio Structure

**One monorepo** (recommended for reuse) or **three repos** (clean separation). Monorepo is faster to maintain and lets you reuse logging/plotting/viewer code.

**Monorepo layout**

```
portfolio-autonomy/
  common/                 # logging, plotting, configs, utilities
  tier1_curriculum/       # Project 1
  tier1_comms/            # Project 2
  unity_replay_viewer/    # Unity visualizer used by both Tier 1s (+ Tier 2 later)
  tier2_marl_sar_unity/   # Flagship
  README.md               # portfolio index with links, videos, results
```

**Non-negotiable “portfolio polish” for every project**

* `README.md`: 3 commands (install → train → eval)
* `results/`: at least 2 plots + 1 table
* `demo/`: 30–90s video/GIF
* Reproducibility: fixed seeds + evaluation suite
* Clear “What’s new / What I learned / Limitations”

---

# Tier 1 Project 1 (2–4 weeks): Curriculum Learning for Sparse Reward Exploration

### Goal

Prove you can apply SOTA **curriculum learning** to stabilize training and improve sample efficiency in a sparse-reward, partially observable task.

### Minimal environment (keep it tight)

**“Find the Target in Unknown Maze”** (single-agent, partial observability)

* Map: 2D grid maze, 3 difficulty levels (easy/medium/hard)
* Observation: local egocentric view (or local radius + rays)
* Reward: **sparse** (+1 only when target found) + small step penalty
* Metrics: success rate, steps-to-success, sample efficiency (timesteps to reach 70% SR)

### Curriculum schedule (stage-based)

* Stage 0: easy maze + short horizon
* Stage 1: medium + medium horizon
* Stage 2: hard + long horizon
  Promotion rule: move up when **SR ≥ 70% over 50 eval episodes**.

### Deliverables (what you publish)

* Plot 1: **Success rate vs environment steps** (baseline vs curriculum)
* Plot 2: **Success vs difficulty** (easy/med/hard at convergence)
* Table: **Timesteps to reach SR threshold** + final SR + variance across seeds
* Demo: Unity replay of one “before vs after” run (optional but recommended)

### Definition of done

You can reproduce the figures from scratch in < 30 minutes on your machine.

---

# Tier 1 Project 2 (2–4 weeks): Contested Comms Robustness in MARL (with Unity Replay)

### Goal

Demonstrate “defence-shaped” autonomy: **degraded communications** (loss/latency/jamming) and show robustness curves.

### Benchmark choice (fastest)

Use a known multi-agent environment (e.g., PettingZoo MPE `simple_spread`) and inject a comms model:

* Range-limited comms
* Packet drop `p_drop`
* Latency `L` steps
* Jamming zones (increase drop / blackout)

### What you measure

* Success rate vs packet loss (0% → 60%)
* Success rate vs latency (0 → 5 steps)
* Collision rate / congestion proxy (if available)
* Optional: with/without a simple coordination heuristic as a baseline

### Unity integration (this is your visual hook)

Build a **Unity Replay Viewer** that:

* Loads episode logs (`.json`)
* Replays agent positions and events
* Draws comm links (green when received, red when jammed/dropped)
* Toggles: “show comms”, “show jammer zones”, playback speed

### Deliverables

* Plot 1: **SR vs packet loss**
* Plot 2: **SR vs latency**
* Demo video: same seed replayed under perfect comms vs degraded vs jam zones (very compelling)

### Definition of done

One command generates logs + plots, and Unity can replay any episode log.

---

# Tier 2 Flagship (8–16 weeks): Multi-Agent SAR in Unknown Environment with Curriculum + Comms Denial + Hybrid Recovery (Unity ML-Agents)

### Goal

A defence-relevant flagship: **multi-agent search and rescue** in unknown environments with:

* partial observability
* dynamic targets
* comms denial (loss/latency/jamming)
* curriculum learning
* hybrid “stuck recovery” mechanism (safety/reliability)

This is the project that ties your story together.

### Core features (phased; you ship progressively)

1. **Unity SAR environment (MVP)**
2. **Multi-agent shared policy baseline**
3. **Curriculum training** (reuse Tier 1 principles)
4. **Comms denial** (reuse Tier 1 comm model + evaluation curves)
5. **Hybrid recovery** (rule-based escape when stuck)
6. Optional: LLM “Coordinator” later (Phase 2 extension after portfolio-ready baseline)

### Deliverables

* Results table (baseline vs +curriculum vs +comms denial vs +hybrid recovery)
* Robustness curves (SR vs packet loss/latency)
* Demo video showing:

  * unknown map exploration,
  * jam zones impact,
  * hybrid recovery preventing deadlocks,
  * improved mission completion under comms loss

### Definition of done

A recruiter can run a pre-trained policy in Unity (inference demo) and see the behaviors.

---

# 14-Week Execution Plan (realistic, sustainable)

You will publish something meaningful by Week 4–6 (Tier 1s), then build the flagship.

## Week 0 (setup week, 2–4 hours)

* Create repo + `common/` utilities (logging, configs, plotting template)
* Decide libraries for training (SB3/PPO is fine for Tier 1)
* Create “portfolio index” README with placeholders

**Output:** clean skeleton + first “hello plot”.

---

## Weeks 1–3: Tier 1 Project 1 (Curriculum)

* W1: implement env + baseline training + eval script
* W2: add curriculum schedule + promotion rule + run 2–3 seeds
* W3: plots + table + README + short demo log

**Output:** curriculum figures + reproducible scripts.

---

## Weeks 4–6: Tier 1 Project 2 (Comms Denial) + Unity Replay Viewer

* W4: implement comms model + sweeps (loss/latency) + logs
* W5: Unity replay viewer MVP (load log, replay agents, draw comms)
* W6: finalize plots + record demo video + README

**Output:** robustness curves + Unity visuals.

---

## Weeks 7–14: Tier 2 Flagship (Unity ML-Agents SAR)

* W7: Unity SAR MVP (map + obstacles + targets + reset loop)
* W8: multi-agent shared policy baseline + basic success metric
* W9–W10: curriculum training integrated (difficulty stages)
* W11: add comms denial in Unity (same semantics as Tier 1)
* W12: evaluation harness + robustness curves
* W13: hybrid recovery (“stuck detector” + deterministic escape)
* W14: polish: demo build, video, ablations, final report section in README

**Output:** flagship demo + results + cohesive narrative.

If you are slower than expected, you can still ship a strong flagship by stopping at **W12** (curriculum + comms denial + robustness curves) and adding hybrid recovery later as a “v2”.

---

# Weekly time budget (recommended)

* 3 × 60-minute sessions (weekday): code + small tests
* 1 × 2–3 hour session (weekend): training runs + analysis + documentation
* Keep one full day project-free.

---

# What you should do first (this weekend)

1. Create the monorepo skeleton (structure above).
2. Implement the **common logging format** for episode rollouts (even for Tier 1).
3. Start Tier 1 Curriculum: implement the smallest sparse-reward “find target” env and a baseline PPO run.

If you want, I can provide the **exact checklist and artifacts** for Week 1 (files to create, functions/classes you need, and what “baseline works” looks like), plus a simple episode log schema that Unity can replay.

Absolutely—let’s make this concrete. I’ll give you:

1. A **Week-1 checklist** (files to create, what “working” looks like) for the plan: **Tier 1 Curriculum + Tier 1 Comms + Tier 2 SAR**
2. A clear explanation of **how SB3 + PettingZoo “multi-agent” training actually works** (what it is and what it is not)
3. A practical guide to **MAPPO** (what changes vs SB3+wrappers, and how to run it reliably)
4. A curated **reading list** in a sensible order

---

## Week 1 kickoff plan (deliverables + exact tasks)

### Week 1 objective

By end of Week 1 you should have:

* A reproducible repo skeleton
* A working **Tier 1 Curriculum** baseline run (even if performance is poor)
* A standard **logging + evaluation harness** that you’ll reuse everywhere
* A Unity “viewer shell” that can load a JSON log and replay *something* (even 1 agent)

### Repo skeleton (create these folders/files)

```
portfolio-autonomy/
  common/
    logging.py
    eval.py
    plotting.py
    config.py
  tier1_curriculum/
    envs/
      find_target_env.py
    train_ppo.py
    eval_policy.py
    configs/
      baseline.yaml
      curriculum.yaml
    results/  (auto-generated)
  unity_replay_viewer/
    (Unity project)
  README.md
```

### Day-by-day checklist (minimal, no scope creep)

**Day 1 (1–2 hrs): Setup + “hello training run”**

* Create venv + install:

  * `stable-baselines3`, `gymnasium`, `numpy`, `matplotlib`, `tensorboard`
* Add `tier1_curriculum/envs/find_target_env.py`: a tiny Gymnasium env (grid or continuous 2D).
* Run `train_ppo.py` for 50k–100k steps just to validate the loop.

**Day 2 (1–2 hrs): Evaluation + fixed seeds**

* Implement `common/eval.py`:

  * run 20–50 eval episodes
  * compute success rate, mean steps-to-success, collisions (if applicable)
* Save `results/metrics.json` + `results/episodes/*.json` (trajectory logs).

**Day 3 (1–2 hrs): Plotting**

* Implement `common/plotting.py`:

  * learning curve (success rate vs timesteps)
  * optional: success vs difficulty (easy/med/hard)

**Day 4 (60–90 mins): Unity viewer shell**

* Create a simple Unity scene:

  * plane + agent prefab (cube)
  * `ReplayManager.cs` loads `episodes/*.json` and animates position over time
* This is the “visual layer” you’ll reuse later.

**Definition of done (Week 1)**

* One command produces:

  * a trained checkpoint
  * `metrics.json`
  * at least one plot
  * at least one replayable episode log

---

## How SB3 + PettingZoo “works” (and why it can feel confusing)

### The key mismatch

* **PettingZoo** is multi-agent (many agents take actions).
* **SB3** is single-agent (one policy controlling one agent in one env).

To bridge this, the Farama ecosystem uses **SuperSuit wrappers** to convert a PettingZoo env into something SB3 can consume. SuperSuit explicitly supports PettingZoo and provides conversion wrappers. ([PyPI][1])
PettingZoo also publishes SB3 tutorials using this approach. ([PettingZoo][2])

### What you typically get with SB3 + SuperSuit

Most SB3+PettingZoo tutorials implement **parameter sharing**:

* You train **one policy** π(a|o)
* That same policy controls all agents (each agent feeds its own observation `o_i` into the shared policy)
* Training batches contain transitions from **all agents**, which improves data efficiency

This is a valid MARL baseline commonly called **parameter-sharing PPO** (closely related to “Independent PPO” but with shared weights).

### What you do *not* automatically get

You do **not** automatically get MAPPO/CTDE:

* There is no centralized critic by default
* The critic does not see the full global state unless you explicitly include it in the observation (which changes the problem)

If you want **MAPPO**, you need a setup where:

* Actor uses local obs
* Critic can use global state or joint information during training (CTDE)

---

## MAPPO in practice (what changes and how to train it reliably)

### Concept: what MAPPO adds

MAPPO is essentially PPO with a **centralized critic** (CTDE). This line of work is popular because PPO can be surprisingly strong in cooperative MARL when implemented correctly. ([arXiv][3])

So:

* **SB3+wrappers**: shared actor+critic, usually local obs only
* **MAPPO**: decentralized actors, centralized critic (often global state during training)

### Practical ways to run MAPPO without reinventing everything

Pick one:

**Option A (fastest “known-good”): use the authors’ on-policy MAPPO codebase**

* The MAPPO paper and its ecosystem provide reference implementations and practices; the paper “The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games” is the canonical starting point. ([arXiv][3])
  This reduces the chance you fight framework quirks.

**Option B (engineering-friendly): RLlib multi-agent PPO with parameter sharing and/or variable sharing**

* RLlib has first-class multi-agent support and documents multi-agent env patterns and variable-sharing between policies. ([Ray][4])
  You can start with shared policy mapping, then upgrade to centralized critic.

**Option C (research convenience): MARLlib PPO family**

* MARLlib summarizes PPO-family multi-agent algorithms (IPPO/MAPPO) and can be a quick way to test baselines. ([MarlLib][5])

### What “good MAPPO training hygiene” looks like

* Run multiple seeds (at least 3)
* Use success-rate evaluation episodes on fixed seeds/maps
* Keep reward scaling consistent
* Track entropy / KL / clip fraction (PPO stability indicators)
* Add curriculum only after baseline learning is confirmed

---

## Concrete training guidance (so you’re not guessing)

### If you stick with SB3 + PettingZoo first (recommended for learning)

Follow the PettingZoo SB3 tutorials as your “ground truth” for wrappers and expected patterns. ([PettingZoo][2])

**Mental model:**

* ParallelEnv returns dicts keyed by agent → wrapper converts to vector format SB3 understands
* You’re effectively training a single policy across all agents (parameter sharing)
* “More agents” often acts like “more parallel data,” but also increases non-stationarity

### If you move to MAPPO later

Use the same environment and evaluation suite, but add:

* a “global state” tensor (or joint features) for the critic input during training
* keep actor obs local

You’ll know you’ve done CTDE correctly when:

* Removing global critic input reduces performance on coordination-heavy tasks
* But execution remains decentralized (each agent acts from local obs at test time)

---

## Suggested reading (high value, minimal fluff)

### 1) Framework literacy (to stop fighting tooling)

* PettingZoo SB3 tutorials (official patterns) ([PettingZoo][2])
* SuperSuit overview (what wrappers exist, what they do) ([PyPI][1])

### 2) PPO/MAPPO baseline excellence (the “why it works”)

* **Yu et al., “The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games”** (MAPPO/IPPO practices, ablations, stability) ([arXiv][3])
* BAIR MAPPO explainer (practical intuition) ([BAIR][6])

### 3) Parameter sharing (exactly what you’re doing with SB3+wrappers)

* “Revisiting Parameter Sharing in Multi-Agent Deep RL” (formalizes why shared policies work and when they don’t) ([OpenReview][7])

### 4) Engineering-grade multi-agent training systems

* RLlib multi-agent docs (policy mapping, variable sharing patterns) ([Ray][4])

---

## What I recommend you do next (to start this week)

1. Implement **Tier 1 Curriculum** as a single-agent Gymnasium env (fast and controlled).
2. Reuse the **same logging schema** you’ll later use for MARL: per-timestep position, events, and success outcomes.
3. Build the Unity replay viewer shell early so every run can become a video.

If you want, I can provide:

* a minimal `FindTargetEnv` Gymnasium implementation (grid-based, sparse reward, three difficulty presets), and
* a simple episode log JSON schema that Unity can replay, plus a `ReplayManager.cs` outline.

Tell me whether you prefer the Tier 1 Curriculum environment to be:

* **Gridworld (fast, deterministic)** or
* **Continuous 2D (more robotics-like)**

and I’ll tailor the starter code accordingly.

[1]: https://pypi.org/project/SuperSuit/3.3.5/?utm_source=chatgpt.com "SuperSuit 3.3.5"
[2]: https://pettingzoo.farama.org/tutorials/sb3/index.html?utm_source=chatgpt.com "Stable-Baselines3 Tutorial"
[3]: https://arxiv.org/abs/2103.01955?utm_source=chatgpt.com "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games"
[4]: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html?utm_source=chatgpt.com "Multi-Agent Environments - RLlib - Ray Docs"
[5]: https://marllib.readthedocs.io/en/latest/algorithm/ppo_family.html?utm_source=chatgpt.com "Proximal Policy Optimization Family - MARLlib - Read the Docs"
[6]: https://bair.berkeley.edu/blog/2021/07/14/mappo/?utm_source=chatgpt.com "The Surprising Effectiveness of PPO in Cooperative Multi- ..."
[7]: https://openreview.net/pdf?id=MWj_P-Lk3jC&utm_source=chatgpt.com "REVISITING PARAMETER SHARING IN MULTI-AGENT ..."
