import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque, defaultdict


class SAREnv(gym.Env):
    """
    Single-agent Search-and-Rescue gridworld (Gymnasium).

    Task:
      1) Explore / scan to find victim
      2) Move to victim and pick up (rescue_dropoff)
      3) Return to base and drop off (rescue_dropoff) => success (terminated=True)

    Notes:
      - SB3 expects a scalar reward; we also provide a reward breakdown in info["reward_parts"].
      - reset() regenerates obstacles until the map is solvable (agent->victim and victim->base).
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    ACTIONS = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3,
        "scan": 4,
        "rescue_dropoff": 5,
    }
    ID_TO_ACTION = {v: k for k, v in ACTIONS.items()}

    # Reward weights (tune later). Keeping revisit penalties off by default.
    REWARDS = {
        "step": -0.05,          # efficiency
        "new_grid": 0.2,        # first-time visit bonus
        "bump": -1.0,           # wall/obstacle bump
        "scan_success": 3.0,    # scan reveals victim
        "scan_fail": -0.2,      # wasted scan
        "pickup": 5.0,          # pick up victim
        "dropoff": 20.0,        # drop at base (success)
        "bad_rescue": -0.5,     # invalid pickup/drop attempt
        "timeout": -10.0,       # timeout penalty
        "dist_shaping": 0.0,    # optional dense shaping (off by default)
        "revisit": 0.0,         # optional revisit penalty (off by default)
        "global_revisit": 0.0,  # optional global revisit penalty (off by default)
    }

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 200,
        obstacle_ratio: float = 0.15,
        render_mode: str | None = None,
        scan_radius: int = 1,
        auto_discover: bool = False,
        max_reset_tries: int = 200,
    ):
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not (0.0 <= obstacle_ratio < 1.0):
            raise ValueError("obstacle_ratio must be in [0.0, 1.0)")

        # Core
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.obstacle_ratio = float(obstacle_ratio)
        self.render_mode = render_mode

        # Difficulty knobs (curriculum)
        self.scan_radius = int(scan_radius)
        self.auto_discover = bool(auto_discover)
        self.max_reset_tries = int(max_reset_tries)

        # Action space
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # Observation:
        # [agent_x, agent_y, base_x, base_y, battery_norm, carrying, victim_found, rel_victim_x, rel_victim_y]
        # rel_victim_x/y is signed in [-1, 1] (only meaningful if victim_found else 0s)
        low = np.array([0, 0, 0, 0, 0, 0, 0, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1,  1,  1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Fixed base for now
        self.base_pos = np.array([0, 0], dtype=np.int32)

        # Runtime state
        self.agent_pos: np.ndarray | None = None
        self.victim_pos: np.ndarray | None = None
        self.obstacles: set[tuple[int, int]] = set()

        self.victim_found = False
        self.carrying_victim = False
        self.episode_success = False
        self.steps = 0
        self.battery = 0
        self.max_battery = self.max_steps

        self.visited: np.ndarray | None = None
        self.global_visit_count = defaultdict(int)

    # ---------------------------
    # Curriculum helper
    # ---------------------------
    def set_difficulty(
        self,
        *,
        grid_size: int | None = None,
        max_steps: int | None = None,
        obstacle_ratio: float | None = None,
        scan_radius: int | None = None,
        auto_discover: bool | None = None,
    ):
        if grid_size is not None:
            self.grid_size = int(grid_size)
        if max_steps is not None:
            self.max_steps = int(max_steps)
        if obstacle_ratio is not None:
            self.obstacle_ratio = float(obstacle_ratio)
        if scan_radius is not None:
            self.scan_radius = int(scan_radius)
        if auto_discover is not None:
            self.auto_discover = bool(auto_discover)

        self.max_battery = self.max_steps  # keep normalization consistent

    # ---------------------------
    # Helpers: obstacles + sampling
    # ---------------------------
    def _place_obstacles(self):
        """Random obstacles, never blocking the base."""
        n_cells = self.grid_size * self.grid_size
        n_obstacles = int(n_cells * self.obstacle_ratio)  # floor; deterministic count

        # Ensure we always have at least a few free cells
        n_obstacles = min(n_obstacles, n_cells - 2)

        self.obstacles = set()
        forbidden = {tuple(self.base_pos.tolist())}

        while len(self.obstacles) < n_obstacles:
            x = int(self.np_random.integers(0, self.grid_size))
            y = int(self.np_random.integers(0, self.grid_size))
            if (x, y) not in forbidden:
                self.obstacles.add((x, y))

    def _random_free_grid(self, forbidden=None) -> np.ndarray:
        """Sample a free cell not in obstacles and not in forbidden."""
        if forbidden is None:
            forbidden = []

        while True:
            pos = np.array(
                [
                    int(self.np_random.integers(0, self.grid_size)),
                    int(self.np_random.integers(0, self.grid_size)),
                ],
                dtype=np.int32,
            )
            if tuple(pos.tolist()) in self.obstacles:
                continue
            if any(np.array_equal(pos, x) for x in forbidden):
                continue
            return pos

    # Backwards-compat alias (your code referenced _random_free_cell in one branch)
    _random_free_cell = _random_free_grid

    # ---------------------------
    # Helpers: reachability (BFS)
    # ---------------------------
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _is_reachable(self, start: np.ndarray, goal: np.ndarray) -> bool:
        """BFS check that a path exists given obstacles."""
        s = (int(start[0]), int(start[1]))
        g = (int(goal[0]), int(goal[1]))

        if s == g:
            return True
        if s in self.obstacles or g in self.obstacles:
            return False

        q = deque([s])
        seen = {s}

        while q:
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if not self._in_bounds(nx, ny):
                    continue
                if (nx, ny) in self.obstacles:
                    continue
                if (nx, ny) in seen:
                    continue
                if (nx, ny) == g:
                    return True
                seen.add((nx, ny))
                q.append((nx, ny))
        return False

    def _generate_init_pos(self):
        """Generate obstacles + positions until solvable; otherwise fallback to obstacle-free."""
        for _ in range(self.max_reset_tries):
            self._place_obstacles()
            self.agent_pos = self._random_free_grid(forbidden=[self.base_pos])
            self.victim_pos = self._random_free_grid(forbidden=[self.base_pos, self.agent_pos])

            if self._is_reachable(self.agent_pos, self.victim_pos) and self._is_reachable(self.victim_pos, self.base_pos):
                return

        # Fallback: no obstacles (always solvable)
        self.obstacles = set()
        self.agent_pos = self._random_free_grid(forbidden=[self.base_pos])
        self.victim_pos = self._random_free_grid(forbidden=[self.base_pos, self.agent_pos])

    # ---------------------------
    # Obs / info
    # ---------------------------
    def _get_obs(self) -> np.ndarray:
        if self.victim_found:
            denom = max(self.grid_size - 1, 1)
            rel = (self.victim_pos - self.agent_pos) / denom  # signed
            rel_vx, rel_vy = rel.astype(np.float32)
        else:
            rel_vx, rel_vy = 0.0, 0.0

        return np.array(
            [
                self.agent_pos[0] / (self.grid_size - 1),
                self.agent_pos[1] / (self.grid_size - 1),
                self.base_pos[0] / (self.grid_size - 1),
                self.base_pos[1] / (self.grid_size - 1),
                self.battery / self.max_battery,
                float(self.carrying_victim),
                float(self.victim_found),
                rel_vx,
                rel_vy,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        # Coverage/redundancy for evaluation (even single agent)
        total_cells = self.grid_size * self.grid_size
        coverage = len(self.global_visit_count) / total_cells if total_cells > 0 else 0.0
        redundancy = sum(1 for v in self.global_visit_count.values() if v > 1)

        return {
            "steps": self.steps,
            "battery": self.battery,
            "victim_found": self.victim_found,
            "carrying_victim": self.carrying_victim,
            "success": bool(self.episode_success),
            "coverage": float(coverage),
            "redundancy": int(redundancy),
        }

    # ---------------------------
    # Gym API
    # ---------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Optional difficulty override
        if isinstance(options, dict) and options:
            self.set_difficulty(
                grid_size=options.get("grid_size", None),
                max_steps=options.get("max_steps", None),
                obstacle_ratio=options.get("obstacle_ratio", None),
                scan_radius=options.get("scan_radius", None),
                auto_discover=options.get("auto_discover", None),
            )

        self.steps = 0
        self.battery = self.max_battery
        self.victim_found = False
        self.carrying_victim = False
        self.episode_success = False

        # Generate solvable map + init positions
        self._generate_init_pos()

        # Visited map and global visit counts
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.visited[self.agent_pos[0], self.agent_pos[1]] = 1

        self.global_visit_count = defaultdict(int)
        self.global_visit_count[tuple(self.agent_pos.tolist())] += 1

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Allow stepping via action name (manual debugging), but SB3 uses ints.
        if isinstance(action, str):
            if action not in self.ACTIONS:
                raise ValueError(f"Unknown action name: {action}")
            action_id = self.ACTIONS[action]
        else:
            action_id = int(action)

        # Reward breakdown dict
        r = {k: 0.0 for k in self.REWARDS.keys()}
        terminated = False
        truncated = False

        # Step cost
        r["step"] += self.REWARDS["step"]

        self.steps += 1
        self.battery -= 1

        old_pos = self.agent_pos.copy()

        # -------------------------
        # Movement actions
        # -------------------------
        if action_id in (self.ACTIONS["up"], self.ACTIONS["down"], self.ACTIONS["left"], self.ACTIONS["right"]):
            move_map = {
                self.ACTIONS["up"]: np.array([-1, 0], dtype=np.int32),
                self.ACTIONS["down"]: np.array([1, 0], dtype=np.int32),
                self.ACTIONS["left"]: np.array([0, -1], dtype=np.int32),
                self.ACTIONS["right"]: np.array([0, 1], dtype=np.int32),
            }
            new_pos = self.agent_pos + move_map[action_id]

            if (
                0 <= new_pos[0] < self.grid_size
                and 0 <= new_pos[1] < self.grid_size
                and tuple(new_pos.tolist()) not in self.obstacles
            ):
                self.agent_pos = new_pos

                # Exploration bonus (first visit)
                if self.visited[self.agent_pos[0], self.agent_pos[1]] == 0:
                    r["new_grid"] += self.REWARDS["new_grid"]
                else:
                    # Optional revisit penalty (off by default)
                    r["revisit"] += self.REWARDS["revisit"]

                self.visited[self.agent_pos[0], self.agent_pos[1]] += 1
            else:
                r["bump"] += self.REWARDS["bump"]

            # Optional curriculum: stepping on victim reveals it
            if self.auto_discover and (not self.victim_found) and np.array_equal(self.agent_pos, self.victim_pos):
                self.victim_found = True
                r["scan_success"] += self.REWARDS["scan_success"]

        # -------------------------
        # Scan
        # -------------------------
        elif action_id == self.ACTIONS["scan"]:
            dist = int(np.abs(self.agent_pos - self.victim_pos).sum())
            if dist <= self.scan_radius and not self.victim_found:
                self.victim_found = True
                r["scan_success"] += self.REWARDS["scan_success"]
            else:
                r["scan_fail"] += self.REWARDS["scan_fail"]

            # Count staying on same cell as revisit (optional)
            self.visited[self.agent_pos[0], self.agent_pos[1]] += 1

        # -------------------------
        # Rescue / Drop-off
        # -------------------------
        elif action_id == self.ACTIONS["rescue_dropoff"]:
            if np.array_equal(self.agent_pos, self.victim_pos) and self.victim_found and not self.carrying_victim:
                self.carrying_victim = True
                r["pickup"] += self.REWARDS["pickup"]

            elif np.array_equal(self.agent_pos, self.base_pos) and self.carrying_victim:
                self.carrying_victim = False
                self.episode_success = True
                terminated = True
                r["dropoff"] += self.REWARDS["dropoff"]

            else:
                r["bad_rescue"] += self.REWARDS["bad_rescue"]

            # Count staying on same cell as revisit (optional)
            self.visited[self.agent_pos[0], self.agent_pos[1]] += 1

        # -------------------------
        # Optional distance shaping (only when goal is valid)
        # -------------------------
        if self.REWARDS["dist_shaping"] != 0.0:
            goal = None
            if self.carrying_victim:
                goal = self.base_pos
            elif self.victim_found:
                goal = self.victim_pos

            if goal is not None:
                old_d = int(np.abs(old_pos - goal).sum())
                new_d = int(np.abs(self.agent_pos - goal).sum())
                r["dist_shaping"] += self.REWARDS["dist_shaping"] * (old_d - new_d)

        # -------------------------
        # Global visit count / coverage metrics
        # -------------------------
        pos_t = tuple(self.agent_pos.tolist())
        if self.global_visit_count[pos_t] > 0:
            r["global_revisit"] += self.REWARDS["global_revisit"]
        self.global_visit_count[pos_t] += 1

        # -------------------------
        # Truncation (time/battery)
        # -------------------------
        if self.battery <= 0 or self.steps >= self.max_steps:
            truncated = True
            if not terminated:
                r["timeout"] += self.REWARDS["timeout"]

        reward = float(sum(r.values()))
        obs = self._get_obs()
        info = self._get_info()

        # Add debugging/logging fields
        info["reward_parts"] = r
        info["action_id"] = action_id
        info["action_name"] = self.ID_TO_ACTION.get(action_id, "unknown")

        return obs, reward, terminated, truncated, info

    def render(self):
        """ASCII render for debugging."""
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=object)

        for (x, y) in self.obstacles:
            grid[x, y] = "#"

        bx, by = self.base_pos
        grid[bx, by] = "B"

        # If carrying, hide victim marker; otherwise show V if found, ? if unknown
        if not self.carrying_victim:
            vx, vy = self.victim_pos
            grid[vx, vy] = "V" if self.victim_found else "?"

        ax, ay = self.agent_pos
        grid[ax, ay] = "A"

        print("\n".join(" ".join(row) for row in grid))
        print()