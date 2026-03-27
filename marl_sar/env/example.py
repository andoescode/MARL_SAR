import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class SearchRescueEnv(gym.Env):
    """
    Single-agent grid Search-and-Rescue toy environment (Gymnasium compatible).

    Task loop:
      1) Explore the grid to locate a victim (either by scanning or stepping on it if auto_discover=True)
      2) Go to victim location and pick up (action=5)
      3) Return to base and drop off (action=5) => success/terminate

    Why this env is SB3-friendly:
      - Discrete action space
      - Flat 1D float observation vector
      - Standard Gymnasium API: reset()->(obs,info), step()->(obs,reward,terminated,truncated,info)

    Key upgrades vs your initial version:
      - Reachability check in reset(): avoids impossible episodes caused by random obstacles
      - set_difficulty(): stage-based curriculum control (grid_size, max_steps, obstacles, scan settings)
      - max_battery always syncs with max_steps so obs normalization remains correct
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

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

        # ----------------------------
        # Core environment parameters
        # ----------------------------
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.obstacle_ratio = float(obstacle_ratio)
        self.render_mode = render_mode

        # ----------------------------
        # Curriculum / difficulty knobs
        # ----------------------------
        # scan_radius: Manhattan distance threshold for successful scan
        # auto_discover: if True, stepping on victim cell reveals it automatically (easier Stage 0)
        self.scan_radius = int(scan_radius)
        self.auto_discover = bool(auto_discover)

        # If obstacles make the map unsolvable too many times, fallback to obstacle-free
        self.max_reset_tries = int(max_reset_tries)

        # ----------------------------
        # Action space
        # ----------------------------
        # 0 up, 1 down, 2 left, 3 right, 4 scan, 5 rescue/drop
        self.action_space = spaces.Discrete(6)

        # ----------------------------
        # Observation space (flat vector)
        # ----------------------------
        # All values normalized to [0, 1] where possible.
        #
        # [agent_x, agent_y,
        #  base_x, base_y,
        #  battery_norm,
        #  carrying_flag,
        #  victim_found_flag,
        #  rel_victim_x, rel_victim_y]  (only valid if victim_found else zeros)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )

        # Fixed base location in this toy problem (you can randomize later)
        self.base_pos = np.array([0, 0], dtype=np.int32)

        # Runtime state (set in reset())
        self.agent_pos = None
        self.victim_pos = None
        self.obstacles: set[tuple[int, int]] = set()

        self.victim_found = False
        self.carrying_victim = False
        self.episode_success = False
        self.steps = 0
        self.battery = None

        # Battery is just a step budget, normalized in observations
        self.max_battery = self.max_steps

        # Visited count grid encourages exploration
        self.visited = None

    # ============================
    # Curriculum helper
    # ============================
    def set_difficulty(
        self,
        *,
        grid_size: int | None = None,
        max_steps: int | None = None,
        obstacle_ratio: float | None = None,
        scan_radius: int | None = None,
        auto_discover: bool | None = None,
    ) -> None:
        """
        Update difficulty BEFORE calling reset().

        Important detail:
          - max_battery must match max_steps to keep battery normalization consistent.
        """
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

        # Keep battery normalization consistent across curriculum stages
        self.max_battery = self.max_steps

    # ============================
    # Sampling + obstacles
    # ============================
    def _random_free_cell(self, forbidden=None):
        """
        Sample a random free cell that:
          - is not an obstacle
          - is not in forbidden list
        """
        if forbidden is None:
            forbidden = []

        while True:
            pos = np.array(
                [
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size),
                ],
                dtype=np.int32,
            )
            pos_tuple = tuple(pos.tolist())

            # Reject if obstacle
            if pos_tuple in self.obstacles:
                continue

            # Reject if forbidden (base/agent/victim overlaps)
            if any(np.array_equal(pos, x) for x in forbidden):
                continue

            return pos

    def _build_obstacles(self):
        """
        Randomly place obstacles based on obstacle_ratio.
        WARNING: This can create disconnected regions (unsolvable episodes),
                 so we will verify reachability in reset().
        """
        n_cells = self.grid_size * self.grid_size
        n_obs = int(n_cells * self.obstacle_ratio)

        self.obstacles = set()
        forbidden = {tuple(self.base_pos.tolist())}  # never block the base

        while len(self.obstacles) < n_obs:
            x = int(self.np_random.integers(0, self.grid_size))
            y = int(self.np_random.integers(0, self.grid_size))
            if (x, y) not in forbidden:
                self.obstacles.add((x, y))

    # ============================
    # Reachability check (BFS)
    # ============================
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _is_reachable(self, start: np.ndarray, goal: np.ndarray) -> bool:
        """
        BFS on the grid to check if a path exists between start and goal
        given the obstacles. This prevents impossible episodes which can
        destabilize training.
        """
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

    # ============================
    # Observation + info helpers
    # ============================
    def _get_obs(self):
        """
        Flat numeric observation.

        Note:
          - If victim not found, relative victim vector is set to 0.
          - This makes it a partially observable problem; the agent must explore/scan.
        """
        if self.victim_found:
            rel = (self.victim_pos - self.agent_pos) / self.grid_size
            rel_vx, rel_vy = rel.astype(np.float32)
        else:
            rel_vx, rel_vy = 0.0, 0.0

        return np.array(
            [
                self.agent_pos[0] / self.grid_size,
                self.agent_pos[1] / self.grid_size,
                self.base_pos[0] / self.grid_size,
                self.base_pos[1] / self.grid_size,
                self.battery / self.max_battery,
                float(self.carrying_victim),
                float(self.victim_found),
                rel_vx,
                rel_vy,
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        """info dict is for logging/evaluation (not used by SB3 for learning)."""
        return {
            "steps": self.steps,
            "battery": self.battery,
            "victim_found": self.victim_found,
            "carrying_victim": self.carrying_victim,
            "success": bool(self.episode_success),
        }

    # ============================
    # Gymnasium API: reset/step
    # ============================
    def reset(self, seed=None, options=None):
        """
        Reset environment state at the start of each episode.

        Critical improvement:
          - We generate obstacles and positions until the episode is solvable:
              agent -> victim reachable AND victim -> base reachable
        """
        super().reset(seed=seed)

        # Optional: allow manual override via reset(options={...})
        # (SB3 typically doesn't pass options; this is mainly for your own testing.)
        if isinstance(options, dict) and options:
            self.set_difficulty(
                grid_size=options.get("grid_size", None),
                max_steps=options.get("max_steps", None),
                obstacle_ratio=options.get("obstacle_ratio", None),
                scan_radius=options.get("scan_radius", None),
                auto_discover=options.get("auto_discover", None),
            )

        # Reset counters/state
        self.steps = 0
        self.battery = self.max_battery
        self.victim_found = False
        self.carrying_victim = False
        self.episode_success = False

        # Keep regenerating until solvable (or fallback)
        for _ in range(self.max_reset_tries):
            self._build_obstacles()

            # Sample agent and victim in free cells
            self.agent_pos = self._random_free_cell(forbidden=[self.base_pos])
            self.victim_pos = self._random_free_cell(forbidden=[self.base_pos, self.agent_pos])

            # Ensure solvable episode
            if self._is_reachable(self.agent_pos, self.victim_pos) and self._is_reachable(self.victim_pos, self.base_pos):
                break
        else:
            # If repeated random obstacles keep producing unsolvable maps, fallback to empty obstacles.
            self.obstacles = set()
            self.agent_pos = self._random_free_cell(forbidden=[self.base_pos])
            self.victim_pos = self._random_free_cell(forbidden=[self.base_pos, self.agent_pos])

        # Visited grid used for exploration shaping reward
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.visited[self.agent_pos[0], self.agent_pos[1]] = 1

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Apply action and advance simulation by one step.
        """
        # Small negative reward each step to encourage efficiency
        reward = -0.05
        terminated = False   # "task completed" terminal
        truncated = False    # "time limit / battery depleted" terminal

        self.steps += 1
        self.battery -= 1

        # ----------------------------
        # Movement actions
        # ----------------------------
        if action in [0, 1, 2, 3]:
            move_map = {
                0: np.array([-1, 0]),  # up
                1: np.array([1, 0]),   # down
                2: np.array([0, -1]),  # left
                3: np.array([0, 1]),   # right
            }
            new_pos = self.agent_pos + move_map[action]

            # If move is valid (inside bounds and not an obstacle), update position.
            if (
                0 <= new_pos[0] < self.grid_size
                and 0 <= new_pos[1] < self.grid_size
                and tuple(new_pos.tolist()) not in self.obstacles
            ):
                self.agent_pos = new_pos

                # Exploration shaping: reward first-time visit more than revisits
                if self.visited[self.agent_pos[0], self.agent_pos[1]] == 0:
                    reward += 0.2
                self.visited[self.agent_pos[0], self.agent_pos[1]] += 1
            else:
                # Penalize bumping into a wall/obstacle (helps learning)
                reward -= 1.0

            # Optional "easy curriculum": stepping onto victim reveals it
            if self.auto_discover and (not self.victim_found) and np.array_equal(self.agent_pos, self.victim_pos):
                self.victim_found = True
                reward += 3.0

        # ----------------------------
        # Scan action
        # ----------------------------
        elif action == 4:
            # Manhattan distance to victim
            dist = np.abs(self.agent_pos - self.victim_pos).sum()

            # If within scan radius and victim not already found -> reveal victim
            if dist <= self.scan_radius and not self.victim_found:
                self.victim_found = True
                reward += 3.0
            else:
                # Slight penalty for wasting scan
                reward -= 0.2

        # ----------------------------
        # Rescue / Drop action
        # ----------------------------
        elif action == 5:
            # Pick up victim if you're at victim cell and it has been found
            if np.array_equal(self.agent_pos, self.victim_pos) and self.victim_found and not self.carrying_victim:
                self.carrying_victim = True
                reward += 5.0

            # Drop off at base if carrying
            elif np.array_equal(self.agent_pos, self.base_pos) and self.carrying_victim:
                reward += 20.0
                terminated = True
                self.episode_success = True
                self.carrying_victim = False

            else:
                # Penalize rescue/drop at wrong time/location
                reward -= 0.5

        # ----------------------------
        # Truncation: out of battery or max steps reached
        # ----------------------------
        if self.battery <= 0 or self.steps >= self.max_steps:
            truncated = True
            if not terminated:
                reward -= 10.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """
        Simple ASCII render for debugging.
        - A agent
        - B base
        - ? unknown victim location
        - V known victim location (if victim_found)
        - # obstacles
        """
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