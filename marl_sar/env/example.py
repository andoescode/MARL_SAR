import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque, defaultdict
from enum import Enum, IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    SCAN = 4
    RESCUE_DROPOFF = 5

    @property
    def key(self) -> str:
        return self.name.lower()


class Reward(float, Enum):
    STEP = -0.05
    NEW_GRID = 0.2
    BUMP = -1.0
    SCAN_SUCCESS = 3.0
    SCAN_FAIL = -0.2
    PICKUP = 5.0
    DROPOFF = 20.0
    BAD_RESCUE = -0.5
    OUT_OF_BATTERY = -10.0
    DIST_SHAPING = 0.0
    REVISIT = 0.0
    GLOBAL_REVISIT = 0.0

    @property
    def key(self) -> str:
        return self.name.lower()


class Status(Enum):
    MOVE = (1, "MOVE")
    SCAN = (2, "SCAN")
    RESCUE = (3, "RESCUE")
    DROP_OFF = (4, "DROP_OFF")
    OUT_OF_BATTERY = (5, "OUT_OF_BATTERY")
    IDLE = (6, "IDLE")

    def __init__(self, sid: int, label: str):
        self.sid = sid
        self.label = label


class SAREnv(gym.Env):
    """
    Search and Rescue Environment for single agent 
    to explore, find and retrieve.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    ACTIONS = {a.key: int(a.value) for a in Action}
    ID_TO_ACTION = {int(a.value): a.key for a in Action}
    REWARDS = {r.key: float(r.value) for r in Reward}

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 300,
        obstacle_ratio: float = 0.15,
        n_victims: int = 1,
        scan_radius: int = 3,
        auto_discover: bool = False,
        max_reset_tries: int = 200,
        render_mode: str | None = None,
        reveal_victims_in_render: bool = True,
        # battery model
        battery_max: float = 1.0,
        drain_per_5_actions: float = 0.01,
        carry_extra_drain_per_5: float = 0.02,
        enable_spare_battery: bool | None = None,
        spare_battery_max: float = 1.0,
    ):
        super().__init__()

        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.obstacle_ratio = float(obstacle_ratio)
        self.scan_radius = int(scan_radius)
        self.auto_discover = bool(auto_discover)
        self.max_reset_tries = int(max_reset_tries)
        self.render_mode = render_mode
        self.reveal_victims_in_render = bool(reveal_victims_in_render)

        # victim count constraint: <=10% cells
        max_allowed = max(1, int(0.10 * self.grid_size * self.grid_size))
        if not (1 <= n_victims <= max_allowed):
            raise ValueError(f"n_victims must be in [1, {max_allowed}] for grid_size={self.grid_size}")
        self.n_victims = int(n_victims)

        # battery model
        self.battery_max = float(battery_max)
        self.drain_per_5_actions = float(drain_per_5_actions)
        self.carry_extra_drain_per_5 = float(carry_extra_drain_per_5)

        if enable_spare_battery is None:
            enable_spare_battery = (self.max_steps >= 500) or (self.grid_size * self.grid_size >= 400)
        self.enable_spare_battery = bool(enable_spare_battery)
        self.spare_battery_max = float(spare_battery_max)

        self.base_pos = np.array([0, 0], dtype=np.int32)

        # Action space now uses the Enum count
        self.action_space = spaces.Discrete(len(Action))

        # Simple flat obs (same as before)
        low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1/6, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0,  1,  1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Episode state
        self.obstacles: set[tuple[int, int]] = set()
        self.agent_pos = self.base_pos.copy()

        self.victim_pos: list[np.ndarray] = []
        self.victim_visible: list[bool] = []
        self.victim_delivered: list[bool] = []
        self.carrying = False
        self.carrying_idx: int | None = None

        self.status = Status.IDLE
        self.last_action_id: int | None = None

        self.steps = 0
        self.total_actions = 0
        self.battery = self.battery_max
        self.spare_battery = (self.spare_battery_max if self.enable_spare_battery else 0.0)
        self.episode_success = False

        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.global_visit_count = defaultdict(int)
        self.dist_from_base = None
        self._oob_penalized = False

    # -----------------------
    # Helpers: map generation
    # -----------------------
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _place_obstacles(self):
        n_cells = self.grid_size * self.grid_size
        n_obstacles = int(n_cells * self.obstacle_ratio)
        n_obstacles = min(n_obstacles, n_cells - 2)

        self.obstacles = set()
        forbidden = {tuple(self.base_pos.tolist())}

        while len(self.obstacles) < n_obstacles:
            x = int(self.np_random.integers(0, self.grid_size))
            y = int(self.np_random.integers(0, self.grid_size))
            if (x, y) not in forbidden:
                self.obstacles.add((x, y))

    def _random_free_cell(self, forbidden: list[np.ndarray] | None = None) -> np.ndarray:
        if forbidden is None:
            forbidden = []
        while True:
            pos = np.array(
                [int(self.np_random.integers(0, self.grid_size)),
                 int(self.np_random.integers(0, self.grid_size))],
                dtype=np.int32,
            )
            if tuple(pos.tolist()) in self.obstacles:
                continue
            if any(np.array_equal(pos, x) for x in forbidden):
                continue
            return pos

    def _bfs_reachable(self, start: np.ndarray, goal: np.ndarray) -> bool:
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
                if (nx, ny) in self.obstacles or (nx, ny) in seen:
                    continue
                if (nx, ny) == g:
                    return True
                seen.add((nx, ny))
                q.append((nx, ny))
        return False

    def _bfs_distance_map(self, src: np.ndarray) -> np.ndarray:
        dist = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)
        sx, sy = int(src[0]), int(src[1])
        if (sx, sy) in self.obstacles:
            return dist
        q = deque([(sx, sy)])
        dist[sx, sy] = 0
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if not self._in_bounds(nx, ny):
                    continue
                if (nx, ny) in self.obstacles:
                    continue
                if dist[nx, ny] != -1:
                    continue
                dist[nx, ny] = dist[x, y] + 1
                q.append((nx, ny))
        return dist

    def _generate_solvable_episode(self):
        for _ in range(self.max_reset_tries):
            self._place_obstacles()
            self.agent_pos = self.base_pos.copy()

            self.victim_pos = []
            self.victim_visible = []
            self.victim_delivered = []

            forbidden = [self.base_pos, self.agent_pos]
            used = {tuple(self.base_pos.tolist())}

            for _k in range(self.n_victims):
                for _try in range(200):
                    vp = self._random_free_cell(forbidden=forbidden)
                    t = tuple(vp.tolist())
                    if t in used:
                        continue
                    used.add(t)
                    forbidden.append(vp)
                    self.victim_pos.append(vp)
                    self.victim_visible.append(False)
                    self.victim_delivered.append(False)
                    break

            if all(self._bfs_reachable(self.base_pos, vp) for vp in self.victim_pos):
                self.dist_from_base = self._bfs_distance_map(self.base_pos)
                return

        # fallback obstacle-free
        self.obstacles = set()
        self.agent_pos = self.base_pos.copy()
        self.victim_pos = []
        self.victim_visible = []
        self.victim_delivered = []
        forbidden = [self.base_pos]
        for _k in range(self.n_victims):
            vp = self._random_free_cell(forbidden=forbidden)
            forbidden.append(vp)
            self.victim_pos.append(vp)
            self.victim_visible.append(False)
            self.victim_delivered.append(False)
        self.dist_from_base = self._bfs_distance_map(self.base_pos)

    # -----------------------
    # Scan LOS
    # -----------------------
    def _scan_cells_los(self) -> set[tuple[int, int]]:
        dirs = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        seen = set()
        for dx, dy in dirs:
            x, y = ax, ay
            for _ in range(self.scan_radius):
                x += dx
                y += dy
                if not self._in_bounds(x, y):
                    break
                if (x, y) in self.obstacles:
                    break
                seen.add((x, y))
        return seen

    # -----------------------
    # Battery mechanics
    # -----------------------
    def _maybe_drain_battery(self, rparts: dict):
        self.total_actions += 1
        if self.total_actions % 5 != 0:
            return

        drain = self.drain_per_5_actions + (self.carry_extra_drain_per_5 if self.carrying else 0.0)
        self.battery = max(0.0, self.battery - drain)

        if self.battery <= 0.0 and not np.array_equal(self.agent_pos, self.base_pos):
            if self.enable_spare_battery and self.spare_battery > 0.0:
                self.battery = min(self.battery_max, self.spare_battery)
                self.spare_battery = 0.0
                self._oob_penalized = False
            else:
                self.status = Status.OUT_OF_BATTERY
                if not self._oob_penalized:
                    rparts[Reward.OUT_OF_BATTERY.key] += float(Reward.OUT_OF_BATTERY.value)
                    self._oob_penalized = True

    def _maybe_recharge_at_base(self):
        if np.array_equal(self.agent_pos, self.base_pos):
            self.battery = self.battery_max
            self._oob_penalized = False
            if self.enable_spare_battery:
                self.spare_battery = self.spare_battery_max

    # -----------------------
    # Observations / info
    # -----------------------
    def _current_goal(self) -> np.ndarray | None:
        if self.carrying:
            return self.base_pos

        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        best = None
        best_d = None
        for i, vp in enumerate(self.victim_pos):
            if self.victim_delivered[i] or not self.victim_visible[i]:
                continue
            d = abs(int(vp[0]) - ax) + abs(int(vp[1]) - ay)
            if best is None or d < best_d:
                best, best_d = vp, d
        return best

    def _get_obs(self) -> np.ndarray:
        denom = max(self.grid_size - 1, 1)

        agent_x = float(self.agent_pos[0]) / denom
        agent_y = float(self.agent_pos[1]) / denom

        battery = float(self.battery / self.battery_max) if self.battery_max > 0 else 0.0
        spare = float(self.spare_battery / self.spare_battery_max) if (self.enable_spare_battery and self.spare_battery_max > 0) else 0.0

        carrying = float(self.carrying)
        remaining = sum(1 for d in self.victim_delivered if not d)
        remaining_norm = float(remaining / max(self.n_victims, 1))
        any_visible = float(any(self.victim_visible[i] and not self.victim_delivered[i] for i in range(self.n_victims)))

        d_base = self.dist_from_base[int(self.agent_pos[0]), int(self.agent_pos[1])]
        dist_to_base_norm = 1.0 if d_base < 0 else float(d_base / (2 * denom + 1e-6))

        goal = self._current_goal()
        if goal is not None:
            d_goal = abs(int(goal[0]) - int(self.agent_pos[0])) + abs(int(goal[1]) - int(self.agent_pos[1]))
            dist_to_goal_norm = float(d_goal / (2 * denom + 1e-6))
            rel = (goal - self.agent_pos) / denom
            rel_goal_x = float(np.clip(rel[0], -1.0, 1.0))
            rel_goal_y = float(np.clip(rel[1], -1.0, 1.0))
        else:
            dist_to_goal_norm = 0.0
            rel_goal_x = 0.0
            rel_goal_y = 0.0

        status_norm = float(self.status.sid / 6.0)

        return np.array(
            [
                agent_x, agent_y,
                battery, spare,
                carrying,
                remaining_norm,
                any_visible,
                dist_to_base_norm,
                dist_to_goal_norm,
                status_norm,
                rel_goal_x, rel_goal_y,
            ],
            dtype=np.float32,
        )

    def _coverage_redundancy(self):
        total_cells = self.grid_size * self.grid_size
        coverage = len(self.global_visit_count) / total_cells if total_cells > 0 else 0.0
        redundancy = sum(1 for v in self.global_visit_count.values() if v > 1)
        return float(coverage), int(redundancy)

    def _get_info(self) -> dict:
        coverage, redundancy = self._coverage_redundancy()
        return {
            "steps": self.steps,
            "status": self.status.label,
            "status_id": self.status.sid,
            "battery": float(self.battery),
            "spare_battery": float(self.spare_battery),
            "carrying": bool(self.carrying),
            "victims_total": int(self.n_victims),
            "victims_delivered": int(sum(self.victim_delivered)),
            "success": bool(self.episode_success),
            "coverage": coverage,
            "redundancy": redundancy,
        }

    # -----------------------
    # Gym API
    # -----------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.total_actions = 0
        self.battery = self.battery_max
        self.spare_battery = (self.spare_battery_max if self.enable_spare_battery else 0.0)
        self._oob_penalized = False

        self.carrying = False
        self.carrying_idx = None
        self.episode_success = False
        self.status = Status.IDLE
        self.last_action_id = None

        self._generate_solvable_episode()

        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.visited[int(self.agent_pos[0]), int(self.agent_pos[1])] = 1
        self.global_visit_count = defaultdict(int)
        self.global_visit_count[tuple(self.agent_pos.tolist())] += 1

        return self._get_obs(), self._get_info()

    def step(self, action):
        # --- normalize / validate action ---
        if isinstance(action, str):
            action_id = self.ACTIONS[action]
        else:
            action_id = int(action)

        # Validate: will raise ValueError if invalid id (good for debugging)
        act = Action(action_id)
        self.last_action_id = action_id

        # reward parts keyed by Reward enum keys ("step", "new_grid", ...)
        r = {rw.key: 0.0 for rw in Reward}
        r[Reward.STEP.key] += float(Reward.STEP.value)

        terminated = False
        truncated = False

        self.steps += 1

        old_pos = self.agent_pos.copy()
        old_goal = self._current_goal()

        # If out of battery away from base, block scan/rescue (encourages returning)
        if self.battery <= 0.0 and not np.array_equal(self.agent_pos, self.base_pos):
            if act in (Action.SCAN, Action.RESCUE_DROPOFF):
                r[Reward.BAD_RESCUE.key] += float(Reward.BAD_RESCUE.value)
                self.status = Status.OUT_OF_BATTERY

                self._maybe_drain_battery(r)
                self._maybe_recharge_at_base()

                obs = self._get_obs()
                info = self._get_info()
                info["action_id"] = int(act.value)
                info["action_name"] = act.key
                info["reward_parts"] = r
                return obs, float(sum(r.values())), False, False, info

        # --- MOVE ---
        if act in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            move_map = {
                Action.UP: np.array([-1, 0], dtype=np.int32),
                Action.DOWN: np.array([1, 0], dtype=np.int32),
                Action.LEFT: np.array([0, -1], dtype=np.int32),
                Action.RIGHT: np.array([0, 1], dtype=np.int32),
            }
            new_pos = self.agent_pos + move_map[act]

            if (
                0 <= new_pos[0] < self.grid_size
                and 0 <= new_pos[1] < self.grid_size
                and tuple(new_pos.tolist()) not in self.obstacles
            ):
                self.agent_pos = new_pos
                self.status = Status.MOVE

                # exploration bonus
                if self.visited[int(self.agent_pos[0]), int(self.agent_pos[1])] == 0:
                    r[Reward.NEW_GRID.key] += float(Reward.NEW_GRID.value)
                else:
                    r[Reward.REVISIT.key] += float(Reward.REVISIT.value)

                self.visited[int(self.agent_pos[0]), int(self.agent_pos[1])] += 1
            else:
                r[Reward.BUMP.key] += float(Reward.BUMP.value)
                self.status = Status.IDLE

            # optional auto-discover (curriculum)
            if self.auto_discover:
                for i, vp in enumerate(self.victim_pos):
                    if self.victim_delivered[i]:
                        continue
                    if np.array_equal(self.agent_pos, vp) and not self.victim_visible[i]:
                        self.victim_visible[i] = True
                        r[Reward.SCAN_SUCCESS.key] += float(Reward.SCAN_SUCCESS.value)

        # --- SCAN (LOS) ---
        elif act == Action.SCAN:
            self.status = Status.SCAN
            visible_cells = self._scan_cells_los()

            newly_found = 0
            for i, vp in enumerate(self.victim_pos):
                if self.victim_delivered[i] or self.victim_visible[i]:
                    continue
                if tuple(vp.tolist()) in visible_cells:
                    self.victim_visible[i] = True
                    newly_found += 1

            if newly_found > 0:
                r[Reward.SCAN_SUCCESS.key] += float(Reward.SCAN_SUCCESS.value) * newly_found
            else:
                r[Reward.SCAN_FAIL.key] += float(Reward.SCAN_FAIL.value)

            self.visited[int(self.agent_pos[0]), int(self.agent_pos[1])] += 1

        # --- RESCUE / DROPOFF ---
        elif act == Action.RESCUE_DROPOFF:
            if not self.carrying:
                picked = False
                for i, vp in enumerate(self.victim_pos):
                    if self.victim_delivered[i]:
                        continue
                    if np.array_equal(self.agent_pos, vp) and self.victim_visible[i]:
                        self.carrying = True
                        self.carrying_idx = i
                        picked = True
                        r[Reward.PICKUP.key] += float(Reward.PICKUP.value)
                        self.status = Status.RESCUE
                        break
                if not picked:
                    r[Reward.BAD_RESCUE.key] += float(Reward.BAD_RESCUE.value)
                    self.status = Status.IDLE
            else:
                if np.array_equal(self.agent_pos, self.base_pos):
                    idx = int(self.carrying_idx)
                    self.victim_delivered[idx] = True
                    self.carrying = False
                    self.carrying_idx = None

                    r[Reward.DROPOFF.key] += float(Reward.DROPOFF.value)
                    self.status = Status.DROP_OFF

                    if all(self.victim_delivered):
                        self.episode_success = True
                        terminated = True
                else:
                    r[Reward.BAD_RESCUE.key] += float(Reward.BAD_RESCUE.value)
                    self.status = Status.IDLE

            self.visited[int(self.agent_pos[0]), int(self.agent_pos[1])] += 1

        # --- optional dist shaping ---
        if float(Reward.DIST_SHAPING.value) != 0.0 and old_goal is not None:
            new_goal = self._current_goal() or old_goal
            old_d = abs(int(old_pos[0]) - int(old_goal[0])) + abs(int(old_pos[1]) - int(old_goal[1]))
            new_d = abs(int(self.agent_pos[0]) - int(new_goal[0])) + abs(int(self.agent_pos[1]) - int(new_goal[1]))
            r[Reward.DIST_SHAPING.key] += float(Reward.DIST_SHAPING.value) * (old_d - new_d)

        # --- global revisit ---
        pos_t = tuple(self.agent_pos.tolist())
        if self.global_visit_count[pos_t] > 0:
            r[Reward.GLOBAL_REVISIT.key] += float(Reward.GLOBAL_REVISIT.value)
        self.global_visit_count[pos_t] += 1

        # --- battery updates ---
        self._maybe_drain_battery(r)
        self._maybe_recharge_at_base()

        # --- episode cap / failure penalty ---
        if self.steps >= self.max_steps and not terminated:
            truncated = True
            r[Reward.OUT_OF_BATTERY.key] += float(Reward.OUT_OF_BATTERY.value)

        reward = float(sum(r.values()))
        obs = self._get_obs()
        info = self._get_info()
        info["action_id"] = int(act.value)
        info["action_name"] = act.key
        info["reward_parts"] = r

        return obs, reward, terminated, truncated, info

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=object)

        for (x, y) in self.obstacles:
            grid[x, y] = "#"

        bx, by = int(self.base_pos[0]), int(self.base_pos[1])
        grid[bx, by] = "B"

        for i, vp in enumerate(self.victim_pos):
            if self.victim_delivered[i]:
                continue
            vx, vy = int(vp[0]), int(vp[1])
            if self.victim_visible[i]:
                grid[vx, vy] = "V"
            else:
                if self.reveal_victims_in_render:
                    grid[vx, vy] = "?"

        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        grid[ax, ay] = "A"

        print("\n".join(" ".join(row) for row in grid))
        print(f"status={self.status.label} battery={self.battery:.2f} spare={self.spare_battery:.2f} "
              f"delivered={sum(self.victim_delivered)}/{self.n_victims}")
        print()
