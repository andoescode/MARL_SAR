import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class SAREnv(gym.Env):
    '''
        Search and Rescue Environment for single agent 
        to explore, find and retrieve.
    '''
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    ACTIONS = {
        'up': 0,
        'down': 1,
        'left': 2,
        'right': 3,
        'scan': 4,
        'rescue_dropoff': 5,
    }

    ID_TO_ACTION = {v: k for k, v in ACTIONS.items()}

    REWARDS = {
    'step': -0.05, # Track efficiency (the less step the better)
    'new_grid': 0.2, # Exploration (new grid visit)
    'bump': -1.0, # Bump into obstacles or walls
    'scan_success': 3.0, # Scan find victim
    'scan_fail': -0.2, # Scan not find victim
    'pickup': 5.0, # Successful found and pick up victim
    'dropoff': 20.0, # Successful drop off
    'bad_rescue': -0.5, # Wrong pickup/ drop off attempts
    'timeout': -10.0, # Take more than the max steps
    # Shaping (start with 0.0, then enable later)
    'dist_shaping': 0.0, # Relative distance from agent when navigating to victim and back to base 
    }
    def __init__(self, 
                 grid_size: int = 10, # Size of grid env
                 max_steps: int = 200, # Max steps agent can make. Outloop if steps >= max_steps
                 obstacle_ratio: float = 0.15, # Rate of obstacles in env
                 render_mode: str | None = None,
                 scan_radius: int = 1, # Radius of the agent scanning (visibility)
                 auto_discover: bool = False, # Agent found victim on the same grid
                 max_reset_tries: int = 200, # 
                 ):
        super().__init__()

        # Core
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.obstacle_ratio = obstacle_ratio
        self.render_mode = render_mode

        # Curriculum/ Difficulty knobs
        self.scan_radius = scan_radius # Manhattan distance threshold for successful scan
        self.auto_discover = auto_discover # if True, stepping on victim grid reveals it automatically (easier Stage 0)

        self.max_reset_tries = max_reset_tries # If keep making unsolvable maps >= max_reset_tries times -> go back to obstacle free

        # Action: 0:up, 1:down, 2:left, 3:right, 4:scan, 5:rescue/drop
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # 1D obs, normalise if possible
        # [agent_x, agent_y, base_x, base_y, battery_norm, carrying_victim_flag, victim_found_flag, rel_victim_x, rel_victim_y] 
        # rel_victim_ = value if victim_found else 0s
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )

        self.base_pos = np.array([0,0], dtype=np.int32) # Fixed base location, (0,0) for now

        # Init state
        self.agent_pos = None # Position of agent
        self.victim_pos = None # Position of victim (one victim for now)
        self.obstacles: set[tuple[int, int]] = set() # Locations of obstacles

        # Current status of agent
        self.victim_found = False
        self.carrying_victim = False
        self.episode_success = False
        self.steps = 0
        self.battery = None

        self.max_battery = self.max_steps # Battery = step budget, normlaised in obs

        self.visited = None # Visited count grid (encourage exploration)

    # Curriculum learning helper (Change difficulty)
    def set_difficulty(self,
                       *,
                       grid_size: int | None = None,
                       max_steps: int | None = None,
                       obstacle_ratio: float | None = None,
                       scan_radius: int | None = None,
                       auto_discover: bool | None = None,
                       ):
        
        '''
            Set the difficulty of the map by changing grid_size, max_steps, obstacle_ratio, scan_radius and auto_discover.
        '''
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

    def _place_obstacles(self):
        '''
            Place obstacles at random places across the map.
        '''

        n_grids = self.grid_size * self.grid_size
        n_obstacles = n_grids * self.obstacle_ratio
        self.obstacles = set()
        forbidden = {tuple(self.base_pos.tolist())} # Dont block base

        while len(self.obstacles) < n_obstacles:
            x = self.np_random.integers(0, self.grid_size)
            y = self.np_random.integers(0, self.grid_size)

            if (x, y) not in forbidden:
                self.obstacles.add((x, y))

    # Reset
    def reset(self, *, seed=None, options=None):
        '''
            Reset environment state at the start of each episode.

            Generate the positions of obstacles until the episode is solvable that:

            + agent -> victim: reachable

            + victim -> base: reachable
        '''

        super().reset(seed=seed)

        # Testing for difficulty
        if isinstance(options, dict) and options:
            self.set_difficulty(
                grid_size=options.get('grid_size', None),
                max_steps=options.get('max_steps', None),
                obstacle_ratio=options.get('obstacle_ratio', None),
                scan_radius=options.get('scan_radius', None),
                auto_discover=options.get('auto_discover', None),
            )

        self.steps = 0
        self.battery = self.max_battery
        self.victim_found = False
        self.carrying_victim = False
        self.episode_success = False

        self.np_random = np.random.default_rng(seed)

        # Keep regenerating untill the episode is solvable
        self._generate_init_pos()

        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.visited[self.agent_pos[0], self.agent_pos[1]] = 1
        
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    # Check if reachable (there is at least 1 solution in the map)
    def _in_bounds(self, x: int, y: int) -> bool:
        '''
            Return True if (x, y) within the map.
        '''
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def _is_reachable(self, start: np.ndarray, goal: np.ndarray) -> bool:
        '''
            BFS Check if path exist.
        '''

        s = (int(start[0]), int(start[1]))
        g = (int(goal[0]), int(goal[1]))

        if s == g: return True
        if s in self.obstacles or g in self.obstacles: return False

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

    # Set up starting positions for obstacles, victim and agent 
    def _generate_init_pos(self):
        '''
            Generate and choose the initial positions for obstacles, victim and agent 
            where path navigation is possible.
        '''

        # Loop for desired map until reach limit
        # If reached limit -> generating map without obstacles
        for i in range(self.max_reset_tries):
            self._place_obstacles()

            # Sample agent + victim
            self.agent_pos = self._random_free_grid(forbidden=[self.base_pos])
            self.victim_pos = self._random_free_grid(forbidden=[self.base_pos, self.agent_pos])

            # Check solvable episode
            if self._is_reachable(start=self.agent_pos, goal=self.victim_pos)\
            and self._is_reachable(start=self.victim_pos, goal=self.base_pos):
                break
        else:
            # Regenerate again with 0 obstacle
            self.obstacles = set()
            self.agent_pos = self._random_free_cell(forbidden=[self.base_pos])
            self.victim_pos = self._random_free_cell(forbidden=[self.base_pos, self.agent_pos])      

        return self.agent_pos, self.victim_pos, self.obstacles  
    
    def _random_free_grid(self, forbidden= None):
        '''
            Get random free grid in map (not occupied by obstacles and not in forbidden list).
        '''

        if forbidden:
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

            if pos_tuple in self.obstacles: # not obstacle
                continue

            if any(np.array_equal(pos, x) for x in forbidden): # not forbidden
                continue

            return pos

    # Get obs and info
    def _get_obs(self):
        '''
            Flat numeric observation. 

            Return normalised values for observation.
        '''
        # Only update (rel_victim_x, rel_victim_y) if victim_found = True
        # else set (0, 0)
        if self.victim_found:
            rel = abs(self.victim_pos - self.agent_pos) / self.grid_size
            rel_victim_x, rel_victim_y = rel.astype(np.float32)
        else:
            rel_victim_x, rel_victim_y = 0.0, 0.0

        return np.array(
            [
                self.agent_pos[0] / self.grid_size, # agent_x
                self.agent_pos[1] / self.grid_size, # agent_y
                self.base_pos[0] / self.grid_size, # base_x
                self.base_pos[1] / self.grid_size, # base_y
                self.battery / self.max_battery, # battery_norm
                float(self.carrying_victim), # carrying_victim_flag
                float(self.victim_found), # Victim_found_flag
                rel_victim_x, # rel_victim_x
                rel_victim_y, # rel_victim_y
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        '''
            Info dict for logging and evaluation.
        '''

        return {
            'steps': self.steps,
            'battery': self.battery,
            'victim_found': self.victim_found,
            'carrying_victim': self.carrying_victim,
            'success': bool(self.episode_success),
        }

    # Step
    def step(self, action):
        '''
            Update the next state/ status of the environment based on the given action.
        '''
        if isinstance(action, str):
            if action not in self.ACTIONS:
                raise ValueError(f"Unknown action name: {action}")
            action_id = self.ACTIONS[action]
        else:
            action_id = int(action)
        # Reward components dictionary (for debugging + future multi-agent)
        r = {k: 0.0 for k in self.REWARDS.keys()}
        terminated = False
        truncated = False

        # Always apply step cost
        r['step'] += self.REWARDS['step']

        self.steps += 1
        self.battery -= 1

        old_pos = self.agent_pos.copy()

        # movement 
        if action_id in [self.ACTIONS['up'], self.ACTIONS['down'], self.ACTIONS['left'], self.ACTIONS['right']]:
            move_map = {
                0: np.array([-1, 0]),
                1: np.array([1, 0]),
                2: np.array([0, -1]),
                3: np.array([0, 1]),
            }
            new_pos = self.agent_pos + move_map[action]

            if (
                0 <= new_pos[0] < self.grid_size and
                0 <= new_pos[1] < self.grid_size and
                tuple(new_pos.tolist()) not in self.obstacles
            ):
                self.agent_pos = new_pos

                # Exploration bonus: first visit
                if self.visited[self.agent_pos[0], self.agent_pos[1]] == 0:
                    r['new_grid'] += self.REWARDS['new_grid']

                self.visited[self.agent_pos[0], self.agent_pos[1]] += 1
            else:
                # Hit wall/obstacle
                r['bump'] += self.REWARDS['bump']

            # Auto-discover if you use it in curriculum Stage 0
            if getattr(self, 'auto_discover', False) and (not self.victim_found) and np.array_equal(self.agent_pos, self.victim_pos):
                self.victim_found = True
                r['scan_success'] += self.REWARDS['scan_success']

        # scan 
        elif action == self.ACTIONS['scan']:
            dist = np.abs(self.agent_pos - self.victim_pos).sum()
            scan_radius = getattr(self, 'scan_radius', 1)

            if dist <= scan_radius and not self.victim_found:
                self.victim_found = True
                r['scan_success'] += self.REWARDS['scan_success']
            else:
                r['scan_fail'] += self.REWARDS['scan_fail']

        # rescue/drop 
        elif action == self.ACTIONS['rescue_dropoff']:
            if np.array_equal(self.agent_pos, self.victim_pos) and self.victim_found and not self.carrying_victim:
                self.carrying_victim = True
                r['pickup'] += self.REWARDS['pickup']

            elif np.array_equal(self.agent_pos, self.base_pos) and self.carrying_victim:
                r['dropoff'] += self.REWARDS['dropoff']
                self.carrying_victim = False
                terminated = True

            else:
                r['bad_rescue'] += self.REWARDS['bad_rescue']

        # Distance shaping (enable later) 
        if self.REWARDS['dist_shaping'] != 0.0:
            # Decide current goal:
            # + If carrying => goal is base
            # + Else if victim found => goal is victim
            # + Else no goal shaping (no leak victim location)
            goal = None
            if self.carrying_victim:
                goal = self.base_pos
            elif self.victim_found:
                goal = self.victim_pos

            if goal is not None:
                old_d = np.abs(old_pos - goal).sum()
                new_d = np.abs(self.agent_pos - goal).sum()
                # Reward improvement (getting closer)
                r['dist_shaping'] += self.REWARDS['dist_shaping'] * (old_d - new_d)

        # truncation 
        if self.battery <= 0 or self.steps >= self.max_steps:
            truncated = True
            if not terminated:
                r['timeout'] += self.REWARDS['timeout']

        # Flatten reward
        reward = float(sum(r.values()))

        obs = self._get_obs()
        info = self._get_info()

        # Put reward breakdown into info for debugging/plots
        info['reward_parts'] = r
        info['success'] = bool(terminated) and ('dropoff' in r and r['dropoff'] > 0)

        return obs, reward, terminated, truncated, info

    def render(self):
        '''
        Simple ASCII render for debugging.
        + A = agent
        + B = base
        + ? = unknown victim location
        + V = known victim location (if victim_found)
        + # = obstacles
        '''
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=object)

        for (x, y) in self.obstacles:
            grid[x, y] = '#'

        bx, by = self.base_pos
        grid[bx, by] = 'B'

        # If carrying, hide victim marker; otheREWARDSise show V if found, ? if unknown
        if not self.carrying_victim:
            vx, vy = self.victim_pos
            grid[vx, vy] = 'V' if self.victim_found else '?'

        ax, ay = self.agent_pos
        grid[ax, ay] = 'A'

        print('\n'.join(' '.join(row) for row in grid))
        print()