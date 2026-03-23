import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, List

from background import *

class LogisticSchedulingWarehouseEnv(gym.Env):
    '''
        Task allocation + Scheduling:
        + The agent = Central Dispatcher
        + At decision point: assigns a resource (picker/forklift/AGV) to one candidate task
        + EVENT-DRIVEN simulation (act on next meaningful event/decision)

    '''

    metadata = {'render_modes': ['human']}

    TASK_TYPE_NAMES = {
        0: "pick",
        1: "replenish",
        2: "transfer",
        3: "pack",
        4: "load",
    }

    RESOURCE_TYPE_NAMES = {
        0: "picker",
        1: "forklift",
        2: "agv",
    }

    STATION_TYPE_NAMES = {
        0: "receiving",
        1: "storage",
        2: "packing",
        3: "staging",
        4: "dock",
    }

    TASK_ALLOWED_RESOURCE_TYPES = {
        0: (0,),      # pick -> picker only
        1: (1, 2),    # replenish -> forklift or AGV
        2: (1, 2),    # transfer -> forklift or AGV
        3: (0,),      # pack -> picker only
        4: (1,),      # load -> forklift only
    }

    def __init__(self,
                 max_tasks: int = 20, # Max open/candidate tasks in the observation
                 n_resources: int = 6, # Number of dispatchable resources in warehouse (x number of pickers, y number of forklifts, z number of AGVs, etc)
                 n_stations: int = 6, # Number of stations/zones in observation (receiving, storage, packing, staging, dock)
                 shift_duration: float = 480.0,# Max simulated warehouse time in one ep (480 min/ 8hr shift)
                 max_decisions: int = 300, # Max number of decisions/RL actions made per episode (prevent stucking in policy loop)
                 n_task_features: int = 9, # Number of features used -> each task
                 n_resource_features: int = 6, # Number of features used -> each resources
                 n_station_features: int = 3, # Number of features used -> each stations
                 min_task_release_gap: float = 0.0, # Min gap between successive task release times
                 invalid_action_penalty: float = 1.0, # Penalty applied for invalid action
                 decision_delay_if_idle_choice: float = 1.0, # If action = no_op + feasible assignment exist -> advance + charge idle time
                 ):
        
        super().__init__()

        if n_stations < 2:
            raise ValueError("n_stations must be at least 2")
        if max_tasks < 1 or n_resources < 1:
            raise ValueError("max_tasks and n_resources must both be >= 1")

        # Env dimensions
        self.max_tasks = max_tasks
        self.n_resources = n_resources
        self.n_stations = n_stations

        # Episode control parameters
        self.shift_duration = shift_duration
        self.max_decisions = max_decisions

        # Observation features
        self.n_task_features = n_task_features
        self.n_resource_features = n_resource_features
        self.n_station_features = n_station_features

        # Behavioural / reward-shaping parameters.
        self.min_task_release_gap = float(min_task_release_gap)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.decision_delay_if_idle_choice = float(decision_delay_if_idle_choice)

        # Action space
        # 1 action = assign 1 resource -> 1 task (r<i>, t<i>)
        # => flatten action = resource_idx * max_tasks + task_idx
        # NO_OP action = Idle/ Waiting
        self.no_op_action = n_resources * max_tasks
        self.action_space = spaces.Discrete(n_resources * max_tasks + 1) # All possible (resource, task) action pairs + 1 idle action
        
        # Observation space
        self.observation_space = spaces.Dict({
            'global': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(5,),
                dtype=np.float32
            ),
            'tasks': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_tasks, n_task_features),
                dtype=np.float32
            ),
            'resources': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_resources, n_resource_features),
                dtype=np.float32
            ),
            'stations': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_stations, n_station_features),
                dtype=np.float32
            ),
        })

        # Initialize env
        # Reset episode counters
        self.time = 0.0 # Current time
        self.decision_count = 0 # Current number of decisions made per episode
        
        self.travel_time_matrix = np.zeros((self.n_stations, self.n_stations), dtype=np.float32)
        
        # List of tasks, resources and stations used
        self.tasks: list[Task] = []
        self.resources: list[Resource]
        self.stations = self._init_stations()
        self.task_slot_to_id: list[Optional[int]] = [None] * self.max_tasks

        # Keep track of current status of the env
        self.total_completed = 0
        self.total_overdue = 0
        self.total_travel = 0.0
        self.total_idle = 0.0
        self.total_late_completions = 0
        self.last_reward_breakdown: dict[str, float] = {}

    def reset(self, seed=None, options=None):
        '''
            Reset and start new episode.

            Returns (obs: Dict, info: Dict)
        '''

        super().reset(seed=seed)
        self._init_env()

        # Move to the first meaningful decision point
        self.__advance_to_next_decision_point()

        obs = self._get_obs()
        info = {
            'action_mask': self.action_masks()
        }

        return obs, info

    def action_masks(self):
        '''
        Return a boolean mask over the action space.

        True  = valid/feasible action
        False = invalid action
        '''
        mask = np.zeros(self.action_space.n, dtype=bool)

        for r in range(self.n_resources):
            for t in range(self.max_tasks):
                if self._task_exists(t) and self._is_feasible(r, t):
                    mask[self._encode_action(r, t)] = True

        # NO-OP is usually kept valid so the agent can wait if needed
        mask[self.noop_action] = True
        return mask
        
    def _init_tasks(self):
        '''
            Create task.
        '''
        return []
    
    def _init_resources(self):
        '''
            Create resources.
        '''
        return []
    
    def _init_stations(self):
        '''
            Create stations.
        '''
        return []
  
    def _advance_to_next_decision_point(self):
        '''
        Move simulation to the first meaningful decision point.

        A decision point could be:
        - a task becomes available
        - a resource becomes free
        - a station unblocks
        '''
        pass
    
    def _get_obs(self):
        '''
        Build the observation dictionary.

        global:
            [current_time, backlog_count, overdue_count, idle_resources, mean_queue_len]

        tasks:
            one row per task, 
            [task_type, origin_zone, destination_zone, waiting_time, slack,
             service_time, priority, required_skill, assigned_flag]

        resources:
            one row per resource, 
            [resource_type, current_zone, busy_flag, remaining_busy_time,
             current_load, compatible_task_count]

        stations:
            one row per station, 
            [queue_length, utilization, blocked_flag]
        '''
        global_obs = np.zeros((5,), dtype=np.float32)
        task_obs = np.zeros((self.max_tasks, self.n_task_features), dtype=np.float32)
        resource_obs = np.zeros((self.n_resources, self.n_resource_features), dtype=np.float32)
        station_obs = np.zeros((self.n_stations, self.n_station_features), dtype=np.float32)

        return {
            'global': global_obs,
            'tasks': task_obs,
            'resources': resource_obs,
            'stations': station_obs,
        }



