from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class Task:
    """Represents one warehouse job that may be dispatched to a resource."""

    id: int
    task_type: int
    origin_zone: int
    destination_zone: int
    release_time: float
    due_time: float
    service_time: float
    priority: int
    required_skill: int
    status: str = "hidden"  # hidden -> waiting -> in_progress -> done
    assigned_resource: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None


@dataclass
class Resource:
    """Represents a dispatchable warehouse resource such as a picker or AGV."""

    id: int
    resource_type: int
    current_zone: int
    busy_until: float = 0.0
    assigned_task_id: Optional[int] = None
    current_load: int = 0

    @property
    def busy(self) -> bool:
        return self.assigned_task_id is not None


@dataclass
class Station:
    """Represents a warehouse zone/station with limited downstream capacity."""

    id: int
    station_type: int
    capacity: int
    queue_length: int = 0

    @property
    def blocked(self) -> bool:
        return self.queue_length >= self.capacity

    @property
    def utilization(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return min(1.0, self.queue_length / self.capacity)


class WarehouseDispatchEnv(gym.Env):
    """
    Simplified event-driven warehouse dispatch environment.

    What this environment models
    ----------------------------
    - Tasks are released over simulated time.
    - Resources (pickers / forklifts / AGVs) can be assigned to waiting tasks.
    - Travel from the resource's current zone to the task origin matters.
    - Travel from task origin to task destination matters.
    - Destination stations have limited capacity and can block assignments.
    - The environment only jumps between meaningful events, not every second.

    One RL action = assign one resource to one currently visible task.
    The agent acts as a central dispatcher.
    """

    metadata = {"render_modes": ["human"]}

    # Simple label maps used only for debugging / rendering.
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

    def __init__(
        self,
        max_tasks: int = 20,
        n_resources: int = 6,
        n_stations: int = 5,
        shift_duration: float = 480.0,
        max_decisions: int = 300,
        n_task_features: int = 9,
        n_resource_features: int = 6,
        n_station_features: int = 3,
        min_task_release_gap: float = 0.0,
        invalid_action_penalty: float = 1.0,
        decision_delay_if_idle_choice: float = 1.0,
    ):
        """
        Parameters
        ----------
        max_tasks:
            Maximum number of visible task slots shown in the observation and
            used in the flattened action mapping.

        n_resources:
            Number of warehouse resources controlled by the dispatcher.

        n_stations:
            Number of warehouse stations/zones. Keep this >= 2.

        shift_duration:
            Simulated duration of one warehouse episode, in minutes.

        max_decisions:
            Maximum number of RL decisions allowed in one episode. This is a
            safety cap separate from the simulated shift duration.

        n_task_features / n_resource_features / n_station_features:
            Fixed observation table widths.

        min_task_release_gap:
            Optional minimum gap between successive task release times.

        invalid_action_penalty:
            Penalty applied when the policy selects an invalid action.

        decision_delay_if_idle_choice:
            If the agent chooses NO-OP while feasible assignments exist, the
            simulator advances by this many minutes and charges idle time.
        """
        super().__init__()

        if n_stations < 2:
            raise ValueError("n_stations must be at least 2")
        if max_tasks < 1 or n_resources < 1:
            raise ValueError("max_tasks and n_resources must both be >= 1")

        # Core environment sizes.
        self.max_tasks = max_tasks
        self.n_resources = n_resources
        self.n_stations = n_stations

        # Episode control.
        self.shift_duration = float(shift_duration)
        self.max_decisions = int(max_decisions)

        # Observation table widths.
        self.n_task_features = n_task_features
        self.n_resource_features = n_resource_features
        self.n_station_features = n_station_features

        # Behavioural / reward-shaping parameters.
        self.min_task_release_gap = float(min_task_release_gap)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.decision_delay_if_idle_choice = float(decision_delay_if_idle_choice)

        # Action flattening:
        # action = resource_idx * max_tasks + task_slot_idx
        # plus one extra action for NO-OP.
        self.noop_action = n_resources * max_tasks
        self.action_space = spaces.Discrete(n_resources * max_tasks + 1)

        # Structured observation.
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(5,),
                    dtype=np.float32,
                ),
                "tasks": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max_tasks, n_task_features),
                    dtype=np.float32,
                ),
                "resources": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_resources, n_resource_features),
                    dtype=np.float32,
                ),
                "stations": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_stations, n_station_features),
                    dtype=np.float32,
                ),
            }
        )

        # State containers populated in reset().
        self.time = 0.0
        self.decision_count = 0
        self.travel_time_matrix = np.zeros((n_stations, n_stations), dtype=np.float32)
        self.tasks: list[Task] = []
        self.resources: list[Resource] = []
        self.stations: list[Station] = []
        self.task_slot_to_id: list[Optional[int]] = [None] * self.max_tasks

        # Episode metrics.
        self.total_completed = 0
        self.total_travel = 0.0
        self.total_idle = 0.0
        self.total_late_completions = 0
        self.last_reward_breakdown: dict[str, float] = {}

    # ---------------------------------------------------------------------
    # Standard Gymnasium API
    # ---------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a fresh warehouse episode."""
        super().reset(seed=seed)

        self.time = 0.0
        self.decision_count = 0

        self.travel_time_matrix = self._build_travel_time_matrix()
        self.tasks = self._init_tasks()
        self.resources = self._init_resources()
        self.stations = self._init_stations()
        self.task_slot_to_id = [None] * self.max_tasks

        self.total_completed = 0
        self.total_travel = 0.0
        self.total_idle = 0.0
        self.total_late_completions = 0
        self.last_reward_breakdown = {}

        # Move to the first point where a dispatching decision is meaningful.
        self._advance_to_next_decision_point()
        self._refresh_task_slots()

        obs = self._get_obs()
        info = {
            "action_mask": self.action_masks(),
            "time": self.time,
        }
        return obs, info

    def step(self, action: int):
        """
        Apply one dispatcher decision.

        Notes
        -----
        - If the agent picks a valid (resource, task) assignment, the resource
          starts that job immediately at the current simulated time.
        - The environment then fast-forwards until the next meaningful event.
        - If the agent picks NO-OP while work is feasible, we simulate a small
          hesitation delay and charge idle time.
        """
        prev_cost = self._cost()
        penalty = 0.0
        had_feasible_work = self._has_dispatchable_pair()

        if action == self.noop_action:
            if had_feasible_work:
                penalty -= self.invalid_action_penalty
                self._apply_dispatch_delay(self.decision_delay_if_idle_choice)
        else:
            resource_idx, task_idx = self._decode_action(action)
            if self._is_feasible(resource_idx, task_idx):
                self._assign(resource_idx, task_idx)
            else:
                penalty -= self.invalid_action_penalty
                if had_feasible_work:
                    self._apply_dispatch_delay(self.decision_delay_if_idle_choice)

        # Jump to next decision point or episode end.
        self._simulate_until_next_decision()
        self._refresh_task_slots()

        new_cost = self._cost()
        reward = -(new_cost - prev_cost) + penalty

        self.last_reward_breakdown = {
            "prev_cost": prev_cost,
            "new_cost": new_cost,
            "delta_cost_component": -(new_cost - prev_cost),
            "penalty": penalty,
            "reward": reward,
        }

        self.decision_count += 1

        terminated = self._all_done() or (self.time >= self.shift_duration)
        truncated = self.decision_count >= self.max_decisions

        obs = self._get_obs()
        info = {
            "action_mask": self.action_masks(),
            "time": self.time,
            "reward_breakdown": self.last_reward_breakdown,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        """Simple text render useful while debugging."""
        free_resources = sum(not r.busy for r in self.resources)
        waiting_tasks = sum(t.status == "waiting" for t in self.tasks)
        in_progress = sum(t.status == "in_progress" for t in self.tasks)
        overdue = self._overdue_count()
        print(
            f"time={self.time:.1f} | waiting={waiting_tasks} | in_progress={in_progress} | "
            f"free_resources={free_resources} | overdue={overdue} | completed={self.total_completed}"
        )

    # ---------------------------------------------------------------------
    # Action mask and flattening helpers
    # ---------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """
        Boolean mask over the flattened action space.

        True = valid action
        False = invalid action
        """
        self._refresh_task_slots()
        mask = np.zeros(self.action_space.n, dtype=bool)

        for r in range(self.n_resources):
            for t in range(self.max_tasks):
                if self._task_exists(t) and self._is_feasible(r, t):
                    mask[self._encode_action(r, t)] = True

        mask[self.noop_action] = True
        return mask

    def _encode_action(self, resource_idx: int, task_idx: int) -> int:
        """Convert (resource_idx, task_slot_idx) -> one discrete action id."""
        return resource_idx * self.max_tasks + task_idx

    def _decode_action(self, action: int) -> tuple[int, int]:
        """Convert a flattened action id back into (resource_idx, task_slot_idx)."""
        return action // self.max_tasks, action % self.max_tasks

    # ---------------------------------------------------------------------
    # Initialisation helpers
    # ---------------------------------------------------------------------
    def _build_travel_time_matrix(self) -> np.ndarray:
        """Create a simple symmetric zone-to-zone travel time matrix."""
        mat = self.np_random.integers(2, 12, size=(self.n_stations, self.n_stations)).astype(np.float32)
        mat = ((mat + mat.T) / 2.0).astype(np.float32)
        np.fill_diagonal(mat, 0.0)
        return mat

    def _init_tasks(self) -> list[Task]:
        """
        Create the episode's task list.

        This example pre-generates up to max_tasks tasks for the whole shift.
        Each task becomes visible only when its release_time is reached.
        """
        tasks: list[Task] = []

        current_release = 0.0
        for task_id in range(self.max_tasks):
            # Spread tasks through the first 80% of the shift.
            if task_id == 0:
                release_time = float(self.np_random.uniform(0.0, min(20.0, self.shift_duration * 0.2)))
            else:
                gap = float(
                    self.np_random.uniform(
                        self.min_task_release_gap,
                        max(self.min_task_release_gap + 1.0, self.shift_duration * 0.08),
                    )
                )
                current_release += gap
                release_time = min(current_release, self.shift_duration * 0.8)

            origin = int(self.np_random.integers(0, self.n_stations))
            destination = int(self.np_random.integers(0, self.n_stations - 1))
            if destination >= origin:
                destination += 1  # ensure destination != origin

            required_skill = int(self.np_random.integers(0, 3))
            task_type = required_skill  # keep task/resource compatibility simple
            service_time = float(self.np_random.uniform(4.0, 15.0))
            priority = int(self.np_random.integers(1, 4))

            base_travel = float(self.travel_time_matrix[origin, destination])
            due_slack = float(self.np_random.uniform(10.0, 40.0))
            due_time = release_time + service_time + base_travel + due_slack

            tasks.append(
                Task(
                    id=task_id,
                    task_type=task_type,
                    origin_zone=origin,
                    destination_zone=destination,
                    release_time=float(release_time),
                    due_time=float(due_time),
                    service_time=float(service_time),
                    priority=priority,
                    required_skill=required_skill,
                )
            )

        return tasks

    def _init_resources(self) -> list[Resource]:
        """Create resources and place them randomly in warehouse zones."""
        resources: list[Resource] = []
        for resource_id in range(self.n_resources):
            rtype = resource_id % 3  # cycle through picker / forklift / agv
            zone = int(self.np_random.integers(0, self.n_stations))
            resources.append(Resource(id=resource_id, resource_type=rtype, current_zone=zone))
        return resources

    def _init_stations(self) -> list[Station]:
        """Create stations with small finite capacities."""
        stations: list[Station] = []
        default_capacities = [2, 3, 2, 2, 1]
        for station_id in range(self.n_stations):
            station_type = station_id if station_id < len(self.STATION_TYPE_NAMES) else station_id
            capacity = default_capacities[station_id] if station_id < len(default_capacities) else 2
            stations.append(Station(id=station_id, station_type=station_type, capacity=capacity))
        return stations

    # ---------------------------------------------------------------------
    # Event-driven simulation core
    # ---------------------------------------------------------------------
    def _advance_to_next_decision_point(self):
        """
        Move time forward until one of these becomes true:
        - at least one feasible (resource, task) assignment exists
        - the episode has naturally ended
        - the shift duration is reached

        This method is the heart of the event-driven design.
        """
        # First process all events that already happen at the current time.
        self._release_new_tasks()
        self._complete_finished_tasks()

        while True:
            if self._all_done() or self.time >= self.shift_duration:
                return

            # If work can be dispatched now, stop advancing time.
            if self._has_dispatchable_pair():
                return

            next_time = self._next_event_time()
            if next_time is None:
                # No future arrivals or completions remain.
                return

            # Accumulate idle time for resources that were free while time advances.
            idle_resources = sum(not r.busy for r in self.resources)
            delta_t = max(0.0, next_time - self.time)
            self.total_idle += idle_resources * delta_t
            self.time = next_time

            self._release_new_tasks()
            self._complete_finished_tasks()

    def _simulate_until_next_decision(self):
        """After an action, advance until the next meaningful dispatch event."""
        self._advance_to_next_decision_point()

    def _release_new_tasks(self):
        """Reveal tasks whose release_time has been reached."""
        for task in self.tasks:
            if task.status == "hidden" and task.release_time <= self.time:
                task.status = "waiting"

    def _complete_finished_tasks(self):
        """
        Finalise all tasks whose assigned resource has reached busy_until.

        Completion effects:
        - task becomes done
        - resource becomes free in the task's destination zone
        - destination station queue is decremented
        """
        for resource in self.resources:
            if not resource.busy:
                continue
            if resource.busy_until > self.time:
                continue

            task = self.tasks[resource.assigned_task_id]
            if task.status == "in_progress":
                task.status = "done"
                task.completion_time = self.time
                self.total_completed += 1
                if task.completion_time > task.due_time:
                    self.total_late_completions += 1

                # Release capacity at destination station once the job completes.
                dest_station = self.stations[task.destination_zone]
                dest_station.queue_length = max(0, dest_station.queue_length - 1)

            resource.current_zone = task.destination_zone
            resource.busy_until = 0.0
            resource.assigned_task_id = None
            resource.current_load = 0

    def _next_event_time(self) -> Optional[float]:
        """Return the next future event time, or None if no event remains."""
        candidates: list[float] = []

        for task in self.tasks:
            if task.status == "hidden" and task.release_time > self.time:
                candidates.append(task.release_time)

        for resource in self.resources:
            if resource.busy and resource.busy_until > self.time:
                candidates.append(resource.busy_until)

        if not candidates:
            return None
        return float(min(candidates))

    def _apply_dispatch_delay(self, delay: float):
        """
        Advance a short amount of simulated time when the dispatcher hesitates.

        This prevents the episode from getting stuck in the exact same state if
        the policy repeatedly chooses NO-OP despite feasible work.
        """
        idle_resources = sum(not r.busy for r in self.resources)
        delay = max(0.0, delay)
        if delay <= 0.0:
            return
        self.total_idle += idle_resources * delay
        self.time = min(self.shift_duration, self.time + delay)
        self._release_new_tasks()
        self._complete_finished_tasks()

    # ---------------------------------------------------------------------
    # Assignment and feasibility
    # ---------------------------------------------------------------------
    def _assign(self, resource_idx: int, task_slot_idx: int):
        """
        Assign one free resource to one waiting task.

        Job duration =
            travel(resource -> task.origin)
            + service_time
            + travel(task.origin -> task.destination)
        """
        task = self._slot_task(task_slot_idx)
        if task is None:
            raise ValueError("Tried to assign from an empty task slot")

        resource = self.resources[resource_idx]
        reposition_travel = float(self.travel_time_matrix[resource.current_zone, task.origin_zone])
        loaded_travel = float(self.travel_time_matrix[task.origin_zone, task.destination_zone])
        total_duration = reposition_travel + task.service_time + loaded_travel

        # Reserve downstream capacity immediately. This is a simplifying choice:
        # once the task starts, its destination queue slot is treated as occupied.
        dest_station = self.stations[task.destination_zone]
        dest_station.queue_length += 1

        task.status = "in_progress"
        task.assigned_resource = resource.id
        task.start_time = self.time
        task.completion_time = None

        resource.assigned_task_id = task.id
        resource.busy_until = self.time + total_duration
        resource.current_load = 1

        self.total_travel += reposition_travel + loaded_travel

    def _is_feasible(self, resource_idx: int, task_slot_idx: int) -> bool:
        """
        Check whether a resource-task assignment is currently allowed.

        The rules used in this simplified example are:
        - resource exists and is free
        - task slot exists and the underlying task is waiting
        - resource type matches task required skill
        - destination station still has spare capacity
        """
        if not (0 <= resource_idx < self.n_resources):
            return False
        if not (0 <= task_slot_idx < self.max_tasks):
            return False

        task = self._slot_task(task_slot_idx)
        if task is None:
            return False

        resource = self.resources[resource_idx]
        dest_station = self.stations[task.destination_zone]

        if resource.busy:
            return False
        if task.status != "waiting":
            return False
        if resource.resource_type != task.required_skill:
            return False
        if dest_station.blocked:
            return False
        return True

    def _has_dispatchable_pair(self) -> bool:
        """Return True if at least one valid (resource, task) assignment exists."""
        self._refresh_task_slots()
        for r in range(self.n_resources):
            for t in range(self.max_tasks):
                if self._task_exists(t) and self._is_feasible(r, t):
                    return True
        return False

    # ---------------------------------------------------------------------
    # Task slot helpers
    # ---------------------------------------------------------------------
    def _refresh_task_slots(self):
        """
        Build the visible task table used by both observations and actions.

        Because RL action spaces must stay fixed-size, the environment exposes
        at most `max_tasks` currently visible tasks at once. Hidden and done
        tasks are excluded. Remaining slots are padded with None.
        """
        visible = [t for t in self.tasks if t.status in ("waiting", "in_progress")]

        # Waiting tasks first, then in-progress tasks. Inside each group, sort
        # urgent / important tasks earlier.
        status_rank = {"waiting": 0, "in_progress": 1}
        visible.sort(
            key=lambda t: (
                status_rank[t.status],
                t.due_time,
                -t.priority,
                t.release_time,
                t.id,
            )
        )

        ids = [t.id for t in visible[: self.max_tasks]]
        ids += [None] * (self.max_tasks - len(ids))
        self.task_slot_to_id = ids

    def _task_exists(self, task_slot_idx: int) -> bool:
        """Return True if the visible slot currently references a real task."""
        return self.task_slot_to_id[task_slot_idx] is not None

    def _slot_task(self, task_slot_idx: int) -> Optional[Task]:
        """Return the Task object referenced by a visible task slot."""
        task_id = self.task_slot_to_id[task_slot_idx]
        if task_id is None:
            return None
        return self.tasks[task_id]

    # ---------------------------------------------------------------------
    # Metrics and terminal conditions
    # ---------------------------------------------------------------------
    def _backlog_count(self) -> int:
        """Number of currently visible unfinished tasks."""
        return sum(t.status in ("waiting", "in_progress") for t in self.tasks)

    def _overdue_count(self) -> int:
        """Number of unfinished visible tasks already past due_time."""
        return sum(
            t.status in ("waiting", "in_progress") and t.due_time < self.time for t in self.tasks
        )

    def _all_done(self) -> bool:
        """True when every task for the episode has been completed."""
        return all(t.status == "done" for t in self.tasks)

    def _cost(self) -> float:
        """
        Weighted operational cost used to shape the reward.

        Lower cost is better.
        This is intentionally simple and easy to modify.
        """
        backlog = self._backlog_count()
        overdue = self._overdue_count()
        return (
            1.0 * overdue
            + 0.5 * backlog
            + 0.2 * self.total_travel
            + 0.1 * self.total_idle
        )

    # ---------------------------------------------------------------------
    # Observation construction
    # ---------------------------------------------------------------------
    def _get_obs(self) -> dict[str, np.ndarray]:
        """
        Build the structured observation dictionary.

        Observation layout
        ------------------
        global: [
            current_time,
            backlog_count,
            overdue_count,
            free_resource_count,
            mean_station_queue_length,
        ]

        tasks row = [
            task_type,
            origin_zone,
            destination_zone,
            waiting_time,
            slack,
            service_time,
            priority,
            required_skill,
            status_code,
        ]

        resources row = [
            resource_type,
            current_zone,
            busy_flag,
            remaining_busy_time,
            current_load,
            compatible_waiting_task_count,
        ]

        stations row = [
            queue_length,
            utilization,
            blocked_flag,
        ]
        """
        self._refresh_task_slots()

        global_obs = np.array(
            [
                self.time,
                float(self._backlog_count()),
                float(self._overdue_count()),
                float(sum(not r.busy for r in self.resources)),
                float(np.mean([s.queue_length for s in self.stations])) if self.stations else 0.0,
            ],
            dtype=np.float32,
        )

        task_obs = np.zeros((self.max_tasks, self.n_task_features), dtype=np.float32)
        for slot_idx, task_id in enumerate(self.task_slot_to_id):
            if task_id is None:
                continue
            task = self.tasks[task_id]
            waiting_time = max(0.0, self.time - task.release_time) if task.status != "hidden" else 0.0
            slack = task.due_time - self.time
            status_code = 1.0 if task.status == "waiting" else 2.0 if task.status == "in_progress" else 0.0
            task_obs[slot_idx] = np.array(
                [
                    task.task_type,
                    task.origin_zone,
                    task.destination_zone,
                    waiting_time,
                    slack,
                    task.service_time,
                    task.priority,
                    task.required_skill,
                    status_code,
                ],
                dtype=np.float32,
            )

        resource_obs = np.zeros((self.n_resources, self.n_resource_features), dtype=np.float32)
        waiting_tasks = [t for t in self.tasks if t.status == "waiting"]
        for idx, resource in enumerate(self.resources):
            remaining_busy = max(0.0, resource.busy_until - self.time) if resource.busy else 0.0
            compatible_waiting = sum(
                (t.required_skill == resource.resource_type) and not self.stations[t.destination_zone].blocked
                for t in waiting_tasks
            )
            resource_obs[idx] = np.array(
                [
                    resource.resource_type,
                    resource.current_zone,
                    1.0 if resource.busy else 0.0,
                    remaining_busy,
                    float(resource.current_load),
                    float(compatible_waiting),
                ],
                dtype=np.float32,
            )

        station_obs = np.zeros((self.n_stations, self.n_station_features), dtype=np.float32)
        for idx, station in enumerate(self.stations):
            station_obs[idx] = np.array(
                [
                    float(station.queue_length),
                    float(station.utilization),
                    1.0 if station.blocked else 0.0,
                ],
                dtype=np.float32,
            )

        return {
            "global": global_obs,
            "tasks": task_obs,
            "resources": resource_obs,
            "stations": station_obs,
        }


def demo_random_rollout(seed: int = 7, max_steps: int = 20):
    """Small helper for quickly testing the environment manually."""
    env = WarehouseDispatchEnv(max_tasks=10, n_resources=4, n_stations=5, shift_duration=240.0)
    obs, info = env.reset(seed=seed)

    print("Initial global obs:", obs["global"])
    print("Initial valid actions:", int(info["action_mask"].sum()))

    terminated = truncated = False
    step_idx = 0
    total_reward = 0.0

    while not (terminated or truncated) and step_idx < max_steps:
        valid_actions = np.flatnonzero(info["action_mask"])
        action = int(env.noop_action)
        non_noop = [a for a in valid_actions if a != env.noop_action]
        if non_noop:
            action = int(non_noop[0])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"step={step_idx:02d} action={action:03d} time={info['time']:.1f} "
            f"reward={reward:.3f} backlog={obs['global'][1]:.0f} overdue={obs['global'][2]:.0f}"
        )
        step_idx += 1

    print("Finished. terminated=", terminated, "truncated=", truncated)
    print("Completed tasks:", env.total_completed)
    print("Late completions:", env.total_late_completions)
    print("Total reward:", round(total_reward, 3))
    env.close()


if __name__ == "__main__":
    demo_random_rollout()

from warehouse_dispatch_env import WarehouseDispatchEnv

env = WarehouseDispatchEnv(
    max_tasks=10,
    n_resources=4,
    n_stations=5,
    shift_duration=240.0,
    max_decisions=100,
)

obs, info = env.reset(seed=42)
done = False

while not done:
    valid_actions = info["action_mask"].nonzero()[0]
    action = valid_actions[0]  # replace with your policy
    obs, reward, terminated, truncated, info = env.step(int(action))
    done = terminated or truncated
    env.render()

env.close()