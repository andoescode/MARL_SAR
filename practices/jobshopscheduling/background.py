from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class Task:
    '''
        Represents one warehouse job that may be dispatched to a resource.
    '''

    id: int
    task_type: int
    origin_zone: int
    destination_zone: int
    release_time: float
    due_time: float
    service_time: float
    priority: int
    required_skill: int
    status: str = 'hidden'  # hidden -> waiting -> in_progress -> done
    assigned_resource: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None


@dataclass
class Resource:
    '''
        Represents a dispatchable warehouse resource such as a picker or AGV.
    '''

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
    '''
        Represents a warehouse zone/station with limited downstream capacity.
    '''

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
