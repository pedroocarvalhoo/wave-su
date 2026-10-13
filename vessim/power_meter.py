from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count
from typing import Optional
from datetime import datetime
from enum import Enum


class PowerMode(Enum):
    ECO=1
    NORMAL=2
    TURBO=3

class PowerMeter(ABC):
    _ids = count(0)

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def measure(self) -> float:
        """Abstract method to measure and return the current node power demand."""

    def finalize(self) -> None:
        """Abstract method to finalize the power meter."""

class MockPowerMeter(PowerMeter):
    def __init__(
        self,
        max_p: Optional[float] = None,
        idle_p: float = 0.0,
        name: Optional[str] = None,
        mode: Optional[PowerMode] = None,
        **kwargs,
    ):
        initial_p = kwargs.pop("p", None)
        if max_p is None:
            if initial_p is None:
                raise TypeError("MockPowerMeter requires either max_p or p")
            max_p = initial_p
        if name is None:
            name = f"MockPowerMeter-{next(self._ids)}"
        super().__init__(name)
        if max_p < 0 or idle_p < 0:
            raise ValueError("max_p and idle_p must not be less than 0")
        
        self.absolute_p = max_p
        self.max_p = max_p
        match mode:
            case PowerMode.ECO:
                self.mode = PowerMode.ECO
                self.max_p *= 0.7         # arbitrary max power reduction
            case PowerMode.NORMAL:
                self.mode=PowerMode.NORMAL
                self.max_p *= 0.9         # arbitrary max power reduction
            case PowerMode.TURBO:
                self.mode = PowerMode.TURBO
            case _:
                self.mode = PowerMode.TURBO
            
        self._p = initial_p if initial_p is not None else self.max_p
        self.idle_p = idle_p
        self.kwargs = kwargs
        
    def set_power_mode(self, mode: PowerMode):
        match mode:
            case PowerMode.ECO:
                self.mode = PowerMode.ECO
                self.max_p = 0.7 * self.absolute_p         # arbitrary max power reduction
            case PowerMode.NORMAL:
                self.mode = PowerMode.NORMAL
                self.max_p = 0.9 * self.absolute_p         # arbitrary max power reduction
            case PowerMode.TURBO:
                self.mode = PowerMode.TURBO
            case _:
                self.mode = PowerMode.TURBO

    def set_power(self, value):
        if value < 0 or value > self.max_p:
            raise ValueError("p must not be less than 0 and higher than max_p")
        self._p = value

    def set_usage(self, value: float):
        if value < 0 or value > 1:
            raise ValueError("usage must be between 0 and 1")
        # very simple power model defined as idle + computing consumption
        self._p = self.idle_p + self.max_p * value

    def simulate_usage(self, value: float) -> float:
        return self.idle_p + self.max_p * value

    def measure(self) -> float:
        return self._p


#WAVE 

class ComputeNode(PowerMeter):  
    _ids = count(0)
    def __init__(
        self, 
        max_p: float, 
        idle_p: float, 
        name: Optional[str] = None, 
        base_frequency_ghz: float = 0.0,
    ):
        if name is None:
            name = f"ComputeNode-{next(self._ids)}"
        super().__init__(name)
        if max_p < 0 or idle_p < 0:
            raise ValueError(f"[{datetime.now()}][COMPUTENODE] max_p and idle_p must not be less than 0 on {name}")

        self.absolute_p = max_p  # max dynamic power at base frequency
        self.max_p = max_p       # current dynamic power (changes with DVFS)
        self.idle_p = idle_p
        self.base_frequency_ghz = base_frequency_ghz
        
        self.current_frequency: Optional[float] = None  # set by scheduler on job start
        self.frequency_history: list[tuple[datetime, Optional[float]]] = []  # (time, freq_ghz)
        
        self._is_executing = False
        
        self._p = self.idle_p
        
        self.status = 'IDLE'
        self.current_job = None 
        self.total_jobs_executed: int = 0
        self.total_failures: int = 0
        
        self._downtime_remaining: float = 0.0
        
        self._idle_frequency = 0.5 * self.base_frequency_ghz if self.base_frequency_ghz > 0 else None
    
    # ============================================================================
    # GETTERS
    # ============================================================================
    
    def is_executing(self) -> bool:
        return self._is_executing
    
    def is_idle(self) -> bool:
        return self.status == 'IDLE'
    
    def is_busy(self) -> bool:
        return self.status == 'BUSY'
    
    def is_down(self) -> bool:
        return self.status == 'DOWN'
    
    def is_booting(self) -> bool:
        return self.status == 'BOOTING'

    def measure(self) -> float:
        return self._p
    
    def get_dynamic_power(self) -> float:
        return self.max_p

    # ============================================================================
    # DVFS
    # ============================================================================

    def set_frequency(
        self,
        freq_ghz: float,
        dynamic_power_w: float,
        time: Optional[datetime] = None,
    ):
        previous_frequency = self.current_frequency
        self.current_frequency = freq_ghz
        self.max_p = dynamic_power_w
        self._update_power()

        frequency_changed = (
            previous_frequency is None
            or abs(previous_frequency - freq_ghz) > 1e-6
        )
        if time is not None and frequency_changed:
            self.frequency_history.append((time, self.current_frequency))
    
    # ============================================================================
    # COMPUTATION
    # ============================================================================
    
    def set_job_executing(self, is_executing: bool):
        self._is_executing = is_executing
        self._update_power()
    
    def start_job(
        self, 
        job, 
        time: datetime, 
        frequency: Optional[float] = None, 
        dynamic_power_w: Optional[float] = None
    ):

        if self.status != 'IDLE':
            raise ValueError(
                f"[{time}][COMPUTENODE] Cannot start job on node {self.name} that is not IDLE "
                f"(current status: {self.status})"
            )
        
        if frequency is not None and dynamic_power_w is not None:
            self.set_frequency(frequency, dynamic_power_w)
        
        self.frequency_history.append((time, self.current_frequency))
        
        self.status = 'BUSY'
        self.current_job = job
        self.set_job_executing(True)
    
    def complete_job(self, time: datetime):
        if self.status != 'BUSY':
            raise ValueError(
                f"[{time}][COMPUTENODE] Cannot complete job on node {self.name} that is not BUSY "
                f"(current status: {self.status})"
            )
        
        completed_job = self.current_job
        
        self.status = 'IDLE'
        self.current_job = None
        self.total_jobs_executed += 1
        self.set_job_executing(False)
        
        self.current_frequency = self._idle_frequency
        self.max_p = self.absolute_p
        
        self.frequency_history.append((time, self.current_frequency))
        
        return completed_job
    
    def fail_job(self, time: datetime, downtime: float):
        failed_job = self.current_job
        
        self.status = 'DOWN'
        self.current_job = None
        self.total_failures += 1
        self._is_executing = False
        self._downtime_remaining = downtime
        
        self.current_frequency = 0
        self.max_p = self.absolute_p
        
        self.frequency_history.append((time, 0))
        self._update_power()
        
        return failed_job
    
    def update_downtime(self, elapsed: float, time: datetime = None) -> bool:
        if self.status == 'DOWN':
            self._downtime_remaining -= elapsed
            if self._downtime_remaining <= 0:
                self._downtime_remaining = 0
                self.status = 'BOOTING'
                self._update_power()
                # TODO: model boot time separately; for now transition immediately to IDLE
                self.status = 'IDLE'
                self.current_frequency = self._idle_frequency
                if time is not None:
                    self.frequency_history.append((time, self.current_frequency))
                self._update_power()
                return True
        return False
    
    def power_off(self, time: datetime):
        if self.status != 'OFF':
            self.status = 'OFF'
            self.current_job = None
            self._is_executing = False
            self.current_frequency = 0
            self.frequency_history.append((time, 0))
            self._update_power()

    def power_on(self, time: datetime):
        if self.status == 'OFF':
            self.status = 'IDLE'
            self.current_frequency = self._idle_frequency
            self.frequency_history.append((time, self.current_frequency))
            self._update_power()


    def _update_power(self): 
        if self.status == 'DOWN' or self.status == 'OFF':
            # DOWN/OFF: node is off / crashed 
            self._p = 0.0
        elif self.status == 'BOOTING':
            # BOOTING: node is powering on - TODO: study realistic boot power profile
            self._p = self.idle_p
        elif self._is_executing:
            # BUSY: idle power + dynamic power at current frequency
            self._p = self.idle_p + self.max_p 
        else:
            # IDLE
            self._p = self.idle_p
    
    
    


    
    

