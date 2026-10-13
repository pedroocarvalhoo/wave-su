from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger

class Storage(ABC):
    @abstractmethod
    def update(self, power: float, duration: int) -> float:
        """Feed or draw energy for specified duration.

        Args:
            power: Power to be (dis)charged in W. Charging if positive, discharging if negative.
            duration: Duration in seconds for which the storage will be (dis)charged.

        Returns:
            The total energy in Ws that has been charged/discharged.
        """

    def state(self) -> dict:
        """Returns information about the current state of the storage. Should be overridden."""
        return {}


class SimpleBattery(Storage):
    """(Way too) simple battery.

    Args:
        capacity: Battery capacity in watt-seconds (Ws).
        charge_level: Initial charge level in watt-seconds (Ws). Defaults to 0Ws.
        min_soc: Minimum allowed state of charge (SoC) for the battery.
            Defaults to 0%. Can be altered during simulation.
        c_rate: Optional C-rate, which defines the charge and discharge rate of the battery.
            For more information on C-rate, see `C-rate explanation <https://www.batterydesign.net/electrical/c-rate/>`_.
            Defaults to None.
    """

    def __init__(
        self,
        capacity: float,
        charge_level: float = 0,
        min_soc: float = 0,
        c_rate: Optional[float] = None,
    ):
        self.capacity = capacity
        assert 0 <= charge_level <= self.capacity
        self.charge_level = charge_level
        assert 0 <= min_soc <= self.soc()
        self.min_soc = min_soc
        self.c_rate = c_rate

    def update(self, power: float, duration: int) -> float:
        if duration <= 0.0:
            raise ValueError("Duration needs to be a positive value")

        assert self.min_soc <= self.soc(), "Minimum SoC can not be smaller than the current SoC"
        if self.c_rate is not None:
            max_power = self.c_rate * self.capacity / 3600
            if power >= max_power:
                # Too high charge rate
                logger.info(
                    f"Trying to charge storage '{self.__class__.__name__}' with "
                    f"{power} W but only {max_power} W are supported."
                )
                power = max_power

            if power <= -max_power:
                # Too high discharge rate
                logger.info(
                    f"Trying to discharge storage '{self.__class__.__name__}' "
                    f"with {power} W but only {max_power} W are supported."
                )
                power = -max_power

        charged_energy = power * duration  # Total energy to be (dis)charged in Ws
        new_charge_level = self.charge_level + charged_energy

        abs_min_soc = self.min_soc * self.capacity
        if new_charge_level < abs_min_soc:
            # Battery can not be discharged further than the minimum state-of-charge
            charged_energy = abs_min_soc - self.charge_level
            self.charge_level = abs_min_soc
        elif new_charge_level > self.capacity:
            # Battery can not be charged past its capacity
            charged_energy = self.capacity - self.charge_level
            self.charge_level = self.capacity
        else:
            self.charge_level = new_charge_level

        return charged_energy

    def soc(self) -> float:
        return self.charge_level / self.capacity

    def state(self) -> dict:
        return {
            "soc": self.soc(),
            "charge_level": self.charge_level,
            "capacity": self.capacity,
            "min_soc": self.min_soc,
            "c_rate": self.c_rate,
        }
        

class BetterBattery(Storage):

    def __init__(self, backpack):
        # Parameters (Battery)
        self._state_of_charge_min = backpack.get('state_of_charge_min')     # minimum state of charge [0-1]
        self._state_of_charge_max = backpack.get('state_of_charge_max')     # maximum state of charge [0-1]
        self._battery_energy_max = backpack.get('battery_energy_max')       # battery capacity in Wh
        self._charge_efficiency = backpack.get('charge_efficiency')         # charging efficiency [0-1]
        self._discharge_efficiency = backpack.get('discharge_efficiency')   # discharging efficiency [0-1]
        self._p_charge_max = backpack.get('p_charge_max')                   # maximum charging power in W
        self._p_discharge_max = backpack.get('p_discharge_max')             # maximum discharging power in W
        self._battery_energy = backpack.get('initial_battery_energy')       # read current capacity in Wh
        self._final_battery_energy = backpack.get('final_battery_energy')   # assume initial or fixed in Wh
        self._expected_charge = 0.0

    def soc (self) -> float:
        return self._battery_energy / self._battery_energy_max
    
    # +power charges, -power discharges
    # assumes power already respects max_charge and max_discharge rates
    def setExpectedCharge (self, power: int):
        chg_power = self._battery_energy + power
        p_min = self._battery_energy * self._state_of_charge_min
        p_max = self._battery_energy * self._state_of_charge_max
        
        # validates if what's expected doesn violate min and max soc
        if power < 0:   # discharging
            if chg_power >= p_min:
                self._expected_charge = power
            else:
                self._expected_charge = p_min - chg_power
        else:   # charging
            if chg_power <= p_max:
                self._expected_charge = power
            else:
                self._expected_charge = chg_power - p_max
        
    
    # este pedido vem da policy...
    # power positivo carrega, negativo descarrega
    def update (self, power: float, duration: int) -> float:
        if duration <= 0.0:
            raise ValueError("Duration needs to be a positive value")

        assert self._state_of_charge_min <= self.soc(), "Minimum SoC can not be smaller than the current SoC "+str(self._state_of_charge_min)+" "+str(self._battery_energy)+" "+str(self._battery_energy_max)
        
        if power >= self._p_charge_max:
            # Too high charge rate
            power = self._p_charge_max

        if power <= -self._p_discharge_max:
            # Too high discharge rate
            power = -self._p_discharge_max
        
        charged_energy = power * duration / 3600.0  # Total energy to be (dis)charged in Ws -> Wh
        new_charge_level = self._battery_energy + charged_energy
        
        #print ("Old: "+str(self._battery_energy)+" Chg:"+str(charged_energy)+" Pred Chg: "+str(self._expected_charge))
        
        # TODO: simplify later
        excess_energy = 0
        if new_charge_level >= self._state_of_charge_min*self._battery_energy_max and new_charge_level <= self._state_of_charge_max*self._battery_energy_max:
            if charged_energy < 0 and self._expected_charge < 0:  # ensure they have the same signal
                if charged_energy >= self._expected_charge: # both negative values
                    self._battery_energy = new_charge_level
                else:    
                    self._battery_energy = self._battery_energy + self._expected_charge
                    excess_energy = charged_energy - self._expected_charge 
            elif charged_energy >= 0 and self._expected_charge >= 0:   # ensure they have the same signal
                if charged_energy <= self._expected_charge: # both positive values
                    self._battery_energy = new_charge_level
                else:
                    self._battery_energy = self._battery_energy + self._expected_charge
                    excess_energy = charged_energy - self._expected_charge
            else:   # no charge or discharge is applied
                excess_energy = charged_energy
        else:
            excess_energy = charged_energy
            
        #print ("New: "+str(self._battery_energy)+" Excess:"+str(excess_energy))
        
        return excess_energy    # returns the excedent
            

    def state(self) -> dict:
        return {
            "soc": self.soc(),
            "charge_level": self._battery_energy,
            "capacity": self._battery_energy_max,
            "min_soc": self._state_of_charge_min,
            #"c_rate": self.c_rate,
        }


#WAVE
class NewBattery(Storage):

    def __init__(self, backpack):
        self._state_of_charge_min = backpack.get('state_of_charge_min')
        self._state_of_charge_max = backpack.get('state_of_charge_max')
        self._battery_energy_max = backpack.get('battery_energy_max')       # Wh
        self._charge_efficiency = backpack.get('charge_efficiency')
        self._discharge_efficiency = backpack.get('discharge_efficiency')
        self._p_charge_max = backpack.get('p_charge_max')                   # W
        self._p_discharge_max = backpack.get('p_discharge_max')             # W
        self._battery_energy = backpack.get('initial_battery_energy')       # Wh
        self._final_battery_energy = backpack.get('final_battery_energy')   # Wh
        self._last_power_w = 0.0

    def soc(self) -> float:
        return self._battery_energy / self._battery_energy_max

    def update(self, power: float, duration: int) -> float:
        
        if duration <= 0.0:
            raise ValueError("Duration needs to be a positive value")

        if power > 0:
            power = min(power, self._p_charge_max)
        elif power < 0:
            power = max(power, -self._p_discharge_max)

        energy_wh = power * duration / 3600.0

        if energy_wh >= 0:  
            battery_change = energy_wh * self._charge_efficiency
        else:  
            battery_change = energy_wh / self._discharge_efficiency

        new_level = self._battery_energy + battery_change
        min_energy = self._state_of_charge_min * self._battery_energy_max
        max_energy = self._state_of_charge_max * self._battery_energy_max

        if new_level < min_energy:
            battery_change = min_energy - self._battery_energy
            self._battery_energy = min_energy
        elif new_level > max_energy:
            battery_change = max_energy - self._battery_energy
            self._battery_energy = max_energy
        else:
            self._battery_energy = new_level

        if battery_change >= 0:  
            system_energy_ws = (battery_change / self._charge_efficiency) * 3600.0
        else: 
            system_energy_ws = (battery_change * self._discharge_efficiency) * 3600.0

        self._last_power_w = system_energy_ws / duration

        return system_energy_ws

    def state(self) -> dict:
        return {
            "soc": self.soc(),
            "charge_level": self._battery_energy,
            "capacity": self._battery_energy_max,
            "min_soc": self._state_of_charge_min,
            "battery_power_w": self._last_power_w,
        }
