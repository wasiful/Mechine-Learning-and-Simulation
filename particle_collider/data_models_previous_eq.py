import numpy as np
from dataclasses import dataclass
import typing as t


@dataclass
class Constants:
    BOHR_RADIUS = 5.29177210903e-11



    VOLUME = (4 / 3) * np.pi * BOHR_RADIUS ** 3
    BLT_CONST = 1.380649e-23
    INIT_VELOCITY = 1.845e3

@dataclass
class CollisionModel:
    """

    """
    temp: float
    mass: float
    time1: float
    time2: float
    del_t: float
    force: t.Optional[float] = None
    BLT_CONST = Constants.BLT_CONST
    INIT_VELOCITY = Constants.INIT_VELOCITY
    VOLUME = Constants.VOLUME
    FORCE_CONST = 1
    const = float
    amu = float
    # SPACIFIC_HEAT = 1

    def __post_init__(self):
        if not self.force:
            self.force = self.collide()

    def collide(self) -> float:
        """

        :return:
        """
        v_rms = np.sqrt((3 * self.BLT_CONST * self.temp) / self.mass)
        v1 = self.INIT_VELOCITY + v_rms  # it has absorbed radiated energy itself while moving
        a = v_rms / self.time1  # an unknown particle with different acceleration about to hit V1 particle
        v2 = v1 + a * self.time2  # it is being hit by a particle thus received some energy from that's acceleration

        # del_v_square = (v2 - v1) ** 2
        v_final = v2  # a tiny amount of lost energy in rotation and randomness
        del_v_square = v2 ** 2 - v1 ** 2

        del_e = (1 / 2) * self.mass * del_v_square

        enr_ex_rate = (del_e/np.abs(self.time2 - self.time1)) / self.del_t
        concentration = (1/2) * self.mass * v_final ** 2 / self.VOLUME
        # concentration = self.mass * self.SPACIFIC_HEAT * (T2 - T1) / self.VOLUME
        temperature_from_velocity = (1 / 3) * (self.mass * v2 ** 2)
        final_force = self.FORCE_CONST * temperature_from_velocity * concentration * enr_ex_rate
        # final_force = enr_ex_rate
        return final_force


    def to_dict(self):
        return dict(
            temp=self.temp,
            mass=self.mass,
            time1=self.time1,
            time2=self.time2,
            del_t=self.del_t,
            force=self.force
        )



