import numpy as np
from dataclasses import dataclass
import typing as t


@dataclass
class Constants:
    BOHR_RADIUS = 5.29177210903e-11
    VOLUME = (4 / 3) * np.pi * BOHR_RADIUS ** 3
    BLT_CONST = 1.380649e-23
    INIT_VELOCITY = 1.845e3


# @dataclass
# class Separate_collision_modle:
#
#     temp: float
#     mass: float
#     time1: float
#     time2: float
#     del_t: float
#     force: t.Optional[float] = None
#     BLT_CONST = Constants.BLT_CONST
#     INIT_VELOCITY = Constants.INIT_VELOCITY
#     VOLUME = Constants.VOLUME
#     FORCE_CONST = 1
#     # SPACIFIC_HEAT = 1
#
#
#
#     def calc_v_rms(T, m, Kb):
#         v_rms = np.sqrt((3 * Kb * T) / m)
#         return v_rms
#
#     # a list of velocity is required and a list which has the first value as initial velocity at 0 degree
#     def calc_first_velocity(v_rms, v: float):
#         # add new velocity to the initial velocity (first value)  """ v """
#         v1 = v + v_rms
#         return v1
#
#     def accelaration(self):
#         a = v_rms / t1
#
#     # v2 = v1 + a * t      v = u+at
#     def calc_second_velocity(v1):
#         # time = np.linspace(1, 600, 1000)
#         # acceleration = np.linspace(1, 1000, 1000)  # start, stop, step
#         v2 = v1 + a * t2
#         return v2
#
#     def calc_del_E(v1, v2):
#         del_v_square = v2 ** 2 - v1 ** 2
#         del_E = (1 / 2) * m * del_v_square
#         return del_E
#
#     def calc_E_er(del_E):
#         del_t = random time  # t2 is the second index t2 = i+1 th, and t1 is first index t1 = i th del_t is constant
#         E_er = del_E / del_t
#         E_er.append(E_er_list)
#         return E_er
#
#     def calc_temp(m, v_rms, Kb):
#         T = (1 / 3) * ((m * v ** 2) / Kb)  # calculated temperature
#         return T
#
#     def calc_conc_field(m: float, v2: float, volume):
#         # here velocity is constant initial velocity
#         # conc = ((1 / 2) * m * v2 ** 2) / volume
#         # here m is constant initial mass
#         conc = ((3 / 2) * Kb * T) / volume
#         return conc
#
#     def force_vs_T(T):
#         const = 1
#         conc = 1
#         E_er = 1
#         F_T = const * T * conc * E_er
#         return F_T
#
#     def force_vs_conc(conc):
#         const = 1
#         E_er = 1
#         T = 1
#         F_conc = const * T * conc * E_er
#         return F_conc
#
#     def force_vs_E_er(E_er_list: list):
#         const = 1
#         conc = 1
#         T = 1
#         F_Eer = const * T * conc * E_er
#         return F_Eer

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

    # def calc_constant(self):
    #     R = 8.31446262
    #     amu_kg = self.amu / 1000
    #     v_rms = np.sqrt((3 * self.BLT_CONST * self.temp) / self.mass)
    #     F = (1 * R * self.temp) / (6.0221408e23 * amu_kg * v_rms * self.del_t)
    #
    #     v1 = self.INIT_VELOCITY + v_rms
    #     a = v_rms * self.time1
    #     v2 = v1 + a * self.time2
    #     del_v_square = v2 ** 2 - v1 ** 2
    #     del_e = (1 / 2) * self.mass * del_v_square
    #     enr_ex_rate = (del_e/np.abs(self.time2 - self.time1)) / self.del_t
    #     concentration = (3/2) * self.BLT_CONST * 3 / self.VOLUME
    #     temperature_from_velocity = (1 / 3) * ((self.mass * self.INIT_VELOCITY ** 2) / self.BLT_CONST)
    #     constant = F / (temperature_from_velocity * concentration * enr_ex_rate)
    #     return constant

    def collide(self) -> float:
        """

        :return:
        """
        v_rms = np.sqrt((3 * self.BLT_CONST * self.temp) / self.mass)
        v1 = self.INIT_VELOCITY + v_rms  # it has absorbed radiated energy itself while moving
        a = v_rms * self.time1  # an unknown particle with different acceleration about to hit V1 particle
        v2 = v1 + a * self.time2  # it is being hit by a particle thus received some energy from that's acceleration

        # del_v_square = (v2 - v1) ** 2
        v_final = v2 + a * (self.time1 / self.time2)  # a tiny amount of lost energy in rotation and randomness
        del_v_square = (v_final - v1) ** 2

        del_e = (1 / 2) * self.mass * del_v_square

        enr_ex_rate = (del_e/np.abs(self.time2 - self.time1)) / self.del_t
        concentration = (3/2) * self.BLT_CONST * self.temp / self.VOLUME
        # concentration = self.mass * self.SPACIFIC_HEAT * (T2 - T1) / self.VOLUME
        temperature_from_velocity = (1 / 3) * ((self.mass * del_v_square) / self.BLT_CONST)
        final_force = self.FORCE_CONST * temperature_from_velocity * concentration * enr_ex_rate
        # final_force = enr_ex_rate
        return final_force

    def pressure_force(self):
        n = 1   # mole of H
        R = 8.31446261815324
        T = 273.15
        V = 0.0224
        A = (np.cbrt(0.0224)) ** 2
        pressure_force = (n * R * T * A) / V
        print (pressure_force)
        return pressure_force


    def to_dict(self):
        return dict(
            temp=self.temp,
            mass=self.mass,
            time1=self.time1,
            time2=self.time2,
            del_t=self.del_t,
            force=self.force
        )



        # n = 1
        # R = 8.31446261815324
        # T = 273.15
        # V = 0.0224
        # A =  (np.cbrt(0.0224)) ** 2
        # pressure_force = n * R * T * A / V
        # print (pressure_force)
