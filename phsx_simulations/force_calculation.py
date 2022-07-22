import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from dataclasses import dataclass



# properties of proton or hydrogen ion
proton_repulsion_force = 1.9e-44  # proton vs proton
mass_of_hydrogen_ion = 1.67262192e-27  # kg
rest_energy_of_proton = 1.5028e-10  # Joule

density_of_proton = 1.67262192e-27 / 2.5e-45  # mass / volume
charge_density_of_proton = 1.6e-19 / 2.5e-45  # charge / volume
energy_density_of_proton = 1.5028e-10 / 2.5e-45



# initial values for the simulation
initial_Temp = 273.15  # 0 degree c
m = 1.67262192e-27
v = 1.845e3    # h2 atom velocity at 0 degree c
Kb = 1.380649e-23
# volume = 1
const = 1
# C = 299792458
T_list = np.linspace(273.15, 1273.15, 1000)
m_list = np.linspace(1.67262192e-27, 3.9525642e-25, 1000)


# a list of temperature and a list of mass
# problem with time and velocity
# the velocity of hydrogen molecule at different temperature given in the list
def calc_v_rms(T_list: list):
    v_rms_list = []
    # list of temperature given
    for T in T_list:
        v_rms = np.sqrt((3 * Kb * T) / m)
        v_rms.append(v_rms_list)
    return v_rms_list


# a list of velocity is required and a list which has the first value as initial velocity at 0 degree
def calc_first_velocity(v_rms_list: list, v: float):
    v1_list = []
    # add new velocity to the initial velocity (first value)  """ v """
    for v_rms in v_rms_list:
        v1 = v + v_rms
        v1.append(v1_list)
    return v1_list


# v2 = v1 + a * t      v = u+at
def calc_second_velocity(v1_list: list):
    time = np.linspace(1, 600, 1000)
    acceleration = np.linspace(1, 1000, 1000)  # start, stop, step
    v2_list = []
    for v1 in v1_list:
        for a in acceleration:
            for t in time:
                v2 = v1 + a * t
                v2.append(v2_list)
    return v2_list


def calc_del_E(v1_list: list, v2_list: list):
    del_E_list = []
    for v1 in v1_list:
        for v2 in v2_list:
            del_v_square = v2 ** 2 - v1 ** 2
            del_E = (1/2) * m * del_v_square
            del_E.append(del_E_list)
    return del_E_list


def calc_E_er(del_E_list: list, t2, t1):
    del_t = t2 - t1  # t2 is the second index t2 = i+1 th, and t1 is first index t1 = i th del_t is constant
    E_er_list = []
    for del_E in del_E_list:
        E_er = del_E / del_t
        E_er.append(E_er_list)
    return E_er_list


def calc_temp(m_list: list, v_rms_list: list, Kb):
    for m in m_list:
        for v in v_rms_list:
            T = (1/3) * ((m * v ** 2) / Kb)  # calculated temperature
            T.append(T_list)
    return T_list


def calc_conc_field(m: float, m_list: list, v2: float, v2_list: list, volume):
    conc_list = []
    # here velocity is constant initial velocity
    for m in m_list:
        conc = ((1/2) * m * v2 ** 2) / volume
        conc.append(conc_list)

        # here m is constant initial mass
        # for v2 in v2_list:
        #     conc = ((1/2) * m * v2 ** 2) / volume       # energy devided by volume, energ density
        #     conc.append(conc_list)
    return conc_list


def force_vs_T(T_list: list):
    F_T_list = []
    const  = 1
    conc = 1
    E_er = 1
    for T in T_list:
        F = const * T * conc * E_er
        F.append(F_T_list)
    return F_T_list


def force_vs_conc(conc_list: list):
    F_conc_list = []
    const  = 1
    E_er = 1
    T = 1
    for conc in conc_list:
        F = const * T * conc * E_er
        F.append(F_conc_list)
    return F_conc_list


def force_vs_E_er(E_er_list: list):
    F_Eer_list = []
    const  = 1
    conc = 1
    T = 1
    for E_er in E_er_list:
        F = const * T * conc * E_er
        F.append(F_Eer_list)
    return F_Eer_list


def force_vs_all():
    F_all_list = []
    T_list = np.linspace(273.15, 1273.15, 1000)
    # m_list = np.linspace(1.67262192e-27, 3.9525642e-25, 1000)
    m_list = np.random.uniform(1.67262192e-27, 3.9525642e-25, 1000)
    # time_list = np.linspace(1, 600, 1000)
    time_list = np.ramdom.uniform(1, 601, 1000)
    del_t = 2
    acceleration_list = np.linspace(1, 1000, 1000)
    for T in T_list:
        for m in m_list:
            for t in time_list:
                            v_rms = np.sqrt((3 * Kb * T) / m)
                            v1 = v + v_rms
                            a = v_rms / t
                            v2 = v1 + a * t
                            del_v_square = v2 ** 2 - v1 ** 2
                            del_E = (1 / 2) * m * del_v_square
                            E_er = del_E / del_t
                            conc = ((1 / 2) * m * v2 ** 2) / volume
                            T = (1 / 3) * ((m * v ** 2) / Kb)
                            F = const * T * conc * E_er
                            F.append(F_all_list)
    return F_all_list


@dataclass
class CollisionScenario:
    temp: float
    mass: float
    time: float
    del_t: float
    boltz_const = 1.380649e-23
    init_velocity = 1.845e3
    volume = 1
    const = 1

    def collide(self) -> float:
        v_rms = np.sqrt((3 * self.boltz_const * self.temp) / self.mass)
        v1 = self.init_velocity + v_rms
        a = v_rms / self.time
        v2 = v1 + a * self.time
        del_v_square = v2 ** 2 - v1 ** 2
        del_e = (1 / 2) * self.mass * del_v_square
        enr_ex_rate = del_e / self.del_t
        concentration = ((1 / 2) * self.mass * v2 ** 2) / self.volume
        temperature_from_velocity = (1 / 3) * ((self.mass * v2 ** 2) / self.boltz_const)
        final_force = self.const * temperature_from_velocity * concentration * enr_ex_rate
        return final_force


test_scenario = CollisionScenario(1.0, 2.3, 4.4, 5.5)
test_scenario.collide()


def main_math_func(first, second, third):
    result = 2*first + 3*second + 5*third
    return result



fig = plt.figure()
plt.subplot(1, 1, 1)
# plt.clear()
# x, y data
x = np.arange(0, 10 * np.pi, 0.01)
y = np.arange(0, 12 * np.pi, 0.01)
data_skip = 50
plt.xlabel('mass')
plt.ylabel('Force')
plt.title(' Force vs mass graph')


def animate(i):
    plt.plot(x[i:i+data_skip], y[i:i+data_skip])


anim = FuncAnimation(fig, animate, frames=200, interval=0.2 )
plt.show()
# anim.save('continuousSineWave.mp4', writer='ffmpeg', fps=30)


# F(y axis) vs increasing mass(x axis), F vs concentration, F vs energy exchange..... three separate curves

# in this simulation directional change happens after colision
# a up moving object and a right to left moving object if collides(90 degree colission)
# then the up moving object starts moving in the direction of........resultant....... (need to calculate)
# the right to left moving object will loose as much velocity the up moving object will gain
# two objects if straightly(180 degree) collides with same velocity then their velocity become 0

# a function that calculates resultant
# a function that detects collision if particles overlap then velocity change in resultant angle
# t1 is the beginning of the v1 velocity time
# and t2 is the end time when the object coledes and gains V2 velocity and a new angle
