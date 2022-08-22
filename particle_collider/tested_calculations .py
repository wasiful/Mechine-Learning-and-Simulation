import numpy as np

# actual calculation
n = 1
R = 8.31446261815324
T = 273.15
V = 0.0224
A = (np.cbrt(0.0224)) ** 2
force_pressure = (n * R * T * A) / V
print(f"actual force from pressure exerted in one side by one mole gass = {force_pressure}")




time2 = 0.5
time1 = 0.3
BLT_CONST = 1.380649e-23
NA = 6.0221408e23
R = 8.31446262
# mass = 1.627e-27  # mass of 1 H atom in kg
mass = 0.00100784
temp = 273.15
del_t = time2 - time1
volume = 0.0224


v_rms = np.sqrt((3 * R * temp) / mass)
v1 = 0 + v_rms  # it has absorbed radiated energy itself while moving
a = v_rms * time1  # an unknown particle with different acceleration about to hit V1 particle
v2 = v1 + a * time2  # it is being hit by a particle thus received some energy from that's acceleration


# changed equation from force_pressure
mass_n_number_of_particles = mass * NA
periodic_mass = mass
changed_force =(mass_n_number_of_particles * R * 273.15) / (periodic_mass * v_rms * del_t)
print(f" changed force = {changed_force}")

# F = (mass * R * temp) / (1.627e-27 * v_rms * del_t)
number_of_particles = NA
F = number_of_particles * ((mass * v2 ** 2) / volume)
print(f"F= {F}")


del_v_square = (v2 - v1) ** 2


del_e = 0.5 * mass * del_v_square

enr_ex_rate = (del_e / np.abs(time2 - time1)) / del_t
concentration = ((3 / 2) * BLT_CONST * temp) / volume
# concentration = self.mass * self.SPACIFIC_HEAT * (T2 - T1) / self.VOLUME
temperature_from_velocity = (1 / 3) * ((mass * del_v_square) / BLT_CONST)

constant = F / (temperature_from_velocity * concentration * enr_ex_rate)
print(f"constant= {constant}")
final_force = 1 * temperature_from_velocity * concentration * enr_ex_rate
print(f"final_force= {final_force}")


# used calculation
temp = 273.15
mass = 1.627e-27
time1 = 2
time2 = 4
del_t = 3.6
BLT_CONST = 1.380649e-23
INIT_VELOCITY = 2600
VOLUME = 6.027e-31
FORCE_CONST = 1
v_rms = np.sqrt((3 * BLT_CONST * temp) / mass)
v1 = INIT_VELOCITY + v_rms  # it has absorbed radiated energy itself while moving
a = v_rms * time1  # an unknown particle with different acceleration about to hit V1 particle
v2 = v1 + a * time2  # it is being hit by a particle thus received some energy from that's acceleration

# del_v_square = (v2 - v1) ** 2
v_final = v2 + a * (time1 / time2)  # a tiny amount of lost energy in rotation and randomness
del_v_square = (v_final - v1) ** 2

del_e = (1 / 2) * mass * del_v_square

enr_ex_rate = (del_e / np.abs(time2 - time1)) / del_t
concentration = (3 / 2) * BLT_CONST * temp / VOLUME
# concentration = self.mass * self.SPACIFIC_HEAT * (T2 - T1) / self.VOLUME
temperature_from_velocity = (1 / 3) * ((mass * del_v_square) / BLT_CONST)
final_force = FORCE_CONST * temperature_from_velocity * concentration * enr_ex_rate
print(final_force)