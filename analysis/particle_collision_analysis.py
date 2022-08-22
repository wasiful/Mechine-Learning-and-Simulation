import numpy as np
from particle_collider.experiment import CollisionExperiment
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

np.random.seed(10)
sample_size = 10000


temp = np.linspace(273.15, 102730.15, sample_size)
mass = np.linspace(1.67262192e-27, 3.9525642e-25, sample_size)
time1 = np.random.uniform(1, 61, sample_size)
time2 = np.random.uniform(1, 61, sample_size)
del_t = np.random.uniform(1, 61, sample_size)

multi_exp = CollisionExperiment(
    temp=temp,
    mass=mass,
    time1=time1,
    time2=time2,
    del_t=del_t
)
multi_exp.run()
exp_df = multi_exp.to_df()

exp_df["force_log"] = exp_df["force"].apply(lambda x: np.log(x))
exp_df["mass_log"] = exp_df["mass"].apply(lambda x: np.log(x))


fig = plt.figure(figsize=(9, 4))
plt.subplot(1, 1, 1)
plt.plot(np.log(exp_df.mass), np.log(exp_df.force))
# plt.xticks(exp_df.mass)
# plt.yticks(exp_df.force)
plt.xlabel('mass')
plt.ylabel('Force')
plt.title(' Force vs temp graph')
plt.tight_layout()
plt.show()

# fig = plt.figure(figsize=(16, 9))
# plt.subplot(1, 1, 1)
# plt.plot(np.log(exp_df.mass), np.log(exp_df.force))
# # plt.xticks(exp_df.mass)
# # plt.yticks(exp_df.force)
# plt.xlabel('mass')
# plt.ylabel('Force')
# plt.title(' Force vs mass graph')
# plt.tight_layout()
# plt.show()

