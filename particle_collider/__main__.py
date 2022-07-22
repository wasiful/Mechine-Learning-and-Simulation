import numpy as np
from particle_collider.experiment import CollisionExperiment

if __name__ == "__main__":
    np.random.seed(10)
    temp = np.linspace(273.15, 1273.15, 1000)
    mass = np.linspace(1.67262192e-27, 3.9525642e-25, 1000)
    time1 = np.random.uniform(1, 601, 1000)
    time2 = np.random.uniform(1, 601, 1000)
    del_t = np.random.uniform(1, 11, 1000)

    multi_exp = CollisionExperiment(
        temp=temp,
        mass=mass,
        time1=time1,
        time2=time2,
        del_t=del_t
    )
    multi_exp.run()
    exp_df = multi_exp.to_df()
    print(exp_df.head())
