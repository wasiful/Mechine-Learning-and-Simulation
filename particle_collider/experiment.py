from particle_collider.data_models_previous_eq import CollisionModel
import typing as t
import pandas as pd
import numpy as np

ColliderType = t.Union[t.List, np.ndarray]


class CollisionExperiment:

    def __init__(
            self,
            temp: ColliderType,
            mass: ColliderType,
            time1: ColliderType,
            time2: ColliderType,
            del_t: ColliderType
            ):
        self.temp = temp
        self.mass = mass
        self.time1 = time1
        self.time2 = time2
        self.del_t = del_t
        self.validate()
        self.exp_result: t.Optional[t.List[CollisionModel]] = []

    def run(self):
        sample_len = len(self.temp)
        for i in range(sample_len):
            unit = CollisionModel(
                temp=self.temp[i],
                mass=self.mass[i],
                time1=self.time1[i],
                time2=self.time2[i],
                del_t=self.del_t[i],
            )
            if (i+1) % 100 == 0:
                print(f"{i+1} unit has been calculated!")
            self.exp_result.append(unit)

    def to_df(self):
        if not self.exp_result:
            raise ValueError("Experiment has not been run, please execute run method first.")
        value_list = [item.to_dict() for item in self.exp_result]
        return pd.DataFrame(value_list)

    def validate(self):
        if (
            len(self.temp) == len(self.mass) and
            len(self.mass) == len(self.time1) and
            len(self.time1) == len(self.time2) and
            len(self.time2) == len(self.del_t)
        ):
            pass
        ValueError("Provided list are not equal length.")
