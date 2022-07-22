from particle_collider.experiment import CollisionExperiment


def test_collision_experiment_init():
    temp = [1.1, 1.2]
    mass = [2, 3]
    time1 = [4, 1]
    time2 = [6, 2]
    del_t = [5, 3]

    multi_exp = CollisionExperiment(
        temp=temp,
        mass=mass,
        time1=time1,
        time2=time2,
        del_t=del_t
    )
    assert multi_exp.temp == temp
    assert multi_exp.mass == mass
    assert multi_exp.time1 == time1
    assert multi_exp.time2 == time2


def test_collision_experiment_run():
    temp = [1.1, 1.2]
    mass = [2, 3]
    time1 = [4, 1]
    time2 = [6, 2]
    del_t = [5, 3]

    multi_exp = CollisionExperiment(
        temp=temp,
        mass=mass,
        time1=time1,
        time2=time2,
        del_t=del_t
    )
    multi_exp.run()

    assert len(multi_exp.exp_result) == len(temp)
    assert len(multi_exp.exp_result) == len(mass)
    assert len(multi_exp.exp_result) == len(time1)
    assert len(multi_exp.exp_result) == len(time2)
    assert len(multi_exp.exp_result) == len(del_t)


def test_collision_exp_to_df():
    temp = [1.1, 1.2]
    mass = [2, 3]
    time1 = [4, 1]
    time2 = [6, 2]
    del_t = [5, 3]
    multi_exp = CollisionExperiment(
        temp=temp,
        mass=mass,
        time1=time1,
        time2=time2,
        del_t=del_t
    )
    multi_exp.run()
    exp_df = multi_exp.to_df()
    assert len(exp_df) == len(temp)
    assert len(exp_df) == len(mass)
    assert len(exp_df) == len(time1)
    assert len(exp_df) == len(time2)
    assert len(exp_df) == len(del_t)


