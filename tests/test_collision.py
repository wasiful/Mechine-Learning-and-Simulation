from particle_collider.data_models import CollisionModel


def test_collision_init():
    collision_unit = CollisionModel(1.0, 2.3, 4.4, 5.5, 6)
    assert collision_unit.temp == 1.0
    assert collision_unit.mass == 2.3
    assert collision_unit.time1 == 4.4
    assert collision_unit.time2 == 5.5
    assert collision_unit.del_t == 6


def test_collision_collide_return_float():
    collision_unit = CollisionModel(1.0, 2.3, 4.4, 5.5, 6)
    result = collision_unit.collide()
    assert isinstance(result, float)


def test_collision_return_valid_force():
    expected = 8.02
    collision_unit = CollisionModel(
        temp=273.15,
        mass=1.672e-27,
        time1=2.0,
        time2=4.0,
        del_t=3.6,
    )
    actual = collision_unit.collide()
    actual = round(actual*1e6, 2)
    assert expected == actual


def test_collision_post_init():
    expected = 8.02
    collision_unit = CollisionModel(
        temp=273.15,
        mass=1.672e-27,
        time1=2.0,
        time2=4.0,
        del_t=3.6,
    )
    actual = collision_unit.force
    actual = round(actual * 1e6, 2)
    assert expected == actual


def test_collision_model_to_dict():
    collision_unit = CollisionModel(
        temp=273.15,
        mass=1.672e-27,
        time1=2.0,
        time2=4.0,
        del_t=3.6,
    )
    expected = dict(
        temp=273.15,
        mass=1.672e-27,
        time1=2.0,
        time2=4.0,
        del_t=3.6,
        force=8.024616802680115e-06
    )
    actual = collision_unit.to_dict()
    assert expected == actual



