from object_collider.many_object_dev import INIT_CONFIG, init_simulation_result, SimulationResult


def test_simulation_result_init():
    result = init_simulation_result(
        INIT_CONFIG.N,
        INIT_CONFIG.T,
        INIT_CONFIG.dt,
        INIT_CONFIG.dim
    )
    assert isinstance(result, SimulationResult)
