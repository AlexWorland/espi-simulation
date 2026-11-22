import numpy as np

from espi_sim.rng import stream


def test_same_seed_and_name_reproduce_identical_draws():
    a = stream(7, "surface", 0).random(16)
    b = stream(7, "surface", 0).random(16)
    np.testing.assert_array_equal(a, b)


def test_different_names_give_independent_streams():
    a = stream(7, "surface", 0).random(1000)
    b = stream(7, "surface", 1).random(1000)
    c = stream(7, "camera", 0).random(1000)
    assert not np.array_equal(a, b)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.1
    assert abs(np.corrcoef(a, c)[0, 1]) < 0.1


def test_different_seeds_differ():
    assert not np.array_equal(stream(1, "x").random(8), stream(2, "x").random(8))


def test_stream_does_not_depend_on_other_streams_being_drawn():
    # Drawing from one stream must not shift another: this is what keeps the surface
    # identical when a new noise source is added to the simulator.
    expected = stream(3, "surface", 0).random(8)
    stream(3, "camera", 5).random(10_000)
    np.testing.assert_array_equal(stream(3, "surface", 0).random(8), expected)
