from cost_map.cost_map import CostMap, CostMapException
import numpy as np


def compare_two_array(m1, m2):
    return np.equal(m1, m2).all()


def test_map_creation_normal():
    cm = CostMap(width=10, height=10, resolution=1)

    assert cm.get_map().shape == (10, 10)
    assert cm.get_map().dtype == np.float32
    assert compare_two_array(cm.get_map(), np.zeros(shape=(10, 10), dtype=np.float32))

    cm = CostMap(width=10, height=10, resolution=2)
    assert cm.get_map().shape == (20, 20)

    cm = CostMap(width=10, height=10, resolution=2, min_x=-5, min_y=-5)
    assert cm.get_map().shape == (20, 20)

    cm = CostMap(width=10, height=10, resolution=10, min_x=-5, min_y=-5)
    assert cm.get_map().shape == (100, 100)


def test_map_creation_abnormal():
    try:
        cm = CostMap(width=-1, height=-1, resolution=1)
        cm = CostMap(width=0, height=0, resolution=1)
        assert (
            True == False
        ), "Cannot create a map of size width=-1, height=-1, resolution=1"
    except:
        pass


def test_world_to_map_coord():
    cm = CostMap(width=10, height=10, resolution=10, min_x=-5, min_y=-8)
    assert cm.world_to_map_coord(x_meter=0, y_meter=0) == (50, 80)
    try:
        cm.world_to_map_coord(2.5, 2)
        assert False, "should be out of bound"
    except CostMapException:
        pass

    cm = CostMap(width=10, height=10, resolution=10, min_x=0, min_y=0)
    try:
        cm.world_to_map_coord(0, 101, no_bound=False)
        cm.world_to_map_coord(-1, 0, no_bound=False)
        assert False, "Cannot get map beyond bound"
    except CostMapException:
        pass

    assert cm.world_to_map_coord(0, 101, no_bound=True) == (0, 1010)


def test_get_map():
    cm = CostMap(width=10, height=10, resolution=1)
    assert compare_two_array(cm.get_map(), np.zeros(shape=(10, 10)))
    assert compare_two_array(cm.get_map(down_sample=(5, 5)), np.zeros(shape=(5, 5)))


def test_get():
    cm = CostMap(width=10, height=10, resolution=1)
    cm.set_val_from_world_coord(5, 5, 20)
    assert cm.get_val_world_coord(5, 5) == 20

    try:
        cm.get_val_world_coord(11, 11)
    except CostMapException:
        pass


def test_get_map_coords():
    cm = CostMap(width=10, height=10, resolution=1, min_x=-5, min_y=-10)
    mxs, mys = cm.world_to_map_coords(xs_meter=[-5, -4, -3], ys_meter=[-10, -9, -8])
    assert compare_two_array(mxs, np.array([0, 1, 2]))
    assert compare_two_array(mys, np.array([0, 1, 2]))
    mxs, mys = cm.world_to_map_coords(
        xs_meter=[3, 4, 5], ys_meter=[0, -1, -2], no_bound=True
    )
    assert compare_two_array(mxs, np.array([8, 9, 10]))
    assert compare_two_array(mys, np.array([10, 9, 8]))


def test_set_multi():
    cm = CostMap(width=10, height=10, resolution=1, min_x=-5, min_y=-10)
    coords = np.array([[-5, -9, -8], [-4, -10, -3]])
    cm.set_val_from_world_coords(coords=coords, val=1)

    correct_map = np.zeros(shape=(10, 10))
    correct_map[0][6] = 1
    assert compare_two_array(m1=cm.get_map(), m2=correct_map)

    cm.set_val_from_world_coords(coords=np.array([[-5, -5], [20, 20]]), val=1)
    assert compare_two_array(m1=cm.get_map(), m2=correct_map)
