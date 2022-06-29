from cost_map.cost_map import CostMap
from cost_map.inflated_cost_map import InflatedCostMap
import numpy as np


def compare_two_array(m1, m2):
    return np.equal(m1, m2).all()


def test_map_creation_from_costmap():
    correct = np.load("./test/inflation_costmap/map_creation_test_data.npy")
    cm = CostMap(width=10, height=10, resolution=1, min_x=-10, min_y=-10)
    cm.set_val_from_world_coords(np.array([[-5, -4, -3], [-5, -4, -3]]), 0.5)
    icm = InflatedCostMap.from_costmap(cm, obstacle_kernel_len=3)
    assert compare_two_array(icm._map, correct), "Maps are not equal"


def test_set_val_from_world_coords():
    correct = np.load("./test/inflation_costmap/set_val_from_world_coords.npy")
    cm = CostMap(width=10, height=10, resolution=1, min_x=-10, min_y=-10)
    icm = InflatedCostMap.from_costmap(cm, obstacle_kernel_len=3)
    icm.set_val_from_world_coords(coords=np.array([[-5, -5, -5], [-5, -4, -3]]))
    assert compare_two_array(icm._map, correct), "Maps are not equal"

    cm = CostMap(width=5, height=5, resolution=1, min_x=0, min_y=0)
    icm = InflatedCostMap.from_costmap(cm, obstacle_kernel_len=3)
    icm.set_val_from_world_coords(coords=np.array([[0, 4], [0, 4]]))
    correct = np.load("./test/inflation_costmap/set_val_from_world_coords_5x5.npy")
    assert compare_two_array(icm._map, correct)

    cm = CostMap(width=20, height=20, resolution=10, min_x=-10, min_y=-10)
    icm = InflatedCostMap.from_costmap(cm, obstacle_kernel_len=3)
    icm.set_val_from_world_coords(coords=np.array([[-10, 0, 9.8], [-10, 0, 9.8]]))
    correct = np.load("./test/inflation_costmap/set_val_from_world_coords_2.npy")
    assert compare_two_array(icm._map, correct)
