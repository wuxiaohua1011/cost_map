from cost_map.cost_map import CostMap, CostMapException
import numpy as np
from cost_map.inflated_cost_map import InflatedCostMap
import matplotlib.pyplot as plt

# cm = CostMap(width=10, height=10, resolution=1, min_x=-10, min_y=-10)
# icm = InflatedCostMap.from_costmap(cm, obstacle_kernel_len=3)
# icm.set_val_from_world_coords(coords=np.array([[-5, -5, -5], [-5, -4, -3]]))


cm = CostMap(width=20, height=20, resolution=10, min_x=-10, min_y=-10)
icm = InflatedCostMap.from_costmap(cm, obstacle_kernel_len=3)
icm.set_val_from_world_coords(coords=np.array([[-10, 0, 9.8], [-10, 0, 9.8]]))
np.save("./test/inflation_costmap/set_val_from_world_coords_2", icm.get_map())
# print(icm._map[-5:, -5:])
plt.imshow(icm.get_map())
plt.show()
