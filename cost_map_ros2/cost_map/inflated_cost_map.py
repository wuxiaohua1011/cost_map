import numpy as np
from .cost_map import CostMap
from typing import Optional
from scipy import signal
import time
import cv2


class InflatedCostMap(CostMap):
    def __init__(
        self,
        width: int,
        height: int,
        resolution: float,
        min_x: int = 0,
        min_y: int = 0,
        obstacle_kernel_len: int = 3,
        obstacle_kernel_std: float = 1,
        goal_kernel_len: int = 3,
        goal_kernel_std: float = 1,
        obstacle_threshold: float = 0.5,
    ) -> None:
        super().__init__(width, height, resolution, min_x, min_y)
        assert obstacle_kernel_len % 2 == 1, "Kernel has to be odd"
        assert goal_kernel_len % 2 == 1, "Kernel has to be odd"
        self.obstacle_kernel_len: int = obstacle_kernel_len
        self.obstacle_kernel_std: int = obstacle_kernel_std
        self.goal_kernel_len: float = goal_kernel_len
        self.goal_kernel_std: int = goal_kernel_std
        self.obstacle_threshold: float = obstacle_threshold

        self.obstacle_kernel: np.ndarray = self.construct_kernel(
            self.obstacle_kernel_len, self.obstacle_kernel_std
        )
        self.goal_kernel: np.ndarray = self.construct_kernel(
            self.goal_kernel_len, self.goal_kernel_std
        )

    @staticmethod
    def construct_kernel(kernel_len, std) -> np.ndarray:
        """Returns a 2D Gaussian kernel array."""
        gkern1d = signal.gaussian(kernel_len, std=std).reshape(kernel_len, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d

    def set_val_from_world_coord(self, x_meter, y_meter, is_obstacle=True):
        raise NotImplementedError

    def set_val_from_world_coords(self, coords: np.ndarray, is_obstacle=True, gain=1):
        xs_map, ys_map = self.world_to_map_coords(
            xs_meter=coords[0, :], ys_meter=coords[1, :]
        )
        coords = np.vstack([xs_map, ys_map]).T
        self.set_val_from_map_coords(coords=coords, is_obstacle=is_obstacle, gain=gain)

    def set_val_from_map_coords(self, coords: np.ndarray, is_obstacle=True, gain=1):
        kernel = self.obstacle_kernel if is_obstacle else self.goal_kernel
        offset = int(np.ceil(kernel.shape[0] / 2))
        for x, y in coords:
            map_minx = max(0, x - offset + 1)
            map_miny = max(0, y - offset + 1)
            map_maxx = min(x + offset, self.get_map_size()[0])
            map_maxy = min(y + offset, self.get_map_size()[1])

            kernel_min_x = 0 if x - offset > 0 else abs(int(np.ceil((x - offset)))) - 1
            kernel_min_y = 0 if y - offset > 0 else abs(int(np.ceil((y - offset)))) - 1
            kernel_max_x = (
                len(kernel)
                if x + offset < self.get_map_size()[0]
                else len(kernel) - (y + offset - self.get_map_size()[0])
            )
            kernel_max_y = (
                len(kernel)
                if y + offset < self.get_map_size()[1]
                else len(kernel) - (y + offset - self.get_map_size()[1])
            )

            k = kernel[kernel_min_x:kernel_max_x, kernel_min_y:kernel_max_y]
            area = self._map[map_minx:map_maxx, map_miny:map_maxy]
            try:
                self._map[map_minx:map_maxx, map_miny:map_maxy] = area + k * gain
            except Exception as e:
                print(f"Unable to impose this obstacle: {e}")

    def to_rgb_map(self) -> np.ndarray:
        img = self.get_map()

        img = img / np.max(img)
        img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return img

    @classmethod
    def from_costmap(
        cls,
        cost_map: CostMap,
        width: Optional[int] = None,
        height: Optional[int] = None,
        Ox: Optional[float] = None,
        Oy: Optional[float] = None,
        resolution: Optional[int] = None,
        obstacle_kernel_len: int = 3,
        obstacle_kernel_std: int = 1,
        goal_kernel_len: int = 3,
        goal_kernel_std: int = 1,
        obstacle_threshold: float = 0.5,
    ):
        if width is None:
            width = cost_map._width_m
        if height is None:
            height = cost_map._height_m
        if Ox is None:
            Ox = cost_map.get_map_bound_in_meter()[0]
        if Oy is None:
            Oy = cost_map.get_map_bound_in_meter()[1]
        if resolution is None:
            resolution = cost_map.get_resolution()
        assert (
            Ox >= cost_map.get_map_bound_in_meter()[0]
            and Oy >= cost_map.get_map_bound_in_meter()[1]
            and width <= cost_map._width_m
            and height <= cost_map._height_m
        ), "Copy failed due to not in bound"

        icm = InflatedCostMap(
            width=width,
            height=height,
            resolution=resolution,
            min_x=Ox,
            min_y=Oy,
            obstacle_kernel_len=obstacle_kernel_len,
            obstacle_kernel_std=obstacle_kernel_std,
            goal_kernel_len=goal_kernel_len,
            goal_kernel_std=goal_kernel_std,
        )
        regional_costmap = cost_map.get_area(
            wx=Ox, wy=Oy, w_width=width, w_height=height
        )

        obs_map_indices = np.where(regional_costmap >= obstacle_threshold)
        icm.set_val_from_map_coords(np.vstack(obs_map_indices).T, is_obstacle=True)

        return icm
