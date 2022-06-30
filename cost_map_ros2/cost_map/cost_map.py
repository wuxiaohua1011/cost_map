import numpy as np
from typing import Optional, Tuple, List
import cv2
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point


class CostMap:
    def __init__(
        self,
        width: int,
        height: int,
        resolution: float,
        min_x: int = 0,
        min_y: int = 0,
    ) -> None:
        """_summary_

        Args:
            width (int): width of the map
            height (int): height of the map
            resolution (float): 1 meter represents 'resolution' number of cells
            min_x (int, optional): minimum x direction. Defaults to 0.
            min_y (int, optional): minimum y direction. Defaults to 0.
        """
        if width <= 0 or height <= 0 or resolution <= 0:
            raise CostMapException("Map initialization failed")
        self._width_m = width
        self._height_m = height
        self._min_x_m = min_x
        self._min_y_m = min_y
        self._resolution = resolution
        self._map: np.ndarray = self._init_map(
            self._width_m,
            self._height_m,
            self._min_x_m,
            self._min_y_m,
            self._resolution,
        )

    @staticmethod
    def _init_map(width_m, height_m, min_x_m, min_y_m, resolution) -> np.ndarray:
        map_width: int = int((width_m) * resolution)
        map_height: int = int((height_m) * resolution)
        assert map_width > 0 and map_height > 0, "Please increase resolution"
        m = np.zeros(shape=(map_width, map_height), dtype=np.int8)
        return m

    def get_map(self, down_sample: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if down_sample is not None:
            return cv2.resize(self._map, dsize=down_sample)
        return self._map.copy()

    def get_map_size(self) -> Tuple[int, int]:
        return self._map.shape

    def get_map_bound_in_meter(self):
        max_m, max_y = self._min_x_m + self._width_m, self._min_y_m + self._height_m
        return self._min_x_m, self._min_y_m, max_m, max_y

    def world_to_map_coord(self, x_meter: float, y_meter: float, no_bound=False) -> int:
        if x_meter < self._min_x_m or y_meter < self._min_y_m:
            raise CostMapException(f"{x_meter, y_meter} is out of bound")
        xm = int((x_meter - self._min_x_m) * self._resolution)
        ym = int((y_meter - self._min_y_m) * self._resolution)
        if no_bound is True:
            return (xm, ym)
        elif 0 <= xm < self._map.shape[0] and 0 <= ym < self._map.shape[1]:
            return (xm, ym)
        else:
            raise CostMapException(
                f"Requested size {(x_meter, y_meter)} is greater than map size {self._map.shape}"
            )

    def get_val_map_coord(self, mx: int, my: float):
        assert (
            0 <= mx < self.get_map_size()[0] and 0 <= my < self.get_map_size()[1]
        ), "Requested coordinate outside of map size"
        return self._map[mx][my]

    def get_val_world_coord(self, x_meter: float, y_meter: float):
        x, y = self.world_to_map_coord(x_meter=x_meter, y_meter=y_meter)
        return self.get_val_map_coord(x, y)

    def world_to_map_coords(
        self, xs_meter: np.ndarray, ys_meter: np.ndarray, no_bound=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        if type(xs_meter) == list:
            xs_meter = np.array(xs_meter)
        if type(ys_meter) == list:
            ys_meter = np.array(ys_meter)

        if (xs_meter < self._min_x_m).all() or (ys_meter < self._min_y_m).all():
            raise CostMapException(f"{xs_meter, ys_meter} is out of bound")

        xs_map: np.ndarray = ((xs_meter - self._min_x_m) * self._resolution).astype(int)
        ys_map: np.ndarray = ((ys_meter - self._min_y_m) * self._resolution).astype(int)

        if no_bound is True:
            return xs_map, ys_map
        elif (
            np.logical_and(0 <= xs_map, xs_map < self._map.shape[0]).all()
            and np.logical_and(0 <= ys_map, ys_map < self._map.shape[1]).all()
        ):
            return xs_map, ys_map
        else:
            raise CostMapException(
                f"{np.vstack([xs_meter, ys_meter]).T} has component greater than map size {self._map.shape}"
            )

    def get_area(self, wx: float, wy: float, w_width: float, w_height: float):
        mx, my = self.world_to_map_coord(x_meter=wx, y_meter=wy, no_bound=False)
        m_width, m_height = self._resolution * w_width, self._resolution * w_height

        x_min, y_min = int(max(0, mx)), int(max(0, my))
        x_max, y_max = (
            int(min(self.get_map_size()[0] - 1, mx + m_width)),
            int(min(self.get_map_size()[1] - 1, my + m_height)),
        )

        return self._map[x_min:x_max, y_min:y_max]

    def set_val_from_world_coord(self, x_meter, y_meter, v):
        x, y = self.world_to_map_coord(x_meter=x_meter, y_meter=y_meter, no_bound=True)
        if 0 <= x < self._map.shape[0] and 0 <= y < self._map.shape[1]:
            self._map[x][y] = float(v)
            return True
        else:
            return False

    def set_val_from_world_coords(self, coords: np.ndarray, val: float):
        """Set multiple coords to val, where coords are in meters.

        Args:
            coords (np.ndarray): coordiates to set to val, in meters, a nx2 array
            val (np.int8): single value to set it to
            ignore_outside (bool, optional): ignore values of outside the map. Defaults to False.
        """
        xs = coords[0, :]
        ys = coords[1, :]
        xs_map, ys_map = self.world_to_map_coords(xs, ys, no_bound=True)
        coords = np.vstack([xs_map, ys_map]).T
        coords = coords[
            (0 <= coords[:, 0])
            & (coords[:, 0] < self._map.shape[0])
            & (0 <= coords[:, 1])
            & (coords[:, 1] < self._map.shape[1])
        ]  # select only the points that is inside map

        self._map[coords[:, 0], coords[:, 1]] = val

    def get_map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        return (
            mx / self._resolution - abs(self._min_x_m),
            my / self._resolution - abs(self._min_y_m),
        )

    def get_resolution(self) -> float:
        return self._resolution

    def to_occupancy_grid_msg(self, header: Header) -> OccupancyGrid:
        og: OccupancyGrid = OccupancyGrid(header=header)
        info: MapMetaData = MapMetaData(
            resolution=float(self._resolution),
            width=int(self._map.shape[0]),
            height=int(self._map.shape[1]),
            origin=Pose(
                position=Point(
                    x=float(self._min_x_m), y=float(self._min_y_m), z=float(0.0)
                )
            ),
        )
        og.info = info
        # normalized_map = (self._map * 255).astype(np.int8)
        # rescaled_map: np.ndarray = (normalized_map * 255).astype(np.int8)
        data = (
            self._map.reshape((int(self._map.shape[0] * self._map.shape[1])))
            .astype(np.int8)
            .tolist()
        )
        og.data = data
        return og


class CostMapException(Exception):
    pass
