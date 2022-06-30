#!/usr/bin/env python3
from std_msgs.msg import Header
import rclpy
import rclpy.node
import numpy as np
from sensor_msgs.msg import Image
import rclpy
import rclpy.node

import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from typing import Optional, Tuple
from pathlib import Path
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from .cost_map import CostMap, CostMapException
from .inflated_cost_map import InflatedCostMap
import tf_transformations
import cv2
import time


class ObstacleMapNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("obstacle_map")

        self.obstacles_sub = self.create_subscription(
            Marker,
            "/carla/ego_vehicle/center_lidar/obstacle_points/clustered",
            self.obstacles_sub_callback,
            qos_profile=10,
        )
        self.get_logger().info(f"Listening for obstacles on {self.obstacles_sub.topic}")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.costmap = CostMap(
            width=2000, height=2000, resolution=5, min_x=-850, min_y=-850
        )
        self.get_logger().info(
            f"CostMap initialized. map size={self.costmap.get_map_size()} | resolution={self.costmap.get_resolution()} | min_x={self.costmap._min_x_m} | min_y = {self.costmap._min_y_m}"
        )

    def obstacles_sub_callback(self, marker: Marker):
        if len(marker.points) == 0:
            return
        # transform all obstacle to map frame
        points: np.ndarray = self.marker_points_to_numpy(marker=marker).T
        trans: TransformStamped = self.get_transform(
            marker.header.frame_id, to_frame_rel="map"
        )
        if trans is None:
            return
        points_map = self.points_from_lidar_to_map(trans=trans, points=points)

        # plot obstacles onto cost map
        self.costmap.set_val_from_world_coords(points_map[0:2], 1.0)

        width = 200
        height = 200

        icm = InflatedCostMap.from_costmap(
            cost_map=self.costmap,
            width=width,
            height=height,
            Ox=trans.transform.translation.x - width // 2,
            Oy=trans.transform.translation.y - height // 2,
            obstacle_kernel_len=51,
            obstacle_kernel_std=15,
            obstacle_threshold=0.1,
        )
        start = time.time()
        msg = icm.to_occupancy_grid_msg(
            header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
        )
        print(1 / (time.time() - start))

        # cv2.imshow(
        #     "map",
        #     cv2.resize(icm.to_rgb_map(), dsize=(400, 400)),
        # )
        # cv2.waitKey(1)

    @staticmethod
    def points_from_lidar_to_map(
        trans: TransformStamped, points: np.ndarray
    ) -> np.ndarray:
        """transform points using the transfom msg

        Args:
            trans (TransformStamped): transformation that is going to be applied onto points
            points (np.ndarray): points to be transformed, nx3

        Returns:
            np.ndarray: 4xn
        """

        P = tf_transformations.quaternion_matrix(
            [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ]
        )
        T = [
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z,
        ]
        P[0:3, 3] = T
        points = np.vstack([points, np.ones(points.shape[1])])  # 4xn
        points = P @ points  # 4xn
        return points

    @staticmethod
    def marker_points_to_numpy(marker: Marker) -> np.ndarray:
        points = marker.points
        result = []
        for p in points:
            result.append([p.x, p.y, p.z])
        result = np.array(result)
        return result

    def get_transform(self, from_frame_rel, to_frame_rel) -> Optional[TransformStamped]:
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(to_frame_rel, from_frame_rel, now)
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {to_frame_rel} to {from_frame_rel}: {ex}"
            )
            return
        return trans

    def destroy_node(self):
        self.file.close()
        super().destroy_node(self)


def main(args=None):
    rclpy.init()
    node = ObstacleMapNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
