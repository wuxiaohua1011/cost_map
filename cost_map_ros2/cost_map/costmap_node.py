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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pathlib import Path
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from .cost_map import CostMap, CostMapException
from .inflated_cost_map import InflatedCostMap
import tf_transformations
import cv2
import time

from costmap_msgs.msg import CostMapMsg
from costmap_msgs.srv import GetCostMap, GetInflatedCostMap
from example_interfaces.srv import AddTwoInts


class ObstacleMapNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("obstacle_map")

        self.declare_parameter("global_costmap_width", 2000)
        self.declare_parameter("global_costmap_height", 2000)
        self.declare_parameter("global_costmap_min_x", -850.0)
        self.declare_parameter("global_costmap_min_y", -850.0)
        self.declare_parameter("resolution", 0.2)  # meter/cell
        self.declare_parameter("map_frame_id", "map")

        self.obstacles_sub = self.create_subscription(
            Marker,
            "/carla/ego_vehicle/center_lidar/obstacle_points/clustered",
            self.obstacles_sub_callback,
            qos_profile=10,
        )
        self.get_logger().info(f"Listening for obstacles on {self.obstacles_sub.topic}")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Note: inverse the resolution b/c in normal life, we say resolution is in meter/cell, but in implementation, i did cell/meter
        self.costmap = CostMap(
            width=self.get_parameter("global_costmap_width")
            .get_parameter_value()
            .integer_value,
            height=self.get_parameter("global_costmap_height")
            .get_parameter_value()
            .integer_value,
            resolution=int(
                1 / self.get_parameter("resolution").get_parameter_value().double_value
            ),
            min_x=self.get_parameter("global_costmap_min_x")
            .get_parameter_value()
            .double_value,
            min_y=self.get_parameter("global_costmap_min_y")
            .get_parameter_value()
            .double_value,
        )
        self.get_logger().info(
            f"CostMap initialized. map= {self.costmap._width_m}m x {self.costmap._height_m}m | resolution={self.costmap.get_resolution()} cell/meter | min_x={self.costmap._min_x_m}m | min_y = {self.costmap._min_y_m}m | Map size in cells = {self.costmap.get_map_size()}"
        )

        self.costmap_srv = self.create_service(
            GetCostMap, "costmap_srv", self.on_costmap_srv_callback
        )

        self.inflated_costmap_srv = self.create_service(
            GetInflatedCostMap, "inflated_costmap_srv", self.get_inflated_map_callback
        )

        self.map_frame_id = (
            self.get_parameter("map_frame_id").get_parameter_value().string_value
        )

        self.inflated_map_publisher = self.create_publisher(Image, "inflated_map", 10)
        self.bridge = CvBridge()

    def on_costmap_srv_callback(self, request, response):
        header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id=self.map_frame_id
        )
        m: CostMapMsg = self.costmap.to_ros_msg(header=header)
        response.map = m
        return response

    def get_inflated_map_callback(self, request, response):
        header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id=self.map_frame_id
        )
        icm = InflatedCostMap.from_costmap(
            cost_map=self.costmap,
            width=request.width,
            height=request.height,
            Ox=request.min_x,
            Oy=request.min_y,
            obstacle_kernel_len=request.obstacle_kernel_len,
            obstacle_kernel_std=request.obstacle_kernel_std,
            obstacle_threshold=request.obstacle_threshold,
        )
        m: CostMapMsg = icm.to_ros_msg(header=header)
        response.map = m
        self.get_logger().info(f"Response sent {self.get_clock().now()}")
        return response

    def obstacles_sub_callback(self, marker: Marker):
        """
            Given a list of obstacles in the relative frame, project it onto the costmap that is in the map frame

        Args:
            marker (Marker): _description_
        """
        if len(marker.points) == 0:
            return
        # transform all obstacle to map frame
        points: np.ndarray = self.marker_points_to_numpy(marker=marker).T

        if marker.header.frame_id != self.map_frame_id:
            trans: TransformStamped = self.get_transform(
                marker.header.frame_id, to_frame_rel=self.map_frame_id
            )
            if trans is None:
                return
            points = self.points_from_lidar_to_map(trans=trans, points=points)

        # plot obstacles onto cost map
        self.costmap.set_val_from_world_coords(points[0:2], 1.0)

        # header = Header(
        #     stamp=self.get_clock().now().to_msg(), frame_id=self.map_frame_id
        # )
        # width = 100
        # height = 100
        # icm = InflatedCostMap.from_costmap(
        #     cost_map=self.costmap,
        #     width=width,
        #     height=height,
        #     Ox=trans.transform.translation.x - width // 2,
        #     Oy=trans.transform.translation.y - height // 2,
        #     obstacle_kernel_len=51,
        #     obstacle_kernel_std=15,
        #     obstacle_threshold=0.5,
        # )
        # colored = icm.to_rgb_map()
        # cv2.imshow("colored", colored)
        # cv2.waitKey(1)
        # image = self.bridge.cv2_to_imgmsg(colored)
        # self.inflated_map_publisher.publish(image)

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
