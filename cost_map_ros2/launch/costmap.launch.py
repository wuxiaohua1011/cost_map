from launch import LaunchDescription
from launch_ros.actions import Node
import launch


def generate_launch_description():
    node = Node(
        package="cost_map",
        executable="costmap_node",
        name="costmap_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "global_costmap_width": launch.substitutions.LaunchConfiguration(
                    "global_costmap_width"
                ),
                "global_costmap_height": launch.substitutions.LaunchConfiguration(
                    "global_costmap_height"
                ),
                "global_costmap_min_x": launch.substitutions.LaunchConfiguration(
                    "global_costmap_min_x"
                ),
                "global_costmap_min_y": launch.substitutions.LaunchConfiguration(
                    "global_costmap_min_y"
                ),
                "resolution": launch.substitutions.LaunchConfiguration("resolution"),
                "map_frame_id": launch.substitutions.LaunchConfiguration(
                    "map_frame_id"
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                name="global_costmap_width",
                default_value="2000",
                description="Width in meters",
            ),
            launch.actions.DeclareLaunchArgument(
                name="global_costmap_height",
                default_value="2000",
                description="length in meters",
            ),
            launch.actions.DeclareLaunchArgument(
                name="global_costmap_min_x",
                default_value="-850.0",
            ),
            launch.actions.DeclareLaunchArgument(
                name="global_costmap_min_y",
                default_value="-850.0",
            ),
            launch.actions.DeclareLaunchArgument(
                name="resolution", default_value="0.2", description="meter per cell"
            ),
            launch.actions.DeclareLaunchArgument(
                name="map_frame_id",
                default_value="map",
            ),
            node,
        ]
    )
