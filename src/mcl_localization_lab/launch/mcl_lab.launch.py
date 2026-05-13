from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_share = get_package_share_directory('mcl_localization_lab')

    world_file = os.path.join(
        pkg_share,
        'worlds',
        'warehouse_mcl.sdf'
    )

    gazebo_process = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world_file],
        output='screen'
    )

    gazebo_ros_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gazebo_ros_bridge',
        output='screen',
        arguments=[
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
        ]
    )

    mcl_node = Node(
        package='mcl_localization_lab',
        executable='particle_localizer',
        name='particle_localization_node',
        output='screen'
    )

    return LaunchDescription([
        gazebo_process,
        gazebo_ros_bridge,
        mcl_node,
    ])