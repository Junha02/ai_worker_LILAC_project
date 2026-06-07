from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    config = str(Path(get_package_share_directory("lilac_ros")) / "config" / "lilac.yaml")
    return LaunchDescription([
        Node(package="lilac_ros", executable="language_manager", parameters=[config], output="screen"),
        Node(package="lilac_ros", executable="latent_input", parameters=[config], output="screen"),
        Node(package="lilac_ros", executable="policy", parameters=[config], output="screen"),
        Node(package="lilac_ros", executable="ik_bridge", parameters=[config], output="screen"),
        Node(package="lilac_ros", executable="mock_sim", parameters=[config], output="screen"),
        Node(package="lilac_ros", executable="haptic", parameters=[config], output="screen"),
    ])
