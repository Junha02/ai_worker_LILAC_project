from glob import glob
from setuptools import find_packages, setup


package_name = "lilac_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="LILAC Project",
    maintainer_email="nninjiuuoo@gachon.ac.kr",
    description="ROS 2 adapters for LILAC language, policy, IK, simulation, and haptic nodes.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "language_manager = lilac_ros.language_manager_node:main",
            "latent_input = lilac_ros.latent_input_node:main",
            "policy = lilac_ros.policy_node:main",
            "ik_bridge = lilac_ros.ik_bridge_node:main",
            "mock_sim = lilac_ros.mock_sim_node:main",
            "haptic = lilac_ros.haptic_node:main",
            "utterance_client = lilac_ros.utterance_client:main",
        ],
    },
)
