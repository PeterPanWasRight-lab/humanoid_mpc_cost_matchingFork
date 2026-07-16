import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_share = get_package_share_directory("g1_centroidal_mpc")
    base_launch = os.path.join(pkg_share, "launch", "dummy_sim.launch.py")

    src_root = os.environ.get(
        "WB_HUMANOID_MPC_SRC", "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc"
    )

    src_cfg = os.path.join(
        src_root, "robot_models", "unitree_g1", "g1_centroidal_mpc", "config"
    )
    gait_cfg = os.path.join(src_root, "humanoid_nmpc", "humanoid_common_mpc", "config")

    launch_args = {
        "config_name": os.path.join(src_cfg, "mpc", "task.info"),
        "target_command_file": os.path.join(src_cfg, "command", "reference.info"),
        "target_gait_file": os.path.join(gait_cfg, "command", "gait.info"),
    }

    return LaunchDescription(
        [
            LogInfo(msg=["[CM LAUNCH] config_name = ", launch_args["config_name"]]),
            LogInfo(
                msg=[
                    "[CM LAUNCH] target_command_file = ",
                    launch_args["target_command_file"],
                ]
            ),
            LogInfo(
                msg=["[CM LAUNCH] target_gait_file = ", launch_args["target_gait_file"]]
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments=launch_args.items(),
            ),
        ]
    )
