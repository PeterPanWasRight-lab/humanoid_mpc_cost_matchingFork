from setuptools import find_packages, setup

package_name = "humanoid_cost_matching"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("lib/" + package_name, ["scripts/log_observation"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="maintainer",
    maintainer_email="wenqicai.97@gmail.com",
    description="Cost-matching utilities for wb_humanoid_mpc (logging, training, export).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "log_observation = humanoid_cost_matching.data.observation_logger:main",
        ],
    },
)
