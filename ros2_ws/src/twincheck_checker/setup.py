from setuptools import setup

package_name = "twincheck_checker"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team Wicked",
    maintainer_email="team-wicked@example.com",
    description="TwinCheck robot events, expectations, comparator, resync and alerts",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "robot_state_node = twincheck_checker.robot_state_node:main",
        "twincheck_trigger = twincheck_checker.robot_state_node:trigger_main",
        "expectation_node = twincheck_checker.expectation_node:main",
        "comparator_node = twincheck_checker.comparator_node:main",
        "resync_node = twincheck_checker.resync_node:main",
        "alert_node = twincheck_checker.alert_node:main",
    ]},
)
