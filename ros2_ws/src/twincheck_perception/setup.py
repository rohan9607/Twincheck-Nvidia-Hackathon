from setuptools import setup

package_name = "twincheck_perception"

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
    description="TwinCheck overhead detector and wrist grasp check",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "detector_node = twincheck_perception.detector_node:main",
        "grasp_check_node = twincheck_perception.grasp_check_node:main",
    ]},
)
