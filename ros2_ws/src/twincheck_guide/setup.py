from setuptools import setup

package_name = "twincheck_guide"

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
    description="TwinCheck Guide mode: vision-guided pick and sort with retry (disarmed by default)",
    license="Apache-2.0",
    entry_points={"console_scripts": ["guide_node = twincheck_guide.guide_node:main"]},
)
