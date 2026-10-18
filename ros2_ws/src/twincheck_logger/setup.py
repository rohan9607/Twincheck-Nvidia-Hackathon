from setuptools import setup

package_name = "twincheck_logger"

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
    description="TwinCheck trial logger (Postgres)",
    license="Apache-2.0",
    entry_points={"console_scripts": ["trial_logger = twincheck_logger.trial_logger:main"]},
)
