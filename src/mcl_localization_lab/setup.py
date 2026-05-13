from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'mcl_localization_lab'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*.png')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cesar Mateo Sanchez Alvarez',
    maintainer_email='a01541805@tec.mx',
    description='Monte Carlo Localization lab with a known map, Gazebo world, and ROS2 particle filter.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'particle_localizer = mcl_localization_lab.particle_localizer:main',
            'build_warehouse_map = mcl_localization_lab.build_warehouse_map:main',
        ],
    },
)
