# MCL Localization Lab

This package implements a simple Monte Carlo Localization experiment using:

- A known 2D grid map generated with OpenCV.
- A matching Gazebo/GZ Sim `.sdf` world.
- A Python ROS2 node that subscribes to `/odom` and `/scan`.
- Particle sampling, scoring, filtering/resampling, and dead-reckoning propagation.

## Main files

- `mcl_localization_lab/build_warehouse_map.py`: creates `maps/warehouse_layout.png`.
- `mcl_localization_lab/particle_localizer.py`: ROS2 MCL node.
- `worlds/warehouse_mcl.sdf`: simulation world matching the map.
- `launch/mcl_lab.launch.py`: starts GZ Sim and the MCL node.

## Build

```bash
cd ~/mcl_ws
colcon build --symlink-install
source install/setup.bash
```

## Generate the map

```bash
ros2 run mcl_localization_lab build_warehouse_map
```

## Run the simulation and localizer

Terminal 1:

```bash
ros2 launch mcl_localization_lab mcl_lab.launch.py
```

Terminal 2, move the robot:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.4}, angular: {z: 0.2}}" -r 10
```

If you want to use teleop:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel
```
