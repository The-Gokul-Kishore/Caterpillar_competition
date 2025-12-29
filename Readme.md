# How to run the application : 

### 1. In a global terminal:
 ```
    xhost +local:root
 ```
### 2. Intialize gazebo and Rviz:
 1. gazebo:
  ```
   source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```
2. RViz
 ```
  source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=true
 ```
 3. Pub:
 ```
 source install/setup.bash
ros2 run python_trial_pkg publisher
```
4. Sub:
```
 source install/setup.bash
ros2 run python_trial_pkg subscriber
```

## Spiky star:
```

ros2 run ros_gz_sim create -file gazebo_models/tiny_spiky_star.urdf -name spiky_star -x 1.0 -y 0.5 -z 0.0
```

## colcon build:
```
colcon build --packages-select python_trial_pkg
```

ros2 run ros_gz_sim create -file gazebo_models/tiny_spiky_star.urdf -name spiky_star -x 1.0 -y 0.5 -z 0.0  this one right