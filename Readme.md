

# Adaptive Navigation Workspace

This workspace contains the source code, simulation models, and mission control logic for an autonomous rover navigation system. The project implements a state-machine-driven executive that performs systematic area coverage, geometric target identification (Lidar clustering), and dynamic path planning using ROS 2 Jazzy and Nav2.

## Workspace Structure

* **`src/adaptive_navigator`**: Main package containing the mission control node, detectors, and visualizers.
* **`gazebo_models/`**: Custom URDF models for simulation targets (Trinity, Quad, Spiky Star).
* **`debug_lidar/`**: Runtime output directory for OpenCV visualizations of Lidar clustering.
* **`test_scripts/`**: Standalone scripts for unit testing logic.

## Prerequisites

* **ROS 2 Jazzy**
* **Gazebo Harmonic** (or compatible `ros_gz` bridge)
* **Nav2 Stack**
* **TurtleBot3 Packages**

## Build Instructions

1. Navigate to the workspace root:
```bash
cd /home/ws

```


2. Build the packages:
```bash
colcon build --packages-select adaptive_navigator

```


3. Source the overlay:
```bash
source install/setup.bash

```



---

## Usage Guide

Follow these steps in separate terminal windows to execute the full mission simulation.

### 1. Environment Setup

If running inside a Docker container, allow local X11 connections for GUI display:

```bash
xhost +local:root

```

### 2. Launch Simulation (Gazebo + RViz)

Initialize the TurtleBot3 simulation with the Nav2 stack.

```bash
source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch nav2_bringup tb3_simulation_launch.py headless:=False

```

### 3. Spawn Custom Targets

Populate the arena with the required markers using the provided URDF models.

**Spawn Trinity Marker (Target A):**

```bash
ros2 run ros_gz_sim create -file gazebo_models/trinity_marker.urdf -name trinity -x 1.0 -y -0.5 -z 0.0

```

**Spawn Quad Marker (Target B):**

```bash
ros2 run ros_gz_sim create -file gazebo_models/quad_marker.urdf -name quad -x 1.0 -y 0.5 -z 0.0

```

### 4. Start Mission Control

Launch the autonomous executive node. This initiates Phase 1 (Search & Identify).

```bash
source install/setup.bash
# Run the mission control node (ensure executable name matches your setup)
ros2 run adaptive_navigator mission_control_node

```

*Note: The robot will automatically map the area boundaries and begin the search pattern.*

### 5. Trigger Phase 2 (Retrieval)

Once the robot identifies the Trinity marker and enters the `WAITING` state, send the service trigger to proceed to the Quad marker.

```bash
ros2 service call /start_quad_phase std_srvs/srv/Trigger {}

```

---

## Debugging & Visualization

* **Lidar Visualization:** Check the `debug_lidar/` directory for real-time OpenCV images showing Lidar clusters and shape detection logic.
* **RViz Markers:** Add the `/detection_debug` topic (MarkerArray) in RViz to see 3D visualization of candidate clusters and confirmed targets.