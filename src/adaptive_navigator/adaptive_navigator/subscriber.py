import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PointStamped # Changed
from std_srvs.srv import Trigger, SetBool
from std_msgs.msg import Bool
import math

import numpy as np

# IMPORT TF2 (Crucial for coordinate conversion)
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs 

# IMPORT NAV2
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# IMPORT DETECTORS
from .marker_detector import TrinityDetector, QuadDetector
from .debug_visualizer import DebugVisualizer, LidarVisualizer

# STATES
STATE_INITIALIZING = -1
STATE_SEARCHING = 0
STATE_MOVING_TO_TRINITY = 1
STATE_WAITING_AT_TRINITY = 2
STATE_MOVING_TO_QUAD = 3
STATE_DONE = 4

class MissionControlNode(Node):
    def __init__(self):
        super().__init__('mission_control_node')
        
        self.navigator = BasicNavigator()
        self.dig_pub = self.create_publisher(Bool, '/dig', 10)
        
        # --- TF BUFFER (The Fix) ---
        # We need this to calculate "Where is this detected point on the MAP?"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- MEMORY ---
        self.trinity_loc = None
        self.quad_loc = None
        self.state = STATE_INITIALIZING 
        
        self.search_waypoints = []
        self.wp_index = 0
        
        # --- MAP LISTENER ---
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE)
        
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)
            
        # --- COMMS ---
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
            
        # Navigation services
        self.srv_goto_quad = self.create_service(
            Trigger, '/goto_quad', self.handle_goto_quad)
        self.srv_goto_trinity = self.create_service(
            Trigger, '/goto_trinity', self.handle_goto_trinity)
            
        self.trinity_detector = TrinityDetector()
        self.quad_detector = QuadDetector()
        self.debugger = DebugVisualizer(self)
        self.lidar_viz = LidarVisualizer(".")

        # --- DIGGING / CYCLE SETTINGS ---
        # Counts how many offset points we've used around Trinity
        self.dig_cycle = 0
        # 60 cm radius around Trinity marker (meters)
        self.DIG_RADIUS =  0.60
        # How many distinct points per full circle (e.g. 8 -> every 45°)
        self.POINTS_PER_CIRCLE = 8
        # --- NEW QUAD LINEAR SETTINGS ---
        self.quad_cycle = 0            # Keeps track of which point in the line we are on
        self.QUAD_START_OFFSET = 1.30  # Start 1.3 meters away from the marker
        self.QUAD_STEP_SIZE = 0.15     # Move 15cm further away each time
        self.get_logger().info("⏳ WAITING FOR MAP TO AUTO-CALCULATE BOUNDARIES...")
        self.goal_start_time = self.navigator.get_clock().now()

    def transform_to_map(self, local_point,timestamp):
        """
        Converts a point (x,y) from 'base_link' (robot frame) to 'map' frame.
        """
        try:
            # 1. Create a PointStamped in the ROBOT frame (base_link)
            p_local = PointStamped()
            p_local.header.frame_id = "base_link" 
            p_local.header.stamp = timestamp
            p_local.header.stamp = rclpy.time.Time().to_msg() # Use 'now'
            p_local.point.x = float(local_point[0])
            p_local.point.y = float(local_point[1])
            p_local.point.z = 0.0

            # 2. Wait for transform (timeout 1.0s)
            target_frame = "map"
            if not self.tf_buffer.can_transform(target_frame, "base_link", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
                self.get_logger().warn("⚠️ TF lookup failed")
                return None

            # 3. Transform!
            p_map = self.tf_buffer.transform(p_local, target_frame)
            return (p_map.point.x, p_map.point.y)

        except Exception as e:
            self.get_logger().error(f"❌ Transformation Error: {e}")
            return None

    def map_callback(self, msg):
        """ Runs ONCE when map is received """
        if self.state != STATE_INITIALIZING: return

        self.get_logger().info(f"🗺️ Map Received! Size: {msg.info.width}x{msg.info.height}")
        
        data = np.array(msg.data)
        width = msg.info.width
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        free_indices = np.where(data == 0)[0]
        if len(free_indices) == 0: return

        y_grid = free_indices // width
        x_grid = free_indices % width
        
        real_min_x = origin_x + (np.min(x_grid) * resolution)
        real_max_x = origin_x + (np.max(x_grid) * resolution)
        real_min_y = origin_y + (np.min(y_grid) * resolution)
        real_max_y = origin_y + (np.max(y_grid) * resolution)
        
        SAFETY_BUFFER = 0.4 
        self.bounds_x = (real_min_x + SAFETY_BUFFER, real_max_x - SAFETY_BUFFER)
        self.bounds_y = (real_min_y + SAFETY_BUFFER, real_max_y - SAFETY_BUFFER)
        
        self.search_waypoints = self.generate_lawnmower_path()
        self.state = STATE_SEARCHING 
        self.get_logger().info(f"🤖 SEARCH STARTED with {len(self.search_waypoints)} waypoints.")

    def generate_lawnmower_path(self):
        waypoints = []
        step_size = 1.5 
        y_coords = np.arange(self.bounds_y[0], self.bounds_y[1] + 0.1, step_size)
        for i, y in enumerate(y_coords):
            if i % 2 == 0:
                waypoints.append((self.bounds_x[0], y))
                waypoints.append((self.bounds_x[1], y))
            else:
                waypoints.append((self.bounds_x[1], y))
                waypoints.append((self.bounds_x[0], y))
        return waypoints

    def handle_goto_quad(self, request, response):
        """Service handler: Now uses the linear line logic instead of center"""
        goal = self.get_linear_quad_goal()
        return self._goto_marker(goal, "QUAD", STATE_MOVING_TO_QUAD, response)
    
    def handle_goto_trinity(self, request, response):
        """Service handler: Navigate to Trinity marker"""
        # Prefer an offset point on the circumference so each call visits
        # a new digging spot around the Trinity marker.
        goal = self.get_offset_trinity_goal()
        return self._goto_marker(goal, "TRINITY", STATE_MOVING_TO_TRINITY, response)

    def get_offset_trinity_goal(self):
        """Return an (x,y) map coordinate on a circle around Trinity center.

        The point advances each call according to self.dig_cycle. Returns
        None if Trinity center is unknown.
        """
        if self.trinity_loc is None:
            return None

        # angle steps evenly around the circle
        angle = (self.dig_cycle % self.POINTS_PER_CIRCLE) * (2 * math.pi / self.POINTS_PER_CIRCLE)
        offset_x = self.trinity_loc[0] + (self.DIG_RADIUS * math.cos(angle))
        offset_y = self.trinity_loc[1] + (self.DIG_RADIUS * math.sin(angle))

        self.get_logger().info(f"🔄 Dig cycle {self.dig_cycle}: angle={math.degrees(angle):.1f}° target=({offset_x:.3f},{offset_y:.3f})")

        # prepare next cycle
        self.dig_cycle += 1
        return (offset_x, offset_y)
    def get_linear_quad_goal(self):
        """Calculates a point in a straight line from the Quad marker."""
        if self.quad_loc is None:
            return None

        # Distance = 1.3m + (0.15 * current_step)
        dist = self.QUAD_START_OFFSET + (self.quad_cycle * self.QUAD_STEP_SIZE)
        
        # Calculate the new point (X moves horizontally)
        target_x = self.quad_loc[0] + dist
        target_y = self.quad_loc[1]

        self.get_logger().info(f"📏 Quad Step {self.quad_cycle}: Distance is {dist:.2f}m")
        
        # Increment for the next time the service is called
        self.quad_cycle += 1
        return (target_x, target_y)
    
    def _goto_marker(self, location, name, target_state, response):
        """Reusable navigation logic for any marker"""
        if location is None:
            self.get_logger().info(f"🔎 {name} not found yet. Resuming search...")
            self.state = STATE_SEARCHING
            response.success = True
            response.message = f"{name} unknown. Searching..."
        else:
            self.get_logger().info(f"🚀 Navigating to {name} at {location}")
            self.go_to_coord(location)
            self.state = target_state
            response.success = True
            response.message = f"Going to {name}"
        return response

    def scan_callback(self, msg):
        if self.state == STATE_INITIALIZING: return

        # 1. DETECT (Relative Coords)
        points = self.lidar_to_points(msg)
        if not points: return
        pins = self.cluster_points(points)
        
        quad_found_rel, quad_pins = self.quad_detector.detect(pins)
        trinity_found_rel, trinity_pins = None, None
        if not quad_found_rel:
            trinity_found_rel, trinity_pins = self.trinity_detector.detect(pins)
        
        # 2. FOUND TRINITY?
        if trinity_found_rel and self.trinity_loc is None:
            # FIX: Convert Relative -> Map Frame IMMEDIATELY
            map_coords = self.transform_to_map(trinity_found_rel, msg.header.stamp)

            if map_coords:
                self.trinity_loc = map_coords
                self.get_logger().info(f"📍 TRINITY SAVED at MAP {self.trinity_loc}")

                if self.state == STATE_SEARCHING:
                    # Choose an offset goal on the 30cm circumference
                    goal = self.get_offset_trinity_goal()
                    if goal:
                        self.get_logger().info("🛑 FOUND TARGET A! Switching Goal to dig offset...")
                        self.state = STATE_MOVING_TO_TRINITY
                        self.go_to_coord(goal)
                    else:
                        # Fallback to center if something goes wrong
                        self.get_logger().warn("⚠️ Failed to compute offset goal; going to center")
                        self.state = STATE_MOVING_TO_TRINITY
                        self.go_to_coord(self.trinity_loc)

        # 3. FOUND QUAD? (Memory)
        if quad_found_rel and (self.quad_loc is None):
            # FIX: Convert Relative -> Map Frame IMMEDIATELY
            # If we don't do this now, the coordinates become useless once the robot moves.
            map_coords = self.transform_to_map(quad_found_rel, msg.header.stamp)
            
            if map_coords:
                self.quad_loc = map_coords
                self.get_logger().info(f"📍 QUAD SAVED at MAP {self.quad_loc}")
                self.debugger.publish_debug(pins, quad_pins, "QUAD")
                if self.state == STATE_SEARCHING and self.trinity_loc is not None:
                     self.get_logger().info("🛑 FOUND TARGET B (QUAD)! Switching Goal...")
                     self.state = STATE_MOVING_TO_QUAD
                     self.goal_start_time = self.navigator.get_clock().now()
                     self.go_to_coord(self.quad_loc)
        # State machine: Handle navigation
        if self.state == STATE_SEARCHING:
            self.perform_systematic_search()
        elif self.state in [STATE_MOVING_TO_TRINITY, STATE_MOVING_TO_QUAD]:
            if self.navigator.isTaskComplete():
                marker_name = "TRINITY" if self.state == STATE_MOVING_TO_TRINITY else "QUAD"
                self.get_logger().info(f"🏁 Arrived at {marker_name}. Ready for next command.")
                msg = Bool()
                msg.data = True
                self.dig_pub.publish(msg)
                self.state = STATE_WAITING_AT_TRINITY  # Reuse waiting state for both

    def perform_systematic_search(self):
        if not self.navigator.isTaskComplete(): return
        if self.wp_index >= len(self.search_waypoints): self.wp_index = 0
        target = self.search_waypoints[self.wp_index]
        self.go_to_coord(target)
        self.wp_index += 1

    def go_to_coord(self, coords):
        pose = PoseStamped()
        pose.header.frame_id = 'map' # Now valid because coords are transformed!
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = float(coords[0])
        pose.pose.position.y = float(coords[1])
        pose.pose.orientation.w = 1.0
        self.navigator.goToPose(pose)
        self.goal_start_time = self.navigator.get_clock().now()

    # --- HELPERS ---
    def lidar_to_points(self, msg):
        points = []
        angle = msg.angle_min
        for r in msg.ranges:
            if 0.2 < r < 6.0: 
                points.append([r * math.cos(angle), r * math.sin(angle)])
            angle += msg.angle_increment
        return points

    def cluster_points(self, points):
        clusters = []
        if not points: return []
        curr = [points[0]]
        for i in range(1, len(points)):
            dist = math.sqrt((points[i][0]-points[i-1][0])**2 + (points[i][1]-points[i-1][1])**2)
            if dist < 0.10: curr.append(points[i])
            else:
                clusters.append(curr)
                curr = [points[i]]
        if curr: clusters.append(curr)
        pins = []
        for c in clusters:
            if len(c) < 2: continue
            width = math.sqrt((c[0][0]-c[-1][0])**2 + (c[0][1]-c[-1][1])**2)
            if width < 0.10:
                cx = sum(p[0] for p in c)/len(c)
                cy = sum(p[1] for p in c)/len(c)
                pins.append((cx, cy))
        return pins

def main(args=None):
    rclpy.init(args=args)
    node = MissionControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()