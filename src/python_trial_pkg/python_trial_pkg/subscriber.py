import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid 
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from scipy.signal import find_peaks

class Nav2AutoGoalNode(Node):
    def __init__(self):
        super().__init__('destination_to_nav2_node')

        # 1. ALLOW MULTIPLE CALLBACKS AT ONCE
        self.cb_group = ReentrantCallbackGroup()

        # 2. SUBSCRIBE TO THE MAP (For Metadata)
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos,
            callback_group=self.cb_group) 

        # 3. SUBSCRIBE TO THE IMAGE
        self.subscription = self.create_subscription(
            Image, 
            '/map_image', 
            self.process_image, 
            10,
            callback_group=self.cb_group)
        
        self.goal_publisher = self.create_publisher(
            PoseStamped, 
            '/goal_pose', 
            10)
            
        self.br = CvBridge()
        self.goal_sent = False

        # Map Metadata Storage
        self.map_res = None      
        self.map_origin_x = None 
        self.map_origin_y = None 
        self.map_height = None

        self.get_logger().info("Subscriber Started. Waiting for Map...")

    def map_callback(self, msg):
        self.map_res = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_height = msg.info.height
        
        self.get_logger().info(f"✅ MAP RECEIVED! Resolution={self.map_res}")

    def process_image(self, data):
        # Prevent sending multiple goals
        if self.goal_sent:
            return
        
        # Wait until we have map metadata
        if self.map_res is None:
            self.get_logger().warn("Got Image, waiting for Map Data...", throttle_duration_sec=3)
            return

        try:
            img = self.br.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Error: {e}")
            return
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # If map is mostly white (free space), invert to find black obstacles
        if np.mean(gray) > 127:
            gray = cv2.bitwise_not(gray)
            
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Track if we found anything in this frame
        found_target_in_this_frame = False

        # Debug: Print how many shapes we see
        if len(contours) > 0:
            self.get_logger().info(f"👀 Scanning {len(contours)} potential objects...", throttle_duration_sec=2.0)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100: continue

            hull = cv2.convexHull(cnt)
            solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            distances = [math.sqrt((p[0][0]-cx)**2 + (p[0][1]-cy)**2) for p in cnt]
            peaks, _ = find_peaks(distances, prominence=3, width=2)

            # --- CRITERIA CHECK ---
            # Checks for "Star-like" shape (low solidity, multiple peaks)
            if solidity < 0.85 and 4 <= len(peaks) <= 10:
                self.get_logger().info(f"🎯 TARGET FOUND at ({cx}, {cy})! Peaks={len(peaks)}")

                # Coordinate Math
                world_x = self.map_origin_x + (cx * self.map_res)
                real_y_pixel = self.map_height - cy
                world_y = self.map_origin_y + (real_y_pixel * self.map_res)

                self.send_nav2_goal(world_x, world_y)
                
                self.goal_sent = True 
                found_target_in_this_frame = True
                break # Stop searching, we found it

        # --- LOGGING "NOT FOUND" ---
        if not found_target_in_this_frame:
            self.get_logger().info("❌ Target NOT found in current map.", throttle_duration_sec=2.0)

    def send_nav2_goal(self, x, y):
        goal_msg = PoseStamped()
        from rclpy.time import Time
        # Use Time() (zero) to indicate "current valid time" for transforms
        goal_msg.header.stamp = Time().to_msg()
        goal_msg.header.frame_id = "map" 
        
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        
        self.goal_publisher.publish(goal_msg)
        self.get_logger().info(f"🚀 SENDING GOAL: X={x:.2f}, Y={y:.2f}")

def main(args=None):
    rclpy.init(args=args)
    
    # Use MultiThreadedExecutor to handle Map and Image callbacks in parallel
    from rclpy.executors import MultiThreadedExecutor
    
    node = Nav2AutoGoalNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()