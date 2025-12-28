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
import os
from scipy.signal import find_peaks

class Nav2AutoGoalNode(Node):
    def __init__(self):
        super().__init__('destination_to_nav2_node')

        self.cb_group = ReentrantCallbackGroup()

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
        self.get_logger().info(f"✅ MAP RECEIVED! Res={self.map_res}")

    def process_image(self, data):
        if self.goal_sent: return
        if self.map_res is None: return

        try:
            img = self.br.imgmsg_to_cv2(data, "bgr8")
            debug_path = '/home/ws/debug_pngs/debug_input_map.png'
            cv2.imwrite(debug_path, img)

        except Exception as e:
            self.get_logger().error(f"CV Error: {e}")
            return
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) > 127:
            gray = cv2.bitwise_not(gray)
            
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- 📸 DEBUG SAVER: VISUALIZE WHAT THE ROBOT SEES 📸 ---
        # Draw all found contours in GREEN to verify detection
        debug_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)
        
        # Save the image to your workspace folder
        debug_path = '/home/ws/debug_pngs/debug_map.png'
        cv2.imwrite(debug_path, debug_img)
        self.get_logger().info(f"📸 Saved debug view to {debug_path}", throttle_duration_sec=2.0)
        # ---------------------------------------------------------

        found_target = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: continue # Lowered area check

            hull = cv2.convexHull(cnt)
            solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            
            distances = [math.sqrt((p[0][0]-cx)**2 + (p[0][1]-cy)**2) for p in cnt]
            peaks, _ = find_peaks(distances, prominence=2, width=1) # Relaxed prominence

            # --- CRITICAL FIX FOR 12-POINT STAR ---
            # Your star has 12 points, but your code said "peaks <= 10".
            # I increased the limit to 20.
            if solidity < 0.9 and 4 <= len(peaks) <= 20:
                self.get_logger().info(f"🎯 TARGET FOUND at ({cx}, {cy})! Peaks={len(peaks)}, Solidity={solidity:.2f}")

                # Draw the WINNING target in RED
                cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.imwrite(debug_path, debug_img) # Save again with red dot

                world_x = self.map_origin_x + (cx * self.map_res)
                real_y_pixel = self.map_height - cy
                world_y = self.map_origin_y + (real_y_pixel * self.map_res)

                self.send_nav2_goal(world_x, world_y)
                self.goal_sent = True 
                found_target = True
                break 

        if not found_target:
            self.get_logger().info("❌ Target NOT found.", throttle_duration_sec=2.0)

    def send_nav2_goal(self, x, y):
        goal_msg = PoseStamped()
        from rclpy.time import Time
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