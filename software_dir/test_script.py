import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import math

class ShapeHunter(Node):
    def __init__(self):
        super().__init__('shape_hunter')
        
        # --- CONFIGURATION ---
        # Run 'ros2 topic list' in terminal if this topic name is wrong!
        self.camera_topic = '/camera/image_raw' 
        
        self.subscription = self.create_subscription(
            Image, 
            self.camera_topic, 
            self.image_callback, 
            10)
        
        self.bridge = CvBridge()
        self.target_detected = False
        print(f"Shape Hunter Node Started! Listening on {self.camera_topic}...")

    def image_callback(self, msg):
        try:
            # Convert ROS Image -> OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Check for the shape
            is_target, _ = self.detect_square(cv_image)
            
            if is_target:
                self.target_detected = True
                # Optional: Show what the robot sees (might crash in pure container without GUI setup)
                # cv2.imshow("Robot Eyes", cv_image)
                # cv2.waitKey(1)
        except Exception as e:
            pass # Keep silent to avoid spamming console

    def detect_square(self, image):
        # 1. Grayscale & Blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Threshold (Looking for Dark objects on Light floor)
        # We invert the threshold (cv2.THRESH_BINARY_INV) so the BLACK box becomes WHITE blob
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Clean up shape edges
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            
            # --- THE LOGIC ---
            # If it has 4 corners (Square/Cube) AND is big enough (not dust)
            if len(approx) == 4 and cv2.contourArea(cnt) > 1000:
                print("!!! TARGET DETECTED (SQUARE/CUBE) !!!")
                return True, approx
                
        return False, None

def main():
    rclpy.init()
    
    # 1. Start the Vision System
    hunter_node = ShapeHunter()
    
    # 2. Start the Navigation System
    nav = BasicNavigator()
    
    # --- INITIAL POSITION (ZONE 1) ---
    # We assume the robot spawns at 0,0. This tells Nav2 "I am here."
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = nav.get_clock().now().to_msg()
    initial_pose.pose.position.x = 0.0
    initial_pose.pose.position.y = 0.0
    initial_pose.pose.orientation.w = 1.0
    nav.setInitialPose(initial_pose)
    
    print("Waiting for Nav2 to wake up...")
    nav.waitUntilNav2Active()

    # --- GOAL POSITION (ZONE 2) ---
    # We drive to the general area where we spawned the cube (x=1.5, y=0.0)
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = nav.get_clock().now().to_msg()
    goal_pose.pose.position.x = 1.5  
    goal_pose.pose.position.y = 0.0
    goal_pose.pose.orientation.w = 1.0
    
    print("Moving to Zone 2 Coordinates...")
    nav.goToPose(goal_pose)

    # --- MONITORING LOOP ---
    while not nav.isTaskComplete():
        # Process vision once per loop
        rclpy.spin_once(hunter_node, timeout_sec=0.1)
        
        if hunter_node.target_detected:
            print("Target sighted! Stopping robot.")
            nav.cancelTask()
            break

    # Final result
    result = nav.getResult()
    if result == TaskResult.SUCCEEDED:
        print("Reached Zone 2 Destination!")
    elif hunter_node.target_detected:
        print("Mission Success: Target Found!")
    else:
        print("Mission Failed or Canceled.")

    rclpy.shutdown()

if __name__ == '__main__':
    main() 