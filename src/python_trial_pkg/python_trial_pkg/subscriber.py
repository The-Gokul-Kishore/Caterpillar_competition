import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import math
import numpy as np
import cv2
import os

class LidarTrinityDetector(Node):
    def __init__(self):
        super().__init__('lidar_trinity_detector')
        
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.goal_publisher = self.create_publisher(
            PoseStamped, '/goal_pose', 10)
            
        self.goal_sent = False
        
        # --- TARGET FINGERPRINT ---
        # The URDF places pins at dist ~ 0.173m (17 cm) from each other
        # (Based on equilateral geometry 0.1 from center)
        self.TARGET_SIDE_LEN = 0.173 
        self.TOLERANCE = 0.05  # Allow +/- 5cm error
        
        # Only consider small objects (pins)
        self.MAX_PIN_WIDTH = 0.10 
        
        # Visualization
        self.debug_dir = '/home/ws/debug_lidar/'
        if not os.path.exists(self.debug_dir): os.makedirs(self.debug_dir)
        self.IMG_SIZE = 600
        self.SCALE = 120 
        self.ROBOT_U = self.IMG_SIZE // 2
        self.ROBOT_V = self.IMG_SIZE - 50 

        self.get_logger().info("📡 Trinity Detector Started. Looking for 3-Pin Triangle...")

    def world_to_pixel(self, x, y):
        u = self.ROBOT_U - int(y * self.SCALE)
        v = self.ROBOT_V - int(x * self.SCALE)
        return (u, v)

    def scan_callback(self, msg):
        if self.goal_sent: return

        debug_img = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), np.uint8)
        cv2.circle(debug_img, (self.ROBOT_U, self.ROBOT_V), 5, (255, 0, 0), -1)

        # 1. PARSE & CLUSTER (Standard Logic)
        points = []
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        for i, r in enumerate(msg.ranges):
            if r < 0.2 or r > 4.0: continue
            angle = angle_min + (i * angle_inc)
            points.append([r * math.cos(angle), r * math.sin(angle)])
            uv = self.world_to_pixel(points[-1][0], points[-1][1])
            cv2.circle(debug_img, uv, 1, (80, 80, 80), -1)

        if not points: return

        clusters = []
        current_cluster = [points[0]]
        JUMP_THRESHOLD = 0.10 

        for i in range(1, len(points)):
            dist = math.sqrt((points[i][0]-points[i-1][0])**2 + (points[i][1]-points[i-1][1])**2)
            if dist < JUMP_THRESHOLD:
                current_cluster.append(points[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [points[i]]
        if current_cluster: clusters.append(current_cluster)

        # 2. FILTER CANDIDATES (Find Pins)
        pins = []
        for cluster in clusters:
            # Need at least a few points to be a real object
            if len(cluster) < 2: continue 
            
            p_start = cluster[0]
            p_end = cluster[-1]
            width = math.sqrt((p_start[0]-p_end[0])**2 + (p_start[1]-p_end[1])**2)
            
            # Pins are SMALL (< 10cm)
            if width < self.MAX_PIN_WIDTH:
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                pins.append((cx, cy))
                
                # Draw Candidate Pin (Cyan)
                uv = self.world_to_pixel(cx, cy)
                cv2.circle(debug_img, uv, 3, (255, 255, 0), -1)

        # 3. TRIANGULATION (The "Barcode" Check)
        # Look for any group of 3 pins that form the target triangle
        found = False
        num_pins = len(pins)
        
        if num_pins >= 3:
            # Check every combination of 3 pins
            for i in range(num_pins):
                for j in range(i + 1, num_pins):
                    for k in range(j + 1, num_pins):
                        p1, p2, p3 = pins[i], pins[j], pins[k]
                        
                        # Calculate all side lengths
                        d12 = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                        d23 = math.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
                        d31 = math.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)
                        
                        # Check if ALL sides match the target length (Equilateral)
                        # We allow some slop because LiDAR isn't perfect
                        match1 = abs(d12 - self.TARGET_SIDE_LEN) < self.TOLERANCE
                        match2 = abs(d23 - self.TARGET_SIDE_LEN) < self.TOLERANCE
                        match3 = abs(d31 - self.TARGET_SIDE_LEN) < self.TOLERANCE
                        
                        if match1 and match2 and match3:
                            self.get_logger().info(f"🌟 TRINITY FOUND! Sides: {d12:.2f}, {d23:.2f}, {d31:.2f}")
                            
                            # Center of the triangle
                            target_x = (p1[0] + p2[0] + p3[0]) / 3.0
                            target_y = (p1[1] + p2[1] + p3[1]) / 3.0
                            
                            # VISUALIZE (Green Triangle)
                            u1, v1 = self.world_to_pixel(p1[0], p1[1])
                            u2, v2 = self.world_to_pixel(p2[0], p2[1])
                            u3, v3 = self.world_to_pixel(p3[0], p3[1])
                            
                            cv2.line(debug_img, (u1,v1), (u2,v2), (0,255,0), 2)
                            cv2.line(debug_img, (u2,v2), (u3,v3), (0,255,0), 2)
                            cv2.line(debug_img, (u3,v3), (u1,v1), (0,255,0), 2)
                            
                            self.send_nav2_goal(target_x, target_y)
                            self.goal_sent = True
                            found = True
                            break
                    if found: break
                if found: break

        cv2.imwrite(os.path.join(self.debug_dir, 'trinity_debug.png'), debug_img)

    def send_nav2_goal(self, x, y):
        goal_msg = PoseStamped()
        from rclpy.time import Time
        goal_msg.header.stamp = Time().to_msg()
        goal_msg.header.frame_id = "base_link"
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.orientation.w = 1.0
        self.goal_publisher.publish(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarTrinityDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()