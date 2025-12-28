import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point # To send coordinates
import cv2
import numpy as np
import math
from cv_bridge import CvBridge
from scipy.signal import find_peaks

class SubsciberNode(Node):
    def __init__(self):
        super().__init__('destination_detector_node')
        self.subscription = self.create_subscription(Image, '/map_image', self.process_image, 10)
        self.coord_publisher = self.create_publisher(Point, '/destination_coords', 10)
        self.br = CvBridge()

    def process_image(self, data):
        self.get_logger().info('Image received. Processing...')
        img = self.br.imgmsg_to_cv2(data)
        
        # 1. Convert to Grayscale and Threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) > 127:
            gray = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 2. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            img_area = gray.shape[0] * gray.shape[1]
            if area < 100 or area > (img_area * 0.9): continue

            # Solidity Check
            hull = cv2.convexHull(cnt)
            solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            
            # Peak Count Check (Your Core Logic)
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            
            distances = [math.sqrt((p[0][0]-cx)**2 + (p[0][1]-cy)**2) for p in cnt]
            peaks, _ = find_peaks(distances, prominence=3, width=2)

            # Identification
            if solidity < 0.85 and 4 <= len(peaks) <= 10:
                self.get_logger().info(f"✅ Destination Found at: X={cx}, Y={cy}")
                
                # Publish the coordinates
                point_msg = Point()
                point_msg.x = float(cx)
                point_msg.y = float(cy)
                point_msg.z = 0.0
                self.coord_publisher.publish(point_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SubsciberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()