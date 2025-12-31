import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import cv2
import numpy as np

class CostmapToImageConverter(Node):
    def __init__(self):
        super().__init__('costmap_to_image_node')
        
        # 1. QoS for Costmap
        costmap_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # 2. Subscribe to the Costmap
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',
            self.map_callback,
            costmap_qos)
            
        self.publisher_ = self.create_publisher(Image, '/map_image', 10)
        self.br = CvBridge()
        self.get_logger().info('Costmap -> Image Converter Started.')

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # --- THE FIX: IGNORE SAFETY CUSHIONS ---
        # Create a White Background (255)
        img = np.full((height, width, 1), 255, dtype=np.uint8)
        
        # ONLY paint Lethal Obstacles (100) as Black (0)
        # We ignore values 1-99 (Inflation) by leaving them White
        img[data == 100] = 0 
        
        # OPTIONAL: Make lines thicker so they don't break
        # This helps if the star legs are very thin
        kernel = np.ones((3,3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1) # Erode black = Thicker lines

        # Flip to match OpenCV coordinates
        img_flipped = cv2.flip(img, 0)
        img_color = cv2.cvtColor(img_flipped, cv2.COLOR_GRAY2BGR)
        
        out_msg = self.br.cv2_to_imgmsg(img_color, encoding="bgr8")
        self.publisher_.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CostmapToImageConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()