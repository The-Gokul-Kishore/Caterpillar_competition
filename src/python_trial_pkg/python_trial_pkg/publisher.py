import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import cv2
import numpy as np

class MapToImageConverter(Node):
    def __init__(self):
        super().__init__('map_to_image_node')
        
        # 1. QoS for receiving the Map (Latched)
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # 2. Subscribe to the Map
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos)
            
        # 3. Publisher for the Image
        self.publisher_ = self.create_publisher(Image, '/map_image', 10)
        
        # 4. Timer to re-publish the image (Heartbeat)
        self.timer = self.create_timer(2.0, self.timer_callback)
        
        self.br = CvBridge()
        self.latest_image_msg = None  # Storage for the converted image
        
        self.get_logger().info('Map -> Image Converter Started. Waiting for map...')

    def map_callback(self, msg):
        self.get_logger().info('Received map! Converting...')
        
        # --- Map Processing Logic ---
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Create Image
        img = np.full((height, width, 1), 127, dtype=np.uint8)
        img[data == 0] = 255
        img[data == 100] = 0

        # Flip and Convert
        img_flipped = cv2.flip(img, 0)
        img_color = cv2.cvtColor(img_flipped, cv2.COLOR_GRAY2BGR)
        
        # Save to 'self' so the timer can access it
        self.latest_image_msg = self.br.cv2_to_imgmsg(img_color, encoding="bgr8")
        
        # Publish immediately once
        self.publisher_.publish(self.latest_image_msg)

    def timer_callback(self):
        # If we have a map image, publish it again
        if self.latest_image_msg is not None:
            self.publisher_.publish(self.latest_image_msg)
            # self.get_logger().info('Republishing map image...') # Uncomment for debug

def main(args=None):
    rclpy.init(args=args)
    node = MapToImageConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()