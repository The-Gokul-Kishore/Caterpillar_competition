import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import os
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

class PublisherNode(Node):
    def __init__(self):
        super().__init__('map_publisher_node')
        self.publisher_ = self.create_publisher(Image, '/map_image', 10)
        self.timer_period = 2.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.br = CvBridge()

        # FIX: Try to find the image in the current working directory first
        # If not found, you should provide the absolute path.
        self.image_path = '/home/deepak/program_ws/src/demo_pkg/demo_pkg/image_test.png'

        
        # Check if file exists before reading
        if not os.path.exists(self.image_path):
            # Attempt to find it in your home directory as a fallback
            self.image_path = os.path.expanduser('~/program_ws/ggim.png')

        self.image = cv2.imread(self.image_path)

        if self.image is not None:
            self.get_logger().info(f'Successfully loaded: {self.image_path}')
        else:
            self.get_logger().error(f'Could not find or open: {self.image_path}')

    def timer_callback(self):
        if self.image is not None:
            # Convert OpenCV image to ROS Image message
            msg = self.br.cv2_to_imgmsg(self.image, encoding="bgr8")
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing map image...')
        else:
            self.get_logger().error('Cannot publish: Image not loaded!')

def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Publisher stopped by user (Ctrl+C).')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()