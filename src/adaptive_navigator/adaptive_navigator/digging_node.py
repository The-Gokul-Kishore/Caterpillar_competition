import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import lgpio
import time
import threading

# --- CONFIGURATION ---
# For Raspberry Pi 5, gpiochip4 is standard for the 40-pin header
GPIO_CHIP = 4 

# Define your motor pins (adjust these to match your actual wiring)
MOTOR1_A = 25
MOTOR1_B = 24
MOTOR2_A = 14
MOTOR2_B = 15

class DiggingSubscriber(Node):
    def __init__(self):
        super().__init__('digging_subscriber')
        
        # 1. Create the Subscriber
        # This listens to the "/dig" topic we created in your other code
        self.subscription = self.create_subscription(
            Bool,
            '/dig',
            self.listener_callback,
            10)
        
        self.is_running = False
        
        # 2. Setup GPIO for Pi 5
        try:
            self.h = lgpio.gpiochip_open(GPIO_CHIP)
            self.get_logger().info(f"✅ Motor Node Started. Connected to GPIO Chip {GPIO_CHIP}")
        except Exception as e:
            self.get_logger().error(f"❌ GPIO Error: {e}. If on Pi 5, try 'sudo' or check chip number.")
            return

        # Initialize pins
        for pin in [MOTOR1_A, MOTOR1_B, MOTOR2_A, MOTOR2_B]:
            lgpio.gpio_claim_output(self.h, pin)
            lgpio.gpio_write(self.h, pin, 0) # Start with motors OFF

    def listener_callback(self, msg):
        # This function runs automatically whenever the Brain sends 'True'
        if msg.data is True and not self.is_running:
            self.get_logger().info("🚜 Signal Received! Starting digging sequence...")
            # Run the motors in a separate thread so the node doesn't freeze
            thread = threading.Thread(target=self.motor_sequence)
            thread.start()

    def motor_sequence(self):
        self.is_running = True
        
        # --- Motor 1 Action (e.g., Lowering the tool) ---
        self.get_logger().info("🔄 Motor 1 spinning...")
        lgpio.gpio_write(self.h, MOTOR1_A, 0)
        lgpio.gpio_write(self.h, MOTOR1_B, 0)
        time.sleep(5.0) 
        lgpio.gpio_write(self.h, MOTOR1_A, 1)
        lgpio.gpio_write(self.h, MOTOR1_B, 0)
        time.sleep(5.0) 
        lgpio.gpio_write(self.h, MOTOR1_A, 0) # Stop
        
        time.sleep(1.0) # Short pause
        lgpio.gpio_write(self.h, MOTOR1_A, 0)
        lgpio.gpio_write(self.h, MOTOR1_B, 1)
        time.sleep(5.0) 
        lgpio.gpio_write(self.h, MOTOR1_A, 0) # Stop
        
        time.sleep(1.0) # Short pause

        # --- Motor 2 Action (e.g., Digging/Drilling) ---
        self.get_logger().info("🔄 Motor 2 spinning...")
        lgpio.gpio_write(self.h, MOTOR2_A, 0)
        lgpio.gpio_write(self.h, MOTOR2_B, 1)
        time.sleep(5.0)
        lgpio.gpio_write(self.h, MOTOR2_A, 0) # Stop

        self.get_logger().info("✅ Sequence complete. Waiting for next arrival.")
        self.is_running = False

    def __del__(self):
        # Cleanup when the program stops
        if hasattr(self, 'h'):
            lgpio.gpiochip_close(self.h)

def main(args=None):
    rclpy.init(args=args)
    node = DiggingSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()