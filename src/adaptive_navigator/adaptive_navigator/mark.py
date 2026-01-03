import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
import math
import cv2
import numpy as np

# --- SETTINGS FOR CONSISTENCY ---
# 1. CLUSTERING
# If points are within 8cm, they are the same object.
CLUSTER_GROUP_DIST = 0.08  
# Even a single dot can be a pin (helps if lidar misses slightly)
MIN_POINTS_PER_PIN = 1     
# Ignore massive walls
MAX_PIN_WIDTH = 0.25       

# 2. TRIANGLE "LOOSE" RULES
# Accept triangles between 4cm and 35cm
MIN_SIDE_LEN = 0.04        
MAX_SIDE_LEN = 0.35        
# The "Wobble" Factor: How different can the sides be?
# 8cm difference allowed (e.g., sides 20, 20, 28 is now ACCEPTED)
MAX_SIDE_DIFF = 0.08       

# 3. VISUALIZATION
IMG_SIZE = 800             
METERS_PER_PIXEL = 0.01    # 1cm per pixel

class GeometryUtils:
    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def get_centroid(points):
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        return (cx, cy)

class ConsistentScanner(Node):
    def __init__(self):
        super().__init__('consistent_scanner')
        self.image_count = 0
        
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_policy)
        
        self.get_logger().info("🚀 CONSISTENT MODE: Looser tolerances for steady detection...")

    def scan_callback(self, msg):
        raw_points = self.lidar_to_points(msg)
        if not raw_points: return

        # 1. Cluster
        pins, valid_clusters = self.cluster_points(raw_points)

        # 2. Detect
        trinity_center, trinity_pins, avg_size, candidates_checked = self.find_loose_triangle(pins)

        # 3. ALWAYS SAVE DEBUG IMAGE (The "Video Feed")
        self.save_image("lidar_debug.png", raw_points, valid_clusters, trinity_pins, trinity_center, avg_size)

        # 4. Handle Detections
        if trinity_center:
            self.image_count += 1
            filename = f"detection_{self.image_count:03d}.png"
            self.get_logger().info(f"✅ DETECTED! Size: {avg_size*100:.1f}cm (Saved {filename})")
            self.save_image(filename, raw_points, valid_clusters, trinity_pins, trinity_center, avg_size)
        
        elif candidates_checked > 0:
            # If we checked pins but found nothing, say why (helps debugging)
            pass 
            # self.get_logger().info(f"👀 Saw {len(pins)} pins, but no triangle shape matched.")

    def find_loose_triangle(self, pins):
        """ Returns center, pins, size, and how many groups it checked """
        num = len(pins)
        candidates = 0
        if num < 3: return None, [], 0, 0

        for i in range(num):
            for j in range(i + 1, num):
                for k in range(j + 1, num):
                    candidates += 1
                    group = [pins[i], pins[j], pins[k]]
                    
                    d1 = GeometryUtils.dist(group[0], group[1])
                    d2 = GeometryUtils.dist(group[1], group[2])
                    d3 = GeometryUtils.dist(group[2], group[0])
                    sides = [d1, d2, d3]
                    
                    shortest = min(sides)
                    longest = max(sides)
                    avg = sum(sides) / 3.0

                    # Check 1: Size
                    if not (MIN_SIDE_LEN < avg < MAX_SIDE_LEN): 
                        continue 

                    # Check 2: Equilateral-ish
                    diff = longest - shortest
                    if diff < MAX_SIDE_DIFF:
                        return GeometryUtils.get_centroid(group), group, avg, candidates
                    else:
                        # OPTIONAL: Uncomment to see why it fails
                        # print(f"Rejected: Diff {diff:.2f} too high for avg {avg:.2f}")
                        pass

        return None, [], 0, candidates

    def save_image(self, filename, all_points, clusters, trinity_pins, center, size):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        origin_x, origin_y = IMG_SIZE // 2, IMG_SIZE // 2

        def world_to_img(pt):
            # X UP, Y LEFT
            px = int(origin_x - (pt[1] / METERS_PER_PIXEL))
            py = int(origin_y - (pt[0] / METERS_PER_PIXEL))
            return (px, py)

        # Robot Center (Red)
        cv2.circle(img, (origin_x, origin_y), 5, (0, 0, 255), -1)

        # Raw Points (Bright White)
        for p in all_points:
            try: cv2.circle(img, world_to_img(p), 1, (255, 255, 255), -1)
            except: pass

        # Clusters (Cyan)
        for c in clusters:
            for p in c:
                try: cv2.circle(img, world_to_img(p), 2, (255, 255, 0), -1)
                except: pass

        # Triangle (Thick Green)
        if trinity_pins:
            p1 = world_to_img(trinity_pins[0])
            p2 = world_to_img(trinity_pins[1])
            p3 = world_to_img(trinity_pins[2])
            
            cv2.line(img, p1, p2, (0, 255, 0), 3)
            cv2.line(img, p2, p3, (0, 255, 0), 3)
            cv2.line(img, p3, p1, (0, 255, 0), 3)
            
            label = world_to_img(center)
            cv2.putText(img, f"{size*100:.1f}cm", (label[0], label[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(filename, img)

    def lidar_to_points(self, msg):
        points = []
        angle = msg.angle_min
        for r in msg.ranges:
            if 0.05 < r < 5.0: 
                points.append([r * math.cos(angle), r * math.sin(angle)])
            angle += msg.angle_increment
        return points

    def cluster_points(self, points):
        clusters = []
        if not points: return [], []
        curr = [points[0]]
        for i in range(1, len(points)):
            dist = math.sqrt((points[i][0]-points[i-1][0])**2 + (points[i][1]-points[i-1][1])**2)
            if dist < CLUSTER_GROUP_DIST: 
                curr.append(points[i])
            else:
                clusters.append(curr)
                curr = [points[i]]
        if curr: clusters.append(curr)
        
        pins = []
        valid_clusters = []
        for c in clusters:
            if len(c) < MIN_POINTS_PER_PIN: continue
            
            width = math.sqrt((c[0][0]-c[-1][0])**2 + (c[0][1]-c[-1][1])**2)
            if width < MAX_PIN_WIDTH:
                cx = sum(p[0] for p in c)/len(c)
                cy = sum(p[1] for p in c)/len(c)
                pins.append((cx, cy))
                valid_clusters.append(c)
        return pins, valid_clusters

def main(args=None):
    rclpy.init(args=args)
    node = ConsistentScanner()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
