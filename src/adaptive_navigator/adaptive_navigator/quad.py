import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
import math
import cv2
import numpy as np

# --- SETTINGS ---
# 1. CLUSTERING (Same as Triangle success)
CLUSTER_GROUP_DIST = 0.08   # 8cm gap breaks a cluster
MIN_POINTS_PER_PIN = 1      # Even single dots count
MAX_PIN_WIDTH = 0.25        

# 2. QUAD DIMENSIONS (Target: 12cm Side, 17cm Diag)
# We accept a "Loose" range for real-world noise
MIN_SIDE = 0.08   # 8cm
MAX_SIDE = 0.16   # 16cm

# 3. GEOMETRY RATIOS
# In a square, Diag = Side * 1.414
# We allow the ratio to be between 1.2 and 1.6
MIN_RATIO = 1.2
MAX_RATIO = 1.6

# Visualization
IMG_SIZE = 800             
METERS_PER_PIXEL = 0.01    

class GeometryUtils:
    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def get_centroid(points):
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        return (cx, cy)

class QuadScanner(Node):
    def __init__(self):
        super().__init__('quad_scanner')
        self.image_count = 0
        
        # Robust Connection
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_policy)
        
        self.get_logger().info("🔲 QUAD SCANNER: Looking for Squares (4-pin) & Corners (3-pin)...")

    def scan_callback(self, msg):
        raw_points = self.lidar_to_points(msg)
        if not raw_points: return

        # 1. Cluster
        pins, valid_clusters = self.cluster_points(raw_points)

        # 2. Detect (Try 4-pin first, then 3-pin)
        center, used_pins, type_str, size = self.detect_quad_or_corner(pins)

        # 3. Save Debug Image
        self.save_image("lidar_quad_debug.png", raw_points, valid_clusters, used_pins, center, type_str)

        # 4. Log Success
        if center:
            self.image_count += 1
            filename = f"quad_{self.image_count:03d}.png"
            self.get_logger().info(f"✅ FOUND {type_str}! Side: {size*100:.1f}cm -> Saved {filename}")
            self.save_image(filename, raw_points, valid_clusters, used_pins, center, type_str)

    def detect_quad_or_corner(self, pins):
        """ Returns: Center, List of Pins, Type ("QUAD" or "CORNER"), AvgSideLen """
        num = len(pins)
        if num < 3: return None, [], "", 0

        # --- PRIORITY 1: FULL 4-PIN SQUARE ---
        if num >= 4:
            for i in range(num):
                for j in range(i+1, num):
                    for k in range(j+1, num):
                        for l in range(k+1, num):
                            group = [pins[i], pins[j], pins[k], pins[l]]
                            
                            # Measure all 6 distances in the group of 4
                            dists = []
                            for p1_idx in range(4):
                                for p2_idx in range(p1_idx+1, 4):
                                    dists.append(GeometryUtils.dist(group[p1_idx], group[p2_idx]))
                            dists.sort() # Smallest to Largest

                            # A Square has 4 short sides (sides) and 2 long sides (diagonals)
                            avg_side = sum(dists[:4]) / 4.0
                            avg_diag = sum(dists[4:]) / 2.0

                            # CHECK 1: Is side length valid? (8-16cm)
                            if not (MIN_SIDE < avg_side < MAX_SIDE): continue

                            # CHECK 2: Is it square-shaped? (Diagonal/Side ratio)
                            ratio = avg_diag / avg_side
                            if MIN_RATIO < ratio < MAX_RATIO:
                                return GeometryUtils.get_centroid(group), group, "FULL QUAD", avg_side

        # --- PRIORITY 2: BROKEN 3-PIN CORNER ---
        # If we didn't find a full square, look for an "L" shape
        for i in range(num):
            for j in range(i+1, num):
                for k in range(j+1, num):
                    group = [pins[i], pins[j], pins[k]]
                    
                    # Measure 3 sides
                    dists = []
                    dists.append((GeometryUtils.dist(group[0], group[1]), group[0], group[1]))
                    dists.append((GeometryUtils.dist(group[1], group[2]), group[1], group[2]))
                    dists.append((GeometryUtils.dist(group[2], group[0]), group[2], group[0]))
                    
                    # Sort by distance: [Small, Small, Large]
                    dists.sort(key=lambda x: x[0])
                    
                    short1 = dists[0][0]
                    short2 = dists[1][0]
                    long_side = dists[2][0]
                    
                    avg_short = (short1 + short2) / 2.0

                    # CHECK 1: Size valid?
                    if not (MIN_SIDE < avg_short < MAX_SIDE): continue

                    # CHECK 2: Isosceles? (Two short sides roughly equal)
                    if abs(short1 - short2) > 0.04: continue # Max 4cm diff

                    # CHECK 3: Right Angle? (Hypotenuse check)
                    ratio = long_side / avg_short
                    if MIN_RATIO < ratio < MAX_RATIO:
                        # Center of a square is the midpoint of the diagonal (the long side)
                        # The long side connects the two outer pins.
                        p_start = dists[2][1]
                        p_end = dists[2][2]
                        cx = (p_start[0] + p_end[0]) / 2.0
                        cy = (p_start[1] + p_end[1]) / 2.0
                        return (cx, cy), group, "CORNER", avg_short

        return None, [], "", 0

    def save_image(self, filename, all_points, clusters, target_pins, center, type_str):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        origin_x, origin_y = IMG_SIZE // 2, IMG_SIZE // 2

        def world_to_img(pt):
            px = int(origin_x - (pt[1] / METERS_PER_PIXEL))
            py = int(origin_y - (pt[0] / METERS_PER_PIXEL))
            return (px, py)

        # Robot (Red)
        cv2.circle(img, (origin_x, origin_y), 5, (0, 0, 255), -1)

        # Raw (White)
        for p in all_points:
            try: cv2.circle(img, world_to_img(p), 1, (255, 255, 255), -1)
            except: pass

        # Clusters (Cyan)
        for c in clusters:
            for p in c:
                try: cv2.circle(img, world_to_img(p), 2, (255, 200, 0), -1)
                except: pass

        # Target (Green for Quad, Yellow for Corner)
        if target_pins:
            color = (0, 255, 0) if type_str == "FULL QUAD" else (0, 255, 255)
            
            # Draw lines between all pins in the group
            for i in range(len(target_pins)):
                for j in range(i+1, len(target_pins)):
                    p1 = world_to_img(target_pins[i])
                    p2 = world_to_img(target_pins[j])
                    cv2.line(img, p1, p2, color, 2)
            
            # Label
            label_pos = world_to_img(center)
            cv2.putText(img, type_str, (label_pos[0]+10, label_pos[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(img, label_pos, 5, color, -1)

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
    node = QuadScanner()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
