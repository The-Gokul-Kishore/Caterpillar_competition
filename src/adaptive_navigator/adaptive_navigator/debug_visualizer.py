from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import cv2
import numpy as np
import os
class DebugVisualizer:
    def __init__(self, node):
        self.node = node
        self.pub = node.create_publisher(MarkerArray, '/detection_debug', 10)

    def publish_debug(self, pins, valid_shape_pins=None, shape_type=""):
        """
        pins: List of (x,y) tuples for ALL candidate clusters found
        valid_shape_pins: List of (x,y) tuples for the detected shape (if any)
        shape_type: "TRINITY" or "QUAD"
        """
        msg = MarkerArray()
        id_counter = 0

        # 1. DRAW ALL CANDIDATE PINS (Small Blue Spheres)
        # This shows you everything the clustering algorithm found
        for p in pins:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns = "candidates"
            m.id = id_counter
            id_counter += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(p[0])
            m.pose.position.y = float(p[1])
            m.pose.position.z = 0.1
            m.scale.x = 0.05; m.scale.y = 0.05; m.scale.z = 0.05
            m.color.a = 0.5; m.color.b = 1.0 # Semi-transparent Blue
            msg.markers.append(m)

        # 2. DRAW DETECTED SHAPE (Large Red/Green Spheres + Lines)
        if valid_shape_pins:
            # Color logic
            r, g, b = (0.0, 1.0, 0.0) if shape_type == "QUAD" else (1.0, 0.0, 0.0)

            # Draw Lines connecting the pins
            line_marker = Marker()
            line_marker.header.frame_id = "map"
            line_marker.header.stamp = self.node.get_clock().now().to_msg()
            line_marker.ns = "shapes"
            line_marker.id = 999
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.05 # Line width
            line_marker.color.a = 1.0
            line_marker.color.r = r; line_marker.color.g = g; line_marker.color.b = b
            
            # Close the loop
            points_to_draw = valid_shape_pins + [valid_shape_pins[0]]
            for p in points_to_draw:
                pt = Point()
                pt.x = float(p[0])
                pt.y = float(p[1])
                pt.z = 0.1
                line_marker.points.append(pt)
            msg.markers.append(line_marker)

            # Draw Label
            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = self.node.get_clock().now().to_msg()
            text.ns = "text"
            text.id = 1000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            centroid_x = sum(p[0] for p in valid_shape_pins)/len(valid_shape_pins)
            centroid_y = sum(p[1] for p in valid_shape_pins)/len(valid_shape_pins)
            text.pose.position.x = centroid_x
            text.pose.position.y = centroid_y
            text.pose.position.z = 0.5
            text.scale.z = 0.3 # Text height
            text.color.a = 1.0; text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0
            text.text = f"{shape_type} DETECTED"
            msg.markers.append(text)

        self.pub.publish(msg)
        
        


class LidarVisualizer:
    def __init__(self, save_dir='/home/ws/debug_lidar/'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # SETTINGS
        self.IMG_SIZE = 600       # 600x600 pixels
        self.SCALE = 100          # 1 meter = 100 pixels
        self.CENTER = self.IMG_SIZE // 2
        self.frame_count = 0

    def save_frame(self, raw_points, clusters, detected_shape_pins=None, shape_type=""):
        """
        raw_points: List of [x, y] from LiDAR
        clusters: List of [x, y] centers of found pins
        detected_shape_pins: List of [x, y] if a Trinity/Quad was confirmed
        shape_type: "TRINITY" or "QUAD"
        """
        # 1. Create Blank Black Image
        img = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)

        # 2. Draw Grid (Every 1 meter)
        color = (50, 50, 50) # Dark Gray
        for i in range(0, self.IMG_SIZE, self.SCALE):
            cv2.line(img, (i, 0), (i, self.IMG_SIZE), color, 1)
            cv2.line(img, (0, i), (self.IMG_SIZE, i), color, 1)
        
        # Draw Robot at Center
        cv2.circle(img, (self.CENTER, self.CENTER), 10, (255, 255, 255), -1) # White Robot

        # 3. Draw Raw LiDAR Points (Gray dots)
        for p in raw_points:
            px, py = self._world_to_pixel(p)
            cv2.circle(img, (px, py), 1, (150, 150, 150), -1)

        # 4. Draw Detected CLUSTERS/PINS (Blue Circles)
        # This shows you what the robot thinks is a "Pin"
        for c in clusters:
            px, py = self._world_to_pixel(c)
            # Draw a circle representing the pin
            cv2.circle(img, (px, py), 5, (255, 200, 0), 2) # Cyan/Blue ring
            cv2.putText(img, "Pin", (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # 5. Draw THE DETECTED SHAPE (Connecting lines)
        if detected_shape_pins:
            color = (0, 0, 255) if shape_type == "TRINITY" else (0, 255, 0) # Red for Trinity, Green for Quad
            
            # Connect the dots
            pixels = [self._world_to_pixel(p) for p in detected_shape_pins]
            for i in range(len(pixels)):
                p1 = pixels[i]
                p2 = pixels[(i+1) % len(pixels)] # Wrap around
                cv2.line(img, p1, p2, color, 2)
            
            # Write Label
            cv2.putText(img, f"{shape_type} FOUND!", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 6. Save Image
        filename = f"{self.save_dir}scan_{self.frame_count:05d}.jpg"
        cv2.imwrite(filename, img)
        self.frame_count += 1

    def _world_to_pixel(self, pt):
        # Convert (X, Y) meters to image coordinates
        # Map X -> Horizontal, Y -> Vertical
        # Note: In Grid maps, usually X is forward (Up), Y is Left.
        # Let's rotate -90 degrees so Forward is UP in the image.
        
        # Robot Frame: X=Forward, Y=Left
        # Image Frame: Y=Down, X=Right
        
        # X (Forward) -> moves UP in image (decrease pixel Y)
        # Y (Left)    -> moves LEFT in image (decrease pixel X)
        
        px = self.CENTER - int(pt[1] * self.SCALE) # Y world is X pixel (inverted)
        py = self.CENTER - int(pt[0] * self.SCALE) # X world is Y pixel (inverted)
        return (px, py)