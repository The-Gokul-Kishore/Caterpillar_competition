import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ==========================================
# PART 1: THE GENERATOR (Smoother Rocks, Shorter Marker)
# ==========================================
def generate_simulation(filename="sim_easier_challenge.png"):
    map_size = 640
    img = np.zeros((map_size, map_size), dtype=np.uint8)
    
    # --- 1. DEFINE MARKER (Shorter Spikes) ---
    mx = random.randint(200, 440)
    my = random.randint(200, 440)
    core_radius = 15
    # REDUCED LENGTH as requested (was 75)
    spike_len = 45 
    num_spikes = 3
    
    # Draw Core
    cv2.ellipse(img, (mx, my), (core_radius, core_radius), 0, 180, 360, 255, -1)
    
    # Draw Spikes
    start_angle = 180
    end_angle = 360
    for i in range(num_spikes):
        angle_deg = start_angle + (i * (end_angle - start_angle) / (num_spikes - 1))
        angle_rad = math.radians(angle_deg)
        sx = int(mx + core_radius * math.cos(angle_rad))
        sy = int(my + core_radius * math.sin(angle_rad))
        ex = int(mx + (core_radius + spike_len) * math.cos(angle_rad))
        ey = int(my + (core_radius + spike_len) * math.sin(angle_rad))
        cv2.line(img, (sx, sy), (ex, ey), 255, 10)

    # --- 2. GENERATE ROCKS (Less Spiky) ---
    num_rocks = 8
    rocks_placed = 0
    marker_safe_zone = spike_len + 60
    
    while rocks_placed < num_rocks:
        rx = random.randint(50, 590)
        ry = random.randint(50, 590)
        if math.sqrt((rx-mx)**2 + (ry-my)**2) < marker_safe_zone: continue

        # Make rocks smoother ("lumpy potatoes")
        # Reduced vertices (was 10-20)
        num_vertices = random.randint(8, 15) 
        avg_radius = random.randint(40, 70)
        pts = []
        for i in range(num_vertices):
            angle_rad = math.radians((360 / num_vertices) * i)
            # REDUCED VARIANCE (was 0.5-1.5). Closer to 1.0 = rounder.
            variance = random.uniform(0.8, 1.3) 
            r = avg_radius * variance
            x = int(rx + r * math.cos(angle_rad))
            y = int(ry + r * math.sin(angle_rad))
            pts.append([x, y])
            
        cv2.fillPoly(img, [np.array(pts, np.int32)], 255)
        rocks_placed += 1

    # --- 3. ADD NOISE ---
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    noise = np.random.randint(0, 100, (map_size, map_size))
    blurred[noise > 96] = 255
    cv2.imwrite(filename, blurred)
    print(f"[GEN] Map created with smoother rocks and shorter marker.")
    return filename
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def detect_marker_on_screenshot(image_path):
    print(f"\n[DET] Analyzing {image_path}...")
    
    # 1. Load Image
    img = cv2.imread(image_path, 0)
    if img is None:
        print("Error: Could not load image.")
        return

    # 2. AUTO-INVERT (Fix for White Backgrounds)
    # If the image is mostly bright, it's a white background.
    # We invert it so the objects become White and background becomes Black.
    if np.mean(img) > 127:
        print("[INFO] White background detected. Inverting...")
        img = cv2.bitwise_not(img)
    
    # 3. Thresholding
    # We use 127 to kill the faint grid lines (which are usually grey)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 4. Find Contours (THE CRITICAL FIX)
    # RETR_TREE retrieves all contours, even those nested inside other shapes (like your star inside the arena)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    print(f"[INFO] Found {len(contours)} contours.")
    found_count = 0
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Filter 1: Size
        # Ignore tiny noise (<100) and ignore the GIANT arena wall (>90% of image)
        img_area = img.shape[0] * img.shape[1]
        if area < 100 or area > (img_area * 0.9): 
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # --- METRIC 1: SOLIDITY ---
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        
        # --- METRIC 2: PEAK COUNT ---
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        distances = []
        for point in cnt:
            px, py = point[0]
            d = math.sqrt((px - cx)**2 + (py - cy)**2)
            distances.append(d)
        
        # Lower prominence slightly to catch blurry screenshot spikes
        peaks, _ = find_peaks(distances, prominence=3, width=2)
        num_peaks = len(peaks)
        
        # --- CLASSIFICATION ---
        is_marker = False
        
        # Star Logic:
        # 1. Low Solidity (It's spiky, not a round rock)
        # 2. Correct number of spikes (Your star has 8, so we accept 7-9)
        if solidity < 0.85:
            if 4<= num_peaks <= 10: 
                is_marker = True
            
        # Draw Results
        if is_marker:
            found_count += 1
            color = (0, 255, 0) # Green
            label = "STAR"
            info = f"Sol:{solidity:.2f} | Pks:{num_peaks}"
            print(f"   >>> Object #{i}: ✅ MARKER FOUND! ({info})")
            
            # Thick box for visibility
            cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
            # Draw label above the box
            cv2.putText(output_img, label, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(output_img, info, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            # Draw red box for "Other Stuff" (like the C-shaped markings)
            color = (0, 0, 255) # Red
            cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 1)

    # Save and Show
    cv2.imwrite("fixed_result.png", output_img)
    print(f"\n[DONE] Found {found_count} stars. Saved to 'fixed_result.png'")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Result: {found_count} Stars Detected")
    plt.show()

if __name__ == "__main__":
    detect_marker_on_screenshot("image_many.png")