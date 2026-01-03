import math

class GeometryUtils:
    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def get_centroid(points):
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        return (cx, cy)

class TrinityDetector:
    """ Detects the 3-Pin Equilateral Triangle (Robust/Loose Mode) """
    def __init__(self):
        # --- TUNING SETTINGS (From successful tests) ---
        # Accept triangles side lengths between 4cm and 35cm
        self.MIN_SIDE = 0.04   
        self.MAX_SIDE = 0.35   
        # "Wobble Factor": Max difference between longest and shortest side (8cm)
        self.MAX_DIFF = 0.08   

    def detect(self, pins) -> list:
        """ Returns [Center(x,y), [pin1, pin2, pin3]] if found, else [None, []] """
        if len(pins) < 3: return [None, []]

        # Check every combination of 3 pins
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                for k in range(j + 1, len(pins)):
                    group = [pins[i], pins[j], pins[k]]
                    
                    # Calculate 3 sides
                    d1 = GeometryUtils.dist(group[0], group[1])
                    d2 = GeometryUtils.dist(group[1], group[2])
                    d3 = GeometryUtils.dist(group[2], group[0])
                    sides = [d1, d2, d3]

                    shortest = min(sides)
                    longest = max(sides)
                    avg = sum(sides) / 3.0

                    # CHECK 1: Size Range (Is it too big or too small?)
                    if not (self.MIN_SIDE < avg < self.MAX_SIDE):
                        continue

                    # CHECK 2: Equilateral Shape (Are sides roughly equal?)
                    if (longest - shortest) < self.MAX_DIFF:
                        return [GeometryUtils.get_centroid(group), group]
        
        return [None, []]

class QuadDetector:
    """ Detects 4-Pin Square OR 3-Pin Corner (Robust/Loose Mode) """
    def __init__(self):
        # --- TUNING SETTINGS (From successful tests) ---
        # Side length range (8cm to 16cm)
        self.MIN_SIDE = 0.08   
        self.MAX_SIDE = 0.16   
        # Ratio Check: Diagonal must be ~1.4x the Side (Range 1.2 - 1.6)
        self.MIN_RATIO = 1.2
        self.MAX_RATIO = 1.6

    def detect(self, pins) -> list:
        """ Returns [Center(x,y), [pins...]] if found, else [None, []] """
        num = len(pins)
        if num < 3: return [None, []]

        # --- PRIORITY 1: FULL 4-PIN SQUARE ---
        if num >= 4:
            for i in range(num):
                for j in range(i+1, num):
                    for k in range(j+1, num):
                        for l in range(k+1, num):
                            group = [pins[i], pins[j], pins[k], pins[l]]
                            
                            # Measure all 6 distances
                            dists = []
                            for p1_idx in range(4):
                                for p2_idx in range(p1_idx+1, 4):
                                    dists.append(GeometryUtils.dist(group[p1_idx], group[p2_idx]))
                            dists.sort() # Smallest to Largest

                            # 4 short sides, 2 long diagonals
                            avg_side = sum(dists[:4]) / 4.0
                            avg_diag = sum(dists[4:]) / 2.0

                            # Check Size
                            if not (self.MIN_SIDE < avg_side < self.MAX_SIDE): continue

                            # Check Shape (Diagonal/Side Ratio)
                            ratio = avg_diag / avg_side
                            if self.MIN_RATIO < ratio < self.MAX_RATIO:
                                return [GeometryUtils.get_centroid(group), group]

        # --- PRIORITY 2: BROKEN 3-PIN CORNER ---
        # If no full square, look for "L" shape (Right Isosceles Triangle)
        for i in range(num):
            for j in range(i+1, num):
                for k in range(j+1, num):
                    group = [pins[i], pins[j], pins[k]]
                    
                    # Measure 3 sides and keep track of points
                    dists_map = [] 
                    dists_map.append((GeometryUtils.dist(group[0], group[1]), group[0], group[1]))
                    dists_map.append((GeometryUtils.dist(group[1], group[2]), group[1], group[2]))
                    dists_map.append((GeometryUtils.dist(group[2], group[0]), group[2], group[0]))
                    
                    # Sort by distance: [Small, Small, Large]
                    dists_map.sort(key=lambda x: x[0])
                    
                    short1 = dists_map[0][0]
                    short2 = dists_map[1][0]
                    long_side = dists_map[2][0]
                    
                    avg_short = (short1 + short2) / 2.0

                    # Check 1: Size
                    if not (self.MIN_SIDE < avg_short < self.MAX_SIDE): continue

                    # Check 2: Isosceles (Two short sides roughly equal)
                    if abs(short1 - short2) > 0.04: continue 

                    # Check 3: Right Angle (Hypotenuse Ratio)
                    ratio = long_side / avg_short
                    if self.MIN_RATIO < ratio < self.MAX_RATIO:
                        # Center of square is midpoint of the diagonal (long side)
                        # The long side connects the two outer pins.
                        p_start = dists_map[2][1]
                        p_end = dists_map[2][2]
                        cx = (p_start[0] + p_end[0]) / 2.0
                        cy = (p_start[1] + p_end[1]) / 2.0
                        return [(cx, cy), group]

        return [None, []]
