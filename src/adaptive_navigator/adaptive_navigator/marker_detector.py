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
    """ Detects the 3-Pin Equilateral Triangle (Trinity) """
    def __init__(self, target_side=0.173, tolerance=0.025):
        self.TARGET_SIDE = target_side
        self.TOLERANCE = tolerance

    def detect(self, pins)->list[list]:
        """ Returns Center (x,y) if Trinity found, else None """
        if len(pins) < 3: return [None,[]]

        # Check every combination of 3 pins
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                for k in range(j + 1, len(pins)):
                    group = [pins[i], pins[j], pins[k]]
                    
                    # Calculate sides
                    d1 = GeometryUtils.dist(group[0], group[1])
                    d2 = GeometryUtils.dist(group[1], group[2])
                    d3 = GeometryUtils.dist(group[2], group[0])

                    # Check for Equilateral Triangle (All sides ~17cm)
                    if (abs(d1 - self.TARGET_SIDE) < self.TOLERANCE and 
                        abs(d2 - self.TARGET_SIDE) < self.TOLERANCE and 
                        abs(d3 - self.TARGET_SIDE) < self.TOLERANCE):
                        
                        return [GeometryUtils.get_centroid(group),[pins[i], pins[j], pins[k]]]
        return [None,[]]

class QuadDetector:
    """ Detects the 4-Pin Square (Quad) AND the Broken 3-Pin Corner """
    def __init__(self, side_len=0.12, diag_len=0.17, tolerance=0.025):
        self.SIDE = side_len
        self.DIAG = diag_len
        self.TOLERANCE = tolerance

    def detect(self, pins)->list[list]:
        """ Returns Center (x,y) if Quad found, else None """
        num = len(pins)
        if num < 3: return [None,[]]

        # --- CASE 1: PERFECT QUAD (4 PINS) ---
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
                            dists.sort()

                            # 4 short sides, 2 long diagonals
                            if (all(abs(d - self.SIDE) < self.TOLERANCE for d in dists[:4]) and
                                all(abs(d - self.DIAG) < self.TOLERANCE for d in dists[4:])):
                                return [GeometryUtils.get_centroid(group),[pins[i], pins[j], pins[k], pins[l]]]

        # --- CASE 2: BROKEN QUAD (3 PINS / CORNER) ---
        if num >= 3:
            for i in range(num):
                for j in range(i+1, num):
                    for k in range(j+1, num):
                        group = [pins[i], pins[j], pins[k]]
                        
                        # Measure 3 sides
                        dists = []
                        dists.append(GeometryUtils.dist(group[0], group[1]))
                        dists.append(GeometryUtils.dist(group[1], group[2]))
                        dists.append(GeometryUtils.dist(group[2], group[0]))
                        dists.sort() # Small, Small, Large

                        # Check for Right Triangle (12cm, 12cm, 17cm)
                        if (abs(dists[0] - self.SIDE) < self.TOLERANCE and 
                            abs(dists[1] - self.SIDE) < self.TOLERANCE and
                            abs(dists[2] - self.DIAG) < self.TOLERANCE):
                            
                            # The center of a square is the midpoint of the diagonal
                            # Find the two points that make the long diagonal (the largest distance)
                            # (We know dists[2] is the largest)
                            p1, p2, p3 = group[0], group[1], group[2]
                            
                            # Bruteforce find which pair is the diagonal
                            if abs(GeometryUtils.dist(p1, p2) - dists[2]) < 0.01:
                                return [GeometryUtils.get_centroid([p1, p2]),[p1, p2, p3]]
                            elif abs(GeometryUtils.dist(p2, p3) - dists[2]) < 0.01:
                                return [GeometryUtils.get_centroid([p2, p3]),[p1, p2, p3]]
                            else:
                                return [GeometryUtils.get_centroid([p3, p1]),[p1, p2, p3]]
        return [None,[]]