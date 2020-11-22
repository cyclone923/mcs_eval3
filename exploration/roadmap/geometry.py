import math
import matplotlib.pyplot as plt

class Geometry:
    class Point:
        def __init__(self, x, y):
            self.x = float(x)
            self.y = float(y)

    @staticmethod
    def segmentIntersectCircle(a, b, c, r):
        ax = a.x - c.x
        ay = a.y - c.y
        bx = b.x - c.x
        by = b.y - c.y

        cv = ax**2 + ay**2 - r**2;
        bv = 2*(ax*(bx - ax) + ay*(by - ay))
        av = (bx - ax)**2 + (by - ay)**2


        disc = bv**2 - 4*av*cv;
        
        if disc > 0 : 
            sqrtdisc = math.sqrt(disc);
            t1 = (-bv + sqrtdisc)/(2*av);
            t2 = (-bv - sqrtdisc)/(2*av);

            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                #circle = plt.Circle((c.x, c.y), radius=r, color="r", fill=False)
                #plt.gca().add_artist(circle) 
                return True
            
        #circle = plt.Circle((c.x, c.y), radius=r, color="g", fill=False)
        #plt.gca().add_artist(circle)
        
        return False

    @staticmethod
    def is_seg_intersect(p1, q1, p2, q2):

        def on_segment(p, q, r):
            if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
                    (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
                return True
            return False

        def orientation(p, q, r):
            val = (float(q.y - p.y) * (r.x - q.x)) - (
                    float(q.x - p.x) * (r.y - q.y))
            if val > 0:
                return 1
            if val < 0:
                return 2
            return 0

        # Find the 4 orientations required for
        # the general and special cases
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if (o1 != o2) and (o3 != o4):
            return True
        if (o1 == 0) and on_segment(p1, p2, q1):
            return True
        if (o2 == 0) and on_segment(p1, q2, q1):
            return True
        if (o3 == 0) and on_segment(p2, p1, q2):
            return True
        if (o4 == 0) and on_segment(p2, q1, q2):
            return True

        return False