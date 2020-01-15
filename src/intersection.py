"""Calculate the distance between line segments."""

import math


class Point(object):
    """A two dimensional point."""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class LineSegment(object):
    """A line segment in a two dimensional space."""
    def __init__(self, p1, p2):
        assert isinstance(p1, Point), \
            "p1 is not of type Point, but of %r" % type(p1)
        assert isinstance(p2, Point), \
            "p2 is not of type Point, but of %r" % type(p2)

        if (p1.x < p2.x) or (p1.x == p2.x and p1.y < p2.y):
            self.p1 = p1
            self.p2 = p2
        else:
            self.p1 = p2
            self.p2 = p1

        self.a = None
        self.b = None
        if(self.p1.x != self.p2.x):
            self.a = (self.p2.y - self.p1.y)/(self.p2.x - self.p1.x)
            self.b = self.p1.y - self.p1.x*self.a

    def contains(self, p):
        assert isinstance(p, Point), \
            "p is not of type Point, but of %r" % type(p1)
        if p.x < min(self.p1.x, self.p2.x) or \
        p.x > max(self.p1.x, self.p2.x) or \
        p.y < min(self.p1.y, self.p2.y) or \
        p.y > max(self.p1.y, self.p2.y) :
            return False
        if self.a == None:
            return False
        return self.a*p.x + self.b == p.y


def segments_distance(segment1, segment2):
    """Calculate the distance between two line segments in the plane.
    """
    if segments_intersect(segment1, segment2):
        return 0
    # try each of the 4 vertices w/the other segment
    distances = []
    distances.append(point_segment_distance(segment1.p1, segment2))
    distances.append(point_segment_distance(segment1.p2, segment2))
    distances.append(point_segment_distance(segment2.p1, segment1))
    distances.append(point_segment_distance(segment2.p2, segment1))
    return min(distances)


def segments_intersect(segment1, segment2):
    """Check if two line segments in the plane intersect.
    """
    dx1 = segment1.p2.x - segment1.p1.x
    dy1 = segment1.p2.y - segment1.p2.y
    dx2 = segment2.p2.x - segment2.p1.x
    dy2 = segment2.p2.y - segment2.p1.y
    delta = dx2 * dy1 - dy2 * dx1
    if delta == 0:  # parallel segments
        return segment2.contains(segment1.p1) or \
        segment2.contains(segment1.p2) or \
        segment1.contains(segment2.p1)

    s = (dx1 * (segment2.p1.y - segment1.p1.y) +
         dy1 * (segment1.p1.x - segment2.p1.x)) / delta
    t = (dx2 * (segment1.p1.y - segment2.p1.y) +
         dy2 * (segment2.p1.x - segment1.p1.x)) / (-delta)
    return (0 <= s <= 1) and (0 <= t <= 1)


def point_segment_distance(point, segment):
    """

    """
    assert isinstance(point, Point), \
        "point is not of type Point, but of %r" % type(point)
    dx = segment.p2.x - segment.p1.x
    dy = segment.p2.y - segment.p1.y
    if dx == dy == 0:  # the segment's just a point
        return math.hypot(point.x - segment.p1.x, point.y - segment.p1.y)

    if dx == 0:
        if (point.y <= segment.p1.y or point.y <= segment.p2.y) and \
           (point.y >= segment.p2.y or point.y >= segment.p2.y):
            return abs(point.x - segment.p1.x)

    if dy == 0:
        if (point.x <= segment.p1.x or point.x <= segment.p2.x) and \
           (point.x >= segment.p2.x or point.x >= segment.p2.x):
            return abs(point.y - segment.p1.y)

    # Calculate the t that minimizes the distance.
    t = ((point.x - segment.p1.x) * dx + (point.y - segment.p1.y) * dy) / \
        (dx * dx + dy * dy)

    # See if this represents one of the segment's
    # end points or a point in the middle.
    if t < 0:
        dx = point.x - segment.p1.x
        dy = point.y - segment.p1.y
    elif t > 1:
        dx = point.x - segment.p2.x
        dy = point.y - segment.p2.y
    else:
        near_x = segment.p1.x + t * dx
        near_y = segment.p1.y + t * dy
        dx = point.x - near_x
        dy = point.y - near_y

    return math.hypot(dx, dy)

if __name__ == '__main__':
    import doctest
    doctest.testfile("intersection_doctest.txt")
    # doctest.testmod()
