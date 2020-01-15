
class Ball:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        self.canvas.move(self.id, 245, 100)
    def draw(self):
        pass

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Segment:
    def __init__(self,A,B,width=1):
        self.x1 = A.x
        self.y1 = A.y
        self.x2 = B.x
        self.y2 = B.y
        self.width = width

def segmentIntersection(AB,CD):
    # Finding the intersection of segment in a vector view
    # If A + m*AB = C + k*CD with k and m in [0;1] than there is an intersection
    # Let I be the vector AB and J the vector CD
    I = Point(AB.x2-AB.x1, AB.y2-AB.y1)
    J = Point(CD.x2-CD.x1, CD.y2-CD.y1)
    m = -(-I.x*AB.y1 + I.x*CD.y1 + I.y*AB.x1 + I.y*CD.x1) / (I.x*J.y - I.y*J.x)
    k = -(-J.y*AB.x1 + J.y*CD.x1 + J.x*AB.y1 + J.x*CD.y1) / (I.x*J.y - I.y*J.x)
    return (0 <= m <= 1) and (0 <= k <= 1)

# def rectangleIntersection(rect1,rect2):

s1 = Segment(Point(0,0),Point(10,0))
s2 = Segment(Point(0,-1),Point(10,-0.000000))
print(segmentIntersection(s1,s2))
