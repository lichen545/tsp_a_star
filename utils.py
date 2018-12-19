import math

def rectilinear(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean(a, b):
    (x_1, y_1) = a
    (x_2, y_2) = b
    return math.sqrt(((x_1 - x_2) ** 2) + ((y_1 - y_2) ** 2))