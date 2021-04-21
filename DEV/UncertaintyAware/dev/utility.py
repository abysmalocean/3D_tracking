import numpy as np

def angle_difference(x, y):
    """
    Get the smallest difference between 2 angles - x, y.
    Both the input and output are in radian.
    Output is in (-pi, pi] range.
    >>> angle_difference(math.pi, math.pi)
    0.0
    >>> angle_difference(math.pi, -math.pi)
    0.0
    >>> angle_difference(math.pi/2, -math.pi/2)
    3.141592653589793
    >>> angle_difference(math.pi/4, -math.pi/4)
    1.5707963267948966
    >>> angle_difference(math.pi/6, 11*math.pi/6)
    1.0471975511965983
    """
    diff = (x - y) % (2 * np.pi)
    if diff > np.pi:
        return diff - (2 * np.pi)
    else:
        return diff
