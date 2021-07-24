import math
import numpy as np

def get_pos_in_pixel(pos, size, x_max, y_max):
    return (pos.get("x") * x_max / 100)-(size/2), (pos.get("y") * y_max / 100)-(size/2)

def create_ellipsis(centerx,centery, a, b, points = 100):
    # devide the Ellipse into 100 pieces 
    ellipsis = []
    piece = 2*3.14 / points
    for i in range(points):
        angle = piece*i
        ellipsis.insert(i,{
            "x" : (centerx + a * math.cos(angle)),
            "y" : (centery + b * math.sin(angle))
        })
    return ellipsis

# returns the index of the nearest point to the attacker on the escape-ellipsis
def get_attacker_possition_index_on_ellipse(ellipsis, a_x, a_y):
    dist_to_ellipse_points = []
    for i in range(len(ellipsis)):
        distance = math.sqrt((ellipsis[i].get("x") - a_x)**2 + (ellipsis[i].get("y") - a_y)**2)
        dist_to_ellipse_points.insert(i, distance)
    return dist_to_ellipse_points.index(min(dist_to_ellipse_points))

def ensure_bounds(variable, min, max):
    variable = min if variable <= min else variable
    variable = max if variable >= max else variable
    return variable

#  returns a Sigmoid similar function for 0 <= x <= 1
def GetQuad(x):
    assert((0 <= x) and (x <= 1))
    return  2 * pow(x, 2) if (x < 0.5) else 1 - 2 * pow(x - 1, 2)

def GetQuadInv(y):
    assert((y >= 0) and (y <= 1))
    return math.sqrt(0.5 * y) if (y < 0.5) else 1 - math.sqrt(0.5 * (1 - y))

# 10, self.a_max, self.v_max, self.v, self.position,self.escape_trajectory
def move(dt, v, v_max, a_max, position, trajectory):
    if len(trajectory) > 1:
        # get all x values
        trajectory = np.asarray(trajectory)
        x_traj = trajectory[:,0]
        y_traj = trajectory[:,1]
        # Points where the robot should change direction 
        x_curve = get_next_curve(position.get("x"),x_traj)
        y_curve = get_next_curve(position.get("y"),y_traj)
        # Distance to the direction-change points 
        d_x = x_curve - position.get("x")
        d_y = y_curve - position.get("y")

        # Max velocity desired to reach the curve with (v_max) or without gliding phase (math.sqrt(abs(d_x) * a_max))
        movement_v_x_max = np.sign(d_x) * min(v_max, math.sqrt(abs(d_x) * a_max))
        movement_v_y_max = np.sign(d_x) * min(v_max, math.sqrt(abs(d_x) * a_max))

        s_a, s_g, s_b = get_state_change_positions(d_x, movement_v_x_max, a_max)
        



    #  direction is considered to be positive if the rpj of the latest created curve is greater
    #  than the current point
    #  value {-1,0,1} : {negative,no,positive} direction
def get_next_curve(x, trajectory):
    minima = (np.r_[True, trajectory[1:] < trajectory[:-1]] & np.r_[trajectory[:-1] < trajectory[1:], True])[1:]
    maxima = (np.r_[True, trajectory[1:] > trajectory[:-1]] & np.r_[trajectory[:-1] > trajectory[1:], True])[1:]
    is_turning_point = np.append(False,  np.logical_or(minima,maxima))
    
    d_s = 0
    j = -1
    turning_point_indexes = np.where(is_turning_point == True)
    next_turning_point = trajectory[min(turning_point_indexes[0])]
    return next_turning_point

# s_a = Acceleration, Gliding, Braking
def get_state_change_positions(d_x, v_max, a_max):
    # Robot Position where Acceleration Phase ends:
    s_a = 0.5*pow(v_max,2)/a_max
    # Robot Position where Gliding Phase ends:
    s_g = d_x - s_a
    # Robot Position where Braking Phase ends:
    s_b = d_x
    return s_a, s_g, s_b

    
