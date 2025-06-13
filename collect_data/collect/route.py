import random
from tongsim.common.ue_types import UELocation

def generate_straight_path(start, goal, num_steps=100):
    path = [start]
    for i in range(1, num_steps):
        t = i / num_steps
        x = start.X + t * (goal.X - start.X)
        y = start.Y + t * (goal.Y - start.Y)
        z = start.Z + t * (goal.Z - start.Z)
        path.append(UELocation(x, y, z))
    path.append(goal)
    return path

def apply_disturbance(path, epsilon=0.1):
    disturbed_path = []
    for point in path:
        disturbed_point = UELocation(point.X + point.X * random.uniform(-epsilon, epsilon),
                                    point.Y + point.Y * random.uniform(-epsilon, epsilon),
                                    point.Z + point.Z * random.uniform(-epsilon, epsilon))
        disturbed_path.append(disturbed_point)
    return disturbed_path

def generate_start_point(base):
    delta_x = [random.randint(-50, 400), random.randint(500, 700)]
    delta_y = [random.randint(0, 400), random.randint(200, 500)]
    i = random.randint(0, 1)
    start = UELocation(base.X + delta_x[i], base.Y + delta_y[i], base.Z)
    return start