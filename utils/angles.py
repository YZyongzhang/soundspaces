import quaternion


def quat_to_angle(q):
    angle = quaternion.as_rotation_vector(q)
    return angle


def angle_diff(angle1, angle2):
    if type(angle1) == quaternion.quaternion:
        angle1 = quat_to_angle(angle1)
    if type(angle2) == quaternion.quaternion:
        angle2 = quat_to_angle(angle2)

    return angle1 - angle2
