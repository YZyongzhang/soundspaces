import quaternion
import numpy as np

import numpy as np
from scipy.spatial.transform import Rotation as R

from habitat_sim.utils.common import *

def quat_to_angle(q):
    # rotation = R.from_quat(quaternion.as_float_array(q))
    # base_vector = np.array([1., 0., 0.])
    # rotated_vector = rotation.apply(base_vector)
    # return rotated_vector
    return quat_rotate_vector(q, np.array([1., 0., 0.]))

def angle_to_quat(angle):
    # norm_angle = angle / np.linalg.norm(angle)
    # base_vector = np.array([1., 0., 0.])
    # rotation, _ = R.align_vectors([base_vector], [norm_angle])
    # q = rotation.as_quat()
    # return quaternion.from_float_array(q)
    return quat_from_two_vectors(np.array([1., 0., 0.]), angle)

def random_quat():
    q = np.random.rand(4)
    q[1] = q[3] = 0
    q = q / np.linalg.norm(q)
    return quaternion.from_float_array(q)


def angle_between_vectors(v1, v2):
    if len(v1) == 2:
        v1 = np.insert(v1, 1, 0)
        v2 = np.insert(v2, 1, 0)
    q1 = angle_to_quat(v1)
    q2 = angle_to_quat(v2)

    angle = angle_between_quats(q1, q2)

    if np.cross(v1, v2)[1] < 0:
        angle = -angle

    return np.degrees(angle)


def angle_diff(angle1, angle2):
    if type(angle1) == quaternion.quaternion:
        angle1 = quat_to_angle(angle1)
    if type(angle2) == quaternion.quaternion:
        angle2 = quat_to_angle(angle2)

    return angle1 - angle2


if __name__ == "__main__":
    a1 = np.array([1,0,0])
    a2 = np.array([0.5, 0, -np.sqrt(3)/2])
    q1 = angle_to_quat(a1)
    q2 = angle_to_quat(a2)
    aq1 = quat_to_angle(q1)
    aq2 = quat_to_angle(q2)

    print(f"q1: {q1}")
    print(f"q2: {q2}")
    print(f"aq1: {aq1}")
    print(f"aq2: {aq2}")
    print(f"angle between {angle_between_vectors(a2,a1)}")
    print(f"angle between {angle_between_vectors(aq1,aq2)}")
