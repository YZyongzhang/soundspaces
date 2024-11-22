import math
import random
random.seed(43)
class NextPoint:
    def __init__(self):
        self.KP = 0.25    # Forward step distance
        self.theta = 30 # Angle offset in degrees
    
    def get_step(self, setpoint, measurement):
        # Calculate the straight-line distance between two points
        x_s, y_s = setpoint[0], setpoint[2]
        x_m, y_m = measurement[0], measurement[2]
        e = math.sqrt((x_s - x_m)**2 + (y_s - y_m)**2)
        return self.KP * e
    # 方向对齐
    def alignment_direction():
        pass
    def sign(self):
        sign_ = [0,1]
        return random.choice(sign_)
    
    def get_next_position(self, setpoint, measurement):
        # Calculate the angle to the setpoint
        x_s, y_s = setpoint[0], setpoint[2]
        x_m, y_m = measurement[0], measurement[2]
        
        # Calculate angle from measurement to setpoint
        base_angle = abs(math.atan2(y_m - y_s, x_m - x_s))
        # Offset by kp
        # if self.sign() == 0:
        #     kp = self.KP - random.uniform(0,self.KP)
        # else:
        #     kp = self.KP + random.uniform(0,self.KP)
        # length = self.get_step(setpoint, measurement)
        length = math.sqrt((y_m-y_s)**2 + (x_m-x_s)**2)
        # Offset by theta (30 degrees)
        # if self.sign() == 0:
        angle_with_offset = base_angle + math.radians(random.randint(-self.theta,self.theta))
        # angle_with_offset = base_angle - math.radians(random.randint(0,self.theta))
        # else:
        # angle_with_offset = angle_with_offset + math.radians(random.randint(0,self.theta))
        # Calculate new position with KP step along the adjusted angle
        x_new = x_s + 0.25 * length * math.cos(angle_with_offset)
        y_new = y_s + 0.25 * length * math.sin(angle_with_offset)
        
        return x_new, y_new
