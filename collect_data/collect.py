import numpy as np
import os
import wave
import math
import random
import cv2
import torch
from PIL import Image
import io
from tongsim.common.ue_types import UEImageRequest, UEImageType
from ultralytics import YOLO
import matplotlib.pyplot as plt

MAX_STEPS = 100
SURPRISE_INIT = 100.0
MAX_TURN_ANGLE = 30

# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model = YOLO("yolo11n.pt")
yolo_model.conf = 0.25


def calculate_target_degree(agent_location, object_location):
    dx = object_location.X - agent_location.X
    dy = object_location.Y - agent_location.Y
    theta_rad = math.atan2(dy, dx)
    theta_deg = math.degrees(theta_rad) % 360
    return theta_deg

def get_current_degree(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp)) % 360
    return yaw

# def add_gaussian_noise(img, noise_level=0.1):
#     img = img.astype(np.float32) / 255.0
#     noise = np.random.normal(0, noise_level, img.shape)
#     noised_img = img + noise
#     noised_img = np.clip(noised_img, 0, 1) * 255 
#     return noised_img.astype(np.uint8) 

# def add_white_noise(audio, alpha=0.1):
#     if alpha == 0:
#         return audio
#     rms = np.sqrt(np.mean(audio**2))
#     noise = np.random.normal(0, alpha * rms, audio.shape)
#     return np.clip(audio + noise, -1.0, 1.0)

# def add_action_noise(action, alpha=0.1):
#     if alpha == 0:
#         return action
#     delta = random.uniform(-alpha * abs(action), alpha * abs(action))
#     noisy_action = action + delta
#     return noisy_action

def calculate_relative_angle(agent_location, agent_rotation, object_location):
    current_degree = get_current_degree(agent_rotation.X, agent_rotation.Y, agent_rotation.Z, agent_rotation.W)
    target_degree = calculate_target_degree(agent_location, object_location)
    relative_angle = (target_degree - current_degree) % 360
    if relative_angle > 180:
        relative_angle = relative_angle - 360
    return relative_angle

# def quantize_action(action):
#     discrete_action = min(DISCRETE_ACTIONS, key=lambda x: abs(x - action))
#     return discrete_action

def calculate_distance(agent_location, object_location):
    dx = object_location.X - agent_location.X
    dy = object_location.Y - agent_location.Y
    return math.sqrt(dx**2 + dy**2)

# def calculate_reward(agent, object):
#     agent_location = agent.get_pose().location
#     agent_rotation = agent.get_pose().rotation
#     object_location = object.get_pose().location
#     angle = calculate_relative_angle(agent_location, agent_rotation, object_location)
#     reward = 1 - (abs(angle) / 180)
#     return reward

def collect_acoustic_data(agent, duration=0.5, sample_rate=16000):
    num_samples = int(duration * sample_rate)
    collected_samples = []
    while len(collected_samples) < num_samples:
        new_data = agent.get_acoustics_data()
        collected_samples.extend(new_data)
    return np.array(collected_samples[::2]), np.array(collected_samples[1::2])


def process_acoustic_data(left_channel, right_channel, left_path, right_path):
    left = np.int16(np.array(left_channel) * 32767).flatten()
    right = np.int16(np.array(right_channel) * 32767).flatten()
    # with wave.open(left_path, 'wb') as wav_file:
    #     wav_file.setnchannels(1)
    #     wav_file.setsampwidth(left.itemsize)
    #     wav_file.setframerate(16000)
    #     wav_file.writeframes(left.tobytes())
    # with wave.open(right_path, 'wb') as wav_file:
    #     wav_file.setnchannels(1)
    #     wav_file.setsampwidth(right.itemsize)
    #     wav_file.setframerate(16000)
    #     wav_file.writeframes(right.tobytes())


def collect_camera_data(agent, image_path):
    img_request_left = [UEImageRequest(agent.id + "_CenterEye", UEImageType.RGB)]
    response = agent.get_images(img_request_left)
    image_data = response[0].image_data_bytes
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)
    # cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def update_surprise(S_prev, distance, angle, confidence, alpha=1.0, gamma=0.05):
    f_dist = np.exp(-gamma * distance)
    f_angle = np.exp(-gamma * angle)
    delta = alpha * f_dist * f_angle * confidence
    return max(S_prev - delta, 0)


def estimate_confidence(image_path, event_id):
    results = yolo_model(image_path)
    for result in results:
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        names = [result.names[int(cls)] for cls in result.boxes.cls.int()]  # Get class names for each box
        confs = result.boxes.conf  # confidence scores
        
        target_detections = [(xy, conf) for xy, name, conf in zip(xyxy, names, confs) if name == event_id]
    
        if len(target_detections) == 0:
            return 0.0
        
        max_confidence = max(target_detections, key=lambda x: x[1])[1]
        return float(max_confidence)
    
    return 0.0


def generate_event_data(event_id, agent, object, image_path, S_prev):
    rel_angle = calculate_relative_angle(
        agent.get_pose().location, agent.get_pose().rotation, object.get_pose().location
    )
    rel_dist = calculate_distance(agent.get_pose().location, object.get_pose().location)
    confidence = estimate_confidence(image_path, event_id)
    surprise = update_surprise(S_prev, rel_dist, rel_angle, confidence)
    return {
        "event_id": event_id,
        "rel_angle": rel_angle, "rel_dist": rel_dist,
        "confidence_proxy": confidence, "surprise": surprise
    }


def collect_experience(agent, fan, tv, folder_path):
    audio_folder = os.path.join(folder_path, 'audio')
    os.makedirs(audio_folder, exist_ok=True)
    visual_folder = os.path.join(folder_path, 'visual')
    os.makedirs(visual_folder, exist_ok=True)
    prev_surprise_fan = SURPRISE_INIT
    prev_surprise_tv = SURPRISE_INIT
    experience = []

    for step in range(MAX_STEPS):
        delta_angle = random.uniform(0, MAX_TURN_ANGLE)
        agent.turn_around_to_degree(delta_angle)
        agent.do_task_and_wait_finish()
        
        left_channel_data, right_channel_data = collect_acoustic_data(agent)
        left_path = os.path.join(audio_folder, f'left_{step}.wav')
        right_path = os.path.join(audio_folder, f'right_{step}.wav')
        process_acoustic_data(left_channel_data, right_channel_data, left_path, right_path)
    
        image_path = os.path.join(visual_folder, f'image_{step}.png')
        collect_camera_data(agent, image_path)

        event_fan = generate_event_data('fan', agent, fan, image_path, prev_surprise_fan)
        prev_surprise_fan = event_fan['surprise']
        
        event_tv = generate_event_data('tv', agent, tv, image_path, prev_surprise_tv)
        prev_surprise_tv = event_tv['surprise']

        event_data = [event_fan, event_tv]
        experience.append({'audio_left': left_path, 'audio_right': right_path, 'image': image_path,
                            'action': delta_angle, 'events': event_data })

    return experience