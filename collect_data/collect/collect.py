import os
import numpy as np
import pickle
from pathlib import Path

from tongsim.manager.scene_manager import SceneManager
from tongsim.common.ue_types import UELocation, UERotation, UEScale
from tongsim.common.enums import CameraMoveMode

from collect import collect_experience
from route import generate_start_point

class Camera:
    def __init__(self) -> None:
        self.channel = grpc.insecure_channel('127.0.0.1:5056', options=[
            ('grpc.max_send_message_length', 200 * 1024 * 1024),
            ('grpc.max_receive_message_length', 200 * 1024 * 1024),
        ])

    def subscribe_image(self, obj_id):
        camera_stub = camera_pb2_grpc.CameraServiceStub(self.channel)

        request = camera_pb2.ImageRequest()
        request.camera_config_list.extend([
            camera_pb2.CameraConfig(
                camera_id=obj_id + "_CenterEye",
                b_rgb=True,
                b_depth=False,
                b_segmentation=True,
                b_mirror_segmentation=False
            ),
            # Additional camera configurations...
        ])

        return camera_stub.SubscribeImage(request)

def init_env_event(water_machine , remote_controller):
    water_machine.set_state(True)
    remote_controller.set_state(True)
    remote_controller.set_channel("0")

def main():
    scene_manager = SceneManager(
        server_ip='127.0.0.1',
        server_port='50052',
        proto_server_port='5056'
    )

    scene_manager.open_level("005_1025_Acoustics")


    water_machine = scene_manager.get_object_by_name("BP_WaterMachine_Demo_C_1")
    remote_controller = scene_manager.get_object_by_name("BP_RemoteController_01_C_1")
    tv = scene_manager.get_object_by_name("BP_TV_Curved_C_3")

    init_env_event(water_machine,remote_controller)

    baby = scene_manager.spawn_ai_character(
        ai_character_asset_name="AIBabyV7",
        desired_name="agent",
        location=UELocation(100, 500, 60),
        rotation=UERotation(X=0, Y=0, Z=0, W=1),
        bsim_physics=False,
        with_camera=True,
        scale=UEScale(X=1, Y=1, Z=1),
        camera_mode=CameraMoveMode.THIRD_PERSON
    )

    base_location = UELocation(100, 500, 60)

    num_round = 10
    num_episode = 25
    
    # while step_count < num_steps:
    for episode_id in range(1):
        # target_angle = calculate_relative_angle(baby.get_pose().location, baby.get_pose().rotation, fan.get_pose().location)
        # offset_angle = np.random.uniform(90, 180)  
        # if np.random.rand() > 0.5:  
        #     offset_angle = -offset_angle
        # start_angle = target_angle + offset_angle  
        # start_angle = (start_angle + 180) % 360 - 180
        episode_folder = os.path.join(RAW_DIR, f"episode_{episode_id}")
        os.makedirs(episode_folder, exist_ok=True)
        data = []

        for round_id in range(num_round):
            round_folder = os.path.join(episode_folder, f"round_{round_id}")
            os.makedirs(round_folder, exist_ok=True)

            start_location = generate_start_point(base_location)
            baby.set_pose(UEPose(start_location,UERotation(random_rotation=True)))
            # start_location = scene_manager.get_random_point_in_room()
            # for alpha_action in NOISE_LEVELS:
            # alpha_sensor = sample_noise_level()
            experience = collect_experience(baby, water_machine, tv, round_folder)
            # step_count = step_count + len(experience)
            data.appends(experience)

        file_path = './test.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Experiences saved to {file_path}")
        data.clear()

    print('Data collection completed.')
    scene_manager.close_session()


if __name__ == "__main__":
    main()