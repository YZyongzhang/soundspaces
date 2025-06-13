import os
import numpy as np
import pickle
from pathlib import Path

from tongsim.manager.scene_manager import SceneManager
from tongsim.common.ue_types import UELocation, UERotation, UEScale
from tongsim.common.enums import CameraMoveMode

from collect import collect_experience
from route import generate_start_point

RAW_DIR = Path("D:/projects/soundspaces/collect_data/yolo11/raw_data")
RAW_DIR.mkdir(parents=True, exist_ok=True)
EXPERIENCE_DIR = Path("D:/projects/soundspaces/collect_data/yolo11/data")
EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)

def take_plug_into_socket(scene, baby, plug, socket):
    plug_location = scene.calculate_move_to_location(baby.get_pose().location, plug.get_pose().location, 20)
    baby.move_to_location(plug_location)
    baby.turn_around_to_object(plug)
    baby.take_object(plug)

    socket_location = scene.calculate_move_to_location(baby.get_pose().location, socket.get_pose().location, 20)
    baby.move_to_location(socket_location)
    baby.turn_around_to_object(socket)
    baby.hand_reach_out_location(0, socket.get_interact_pos().location)
    baby.hand_release(0, socket.get_interact_location(), rotation=socket.get_interact_pos().rotation,
                      b_force_release=True, b_auto_rotate = True)
    baby.hand_reach_back(0)


def control_tv(baby, which_hand, remote_controller):
    baby.hand_reach_out_location(which_hand, baby.get_pose().location + baby.get_forward_vector() * 20)
    remote_controller.set_state(True)
    baby.hand_reach_back(which_hand)
    baby.do_task_and_wait_finish()


def main():
    scene_manager = SceneManager(
        server_ip='127.0.0.1',
        server_port='50052',
        proto_server_port='5056'
    )

    scene_manager.open_level("005_1025_Acoustics")

    fan = scene_manager.get_object_by_name("BP_FloorFan_01_C_1")
    fan_plug = fan.get_plug_objects()[0]
    fan_socket = scene_manager.get_object_by_name("BP_Plug_Socket_C_2")

    remote_controller = scene_manager.get_object_by_name("BP_RemoteController_01_C_0")
    tv = scene_manager.get_object_by_name("BP_TV_Curved_C_3")
    tv_plug = tv.get_plug_objects()[0]
    tv_socket = scene_manager.get_object_by_name("BP_Plug_Socket_C_0")

    baby = scene_manager.spawn_ai_character(
        ai_character_asset_name="AIBabyV6_6",
        desired_name="agent",
        location=UELocation(100, 500, 60),
        rotation=UERotation(X=0, Y=0, Z=0, W=1),
        bsim_physics=False,
        with_camera=True,
        scale=UEScale(X=1, Y=1, Z=1),
        camera_mode=CameraMoveMode.THIRD_PERSON
    )

    base_location = UELocation(100, 500, 60)
    take_plug_into_socket(scene_manager, baby, fan_plug, fan_socket)
    baby.do_task_and_wait_finish()

    baby.interact_object(fan)
    baby.do_task_and_wait_finish()

    take_plug_into_socket(scene_manager, baby, tv_plug, tv_socket)
    baby.do_task_and_wait_finish()

    baby.move_and_take_object(remote_controller, 1)
    baby.do_task_and_wait_finish()
    control_tv(baby, 1, remote_controller)

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
            # start_location = scene_manager.get_random_point_in_room()
            baby.move_to_location(start_location)
            baby.do_task_and_wait_finish()
            # for alpha_action in NOISE_LEVELS:
            # alpha_sensor = sample_noise_level()
            experience = collect_experience(baby, fan, tv, round_folder)
            # step_count = step_count + len(experience)
            data.appends(experience)

        # file_path = os.path.join(EXPERIENCE_DIR, f'experiences_{episode_id:03d}.pkl')
        # with open(file_path, 'wb') as f:
        #     pickle.dump(data, f)
        # print(f"Experiences saved to {file_path}")
        data.clear()

    print('Data collection completed.')
    scene_manager.close_session()


if __name__ == "__main__":
    main()