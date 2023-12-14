from config import config
from env.v0d0 import Env
from policy.v0d0 import Model
import torch
import time
import cv2
import IPython.display as ipd
from scipy.io.wavfile import write
import numpy as np

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def run():
    env = Env(config)
    obs = env.reset()
    camera, audio = obs["camera"], obs["audio"]
    # display camera
    # cv2.imshow("RGBA", transform_rgb_bgr(camera[0]))
    # play audio: audio is a numpy array. play it
    # ipd.Audio(audio[0], rate=48000)

    _return = 0
    step = 0
    write(f"res/output_{step}.wav", 48000, audio[0].T.astype(np.float32))

    while True:
        step += 1
        # keystroke = cv2.waitKey(0)
        keystroke = ord(input())

        if keystroke == ord(FORWARD_KEY):
            action = 0
        elif keystroke == ord(LEFT_KEY):
            action = 1
        elif keystroke == ord(RIGHT_KEY):
            action = 2
        else:
            continue

        a = {
            "rl_pred": [action],
            "lstm_h": None,
            "lstm_c": None,
        }

        obs, r, done, info = env.step(a)
        _return += r[0]

        action = "forward" if action == 0 else "left" if action == 1 else "right"
        print(f"action: {action}")
        print(f"reward: {r[0]}")
        print(f"distance to goal:", info["distance"][0])
        print(f"angle to goal:", info["angle"][0])
        print(f"return: {_return}")
        print("position:", env._sim.get_agent(0).get_state().position)

        camera, audio = obs["camera"], obs["audio"]
        write(f"res/output_{step}.wav", 48000, audio[0].T.astype(np.float32))
        print("output:", audio[0].shape)

        print(step)
        if done:
            break


if __name__ == "__main__":
    run()
