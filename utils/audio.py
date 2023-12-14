import librosa
import numpy as np
from scipy.io.wavfile import write


class ChunkedAudio:
    """
    taking in the time and one step time, output the audio chunk.
    the audio repeats with 1s interval, if time inputed is larger than the audio length, the audio will be repeated.
    Return:
        chunk: np.array, shape=(step_time * sample_rate, 2)
    """

    def __init__(self, audio_dir, sample_rate=48000, step_time=0.25, time=0):
        self.sample_rate = sample_rate
        self.audio, _ = librosa.load(audio_dir, sr=sample_rate)
        self.step_time = step_time
        self.time = time  # the clock for this class
        self.len = len(self.audio)
        self.round = 0

    def __call__(self, time, step):
        """
        input time, output the audio chunk
        """
        if time < self.time:
            raise ValueError("time should be larger than self.time")
        else:
            self.time = time
            chunk = self.audio[
                (int(self.time * self.sample_rate) - self.round * self.len) : (
                    int((self.time + self.step_time) * self.sample_rate)
                    - self.round * self.len
                )
            ]
            if len(chunk) < self.step_time * self.sample_rate:
                chunk = np.concatenate(
                    [
                        chunk,
                        self.audio[
                            : (int(self.step_time * self.sample_rate) - len(chunk))
                        ],
                    ]
                )
                self.round += 1

            write(f"res/origin_{step}.wav", 48000, chunk.T.astype(np.float32))
            print("original:", chunk.shape)
            return chunk

    def reset(self):
        """
        reset the audio chunk to the beginning
        """
        self.time = 0
