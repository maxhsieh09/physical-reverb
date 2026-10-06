import numpy as np
from scipy.signal import convolve
import soundfile
import sys

import reverb

delay = 0.19e-3
volume = 1

filename = sys.argv[1]
raw_audio, sample_rate = soundfile.read(filename)
delay_samples = int(delay * sample_rate)

audio = np.zeros_like(raw_audio)
delayed_audio = np.zeros_like(raw_audio)
if delay_samples > 0:
    delayed_audio[delay_samples:, :] = raw_audio[:-delay_samples, :]
else:
    delayed_audio = raw_audio.copy()

audio[:, 0] = raw_audio[:, 0] * (3 / 4) + delayed_audio[:, 1] * (1 / 4)
audio[:, 1] = raw_audio[:, 1] * (3 / 4) + delayed_audio[:, 0] * (1 / 4)

# Reverb
time = 1.5
speaker_pos = np.array([1.5, 0.9, 1.4])
listener_pos = np.array([3, 2.7, 1.2])
room_shape = (6, 5, 3)
#room_shape = (20, 15, 10)
reflect = 0.9
ear_shift = np.array([0.09, 0, 0])

#mono_audio = (raw_audio[:, 0] + raw_audio[:, 1]) / 2

left_ir, direct_amplitude = reverb.reverb_ir(time, sample_rate, speaker_pos, listener_pos - ear_shift, room_shape, reflect, include_direct=False)
right_ir, _ = reverb.reverb_ir(time, sample_rate, speaker_pos, listener_pos + ear_shift, room_shape, reflect, include_direct=False)

audio *= direct_amplitude
audio[:, 0] += convolve(raw_audio[:, 0], left_ir, mode="full")[:len(raw_audio)] / 2
audio[:, 1] += convolve(raw_audio[:, 1], right_ir, mode="full")[:len(raw_audio)] / 2
audio[:, 0] += convolve(raw_audio[:, 1], right_ir, mode="full")[:len(raw_audio)] / 2
audio[:, 1] += convolve(raw_audio[:, 0], left_ir, mode="full")[:len(raw_audio)] / 2
audio *= volume

name, ext = filename.split(".")
name += "_crossfeed"
soundfile.write(name + ".flac", audio, sample_rate)
