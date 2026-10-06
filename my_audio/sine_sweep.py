import numpy as np
import sounddevice as sd
import tkinter as tk

from fr import fr_interp

sample_rate = 44100
current_freq = 440.0
phase = 0.0  # to keep phase continuity

volume = 0.015

# This will be updated by GUI
def generate_wave_chunk(frames):
    global phase
    t = np.arange(frames) / sample_rate

    wave = np.sin(2 * np.pi * (current_freq * t + phase))

    phase += frames / sample_rate * current_freq
    phase = phase % 1.0  # keep phase in [0, 1)

    gain_db = min(-fr_interp(np.log10(current_freq / 2)), 14)
    gain = 10 ** (gain_db / 20)
    print(gain)

    return wave.astype(np.float32) * volume * gain

# Audio callback (runs in a separate thread)
def audio_callback(outdata, frames, time, status):
    if status:
        print("Stream warning:", status)
    wave = generate_wave_chunk(frames)
    outdata[:] = wave.reshape(-1, 1)

# Start/Stop stream control
stream = sd.OutputStream(channels=1, samplerate=sample_rate, callback=audio_callback)
is_playing = False

def start_audio():
    global is_playing
    if not is_playing:
        stream.start()
        is_playing = True

def stop_audio():
    global is_playing
    if is_playing:
        stream.stop()
        is_playing = False

def update_freq(val):
    global current_freq
    current_freq = 2 * 10 ** float(val)
    freq_label.config(text=f"Frequency: {current_freq:.1f}")

# GUI
root = tk.Tk()
root.geometry("300x200")
root.title("Smooth Sine Wave Generator")

freq_label = tk.Label(root)
freq_label.pack()
slider = tk.Scale(root, from_=1, to=4, resolution=0.005, orient="horizontal", command=update_freq)
slider.set(np.log10(current_freq / 2))
slider.pack(fill="x")

tk.Button(root, text="Start", command=start_audio).pack()
tk.Button(root, text="Stop", command=stop_audio).pack()

root.mainloop()

stream.close()
sd.wait()
