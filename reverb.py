import numpy as np
import matplotlib.pyplot as plt

SOUND_SPEED = 343


def reverb_ir(time: float, sample_rate: int, speaker_pos: np.ndarray, listener_pos: np.ndarray,
              room_shape: tuple[int], reflect: float, include_direct: bool = True):
    room_shape = np.asarray(room_shape, dtype=float)
    speaker_pos = np.asarray(speaker_pos, dtype=float)
    listener_pos = np.asarray(listener_pos, dtype=float)

    num_samples = int(time * sample_rate)
    result = np.zeros(num_samples)

    num_bounces = int(time * SOUND_SPEED / min(room_shape))

    # ---- 1. Build reflection index grid ----
    n = np.arange(-num_bounces, num_bounces + 1)
    nx, ny, nz = np.meshgrid(n, n, n, indexing="ij")

    # Flatten
    nx = nx.ravel()
    ny = ny.ravel()
    nz = nz.ravel()

    # ---- 2. Compute flipped speaker positions ----
    # Parity mask (True if odd)
    flip_x = (nx % 2 != 0)
    flip_y = (ny % 2 != 0)
    flip_z = (nz % 2 != 0)

    # Start with tiled base positions
    reflected = np.zeros((len(nx), 3))

    reflected[:, 0] = np.where(
        flip_x,
        room_shape[0] - speaker_pos[0],
        speaker_pos[0]
    ) + nx * room_shape[0]

    reflected[:, 1] = np.where(
        flip_y,
        room_shape[1] - speaker_pos[1],
        speaker_pos[1]
    ) + ny * room_shape[1]

    reflected[:, 2] = np.where(
        flip_z,
        room_shape[2] - speaker_pos[2],
        speaker_pos[2]
    ) + nz * room_shape[2]

    # ---- 3. Distances ----
    diff = reflected - listener_pos
    distances = np.linalg.norm(diff, axis=1)

    # Avoid division by zero
    valid = distances > 0
    distances = distances[valid]
    nx = nx[valid]
    ny = ny[valid]
    nz = nz[valid]

    # ---- 4. Sample indices ----
    sample_idx = (distances / SOUND_SPEED * sample_rate).astype(int)
    valid_time = sample_idx < num_samples - 1

    sample_idx = sample_idx[valid_time]
    distances = distances[valid_time]
    nx = nx[valid_time]
    ny = ny[valid_time]
    nz = nz[valid_time]

    # ---- 5. Amplitudes ----
    bounce_order = np.abs(nx) + np.abs(ny) + np.abs(nz)
    amplitudes = (1.0 / distances) * (reflect ** bounce_order)

    # ---- 6. Accumulate (linear pressure sum) ----
    np.add.at(result, sample_idx, amplitudes)

    # ---- 7. Direct path handling ----
    direct_distance = np.linalg.norm(listener_pos - speaker_pos)
    direct_idx = int(direct_distance / SOUND_SPEED * sample_rate)
    direct_amp = 1.0 / direct_distance

    if not include_direct and direct_idx < num_samples:
        result[direct_idx] -= direct_amp

    # Truncate
    result = result[direct_idx:]

    # Normalize
    rms = np.sqrt(np.sum(result ** 2))
    if rms > 0:
        result /= rms

    return result, direct_amp / rms if rms > 0 else 0.0


if __name__ == "__main__":
    time = 1
    sample_rate = 44100
    speaker_pos = np.array([1.5, 0.9, 1.4])
    listener_pos = np.array([3, 2.7, 1.2])
    room_shape = (6, 5, 3)
    #room_shape = (20, 15, 10)
    reflect = 0.7
    pad_samples = 40000

    ir, _ = reverb_ir(time, sample_rate, speaker_pos, listener_pos, room_shape, reflect, include_direct=True)
    plt.plot(np.linspace(0, time, len(ir)), ir)
    plt.xlim(-0.002, 0.06)
    #plt.ylim(0, 0.3)

    if len(ir) < pad_samples:
        ir = np.pad(ir, (0, pad_samples - len(ir)))
    fr = np.fft.rfft(ir)
    
    #plt.semilogx(np.fft.rfftfreq(len(ir), 1 / sample_rate), 20 * np.log10(np.abs(fr)))
    #plt.xlim(20, 20000)
    #plt.ylim(-30, 12)
    plt.show()
