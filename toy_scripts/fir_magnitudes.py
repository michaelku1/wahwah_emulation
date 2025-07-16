"""
implementation from DDSP, Engel
"""


sample_rate = 44100


fir_filter = ddsp.effects.FIRFilter(scale_fn=None)

# Make up some oscillating gaussians.
n_seconds = audio.size / sample_rate
frame_rate = 100  # Hz

n_frames = int(n_seconds * frame_rate)
n_samples = int(n_frames * sample_rate / frame_rate)

audio_trimmed = audio[:, :n_samples]

n_frequencies = 1000
frequencies = np.linspace(0, sample_rate / 2.0, n_frequencies)

lfo_rate = 0.5  # Hz
n_cycles = n_seconds * lfo_rate
center_frequency = 1000 + 500 * np.sin(np.linspace(0, 2.0*np.pi*n_cycles, n_frames))
width = 500.0
gauss = lambda x, mu: 2.0 * np.pi * width**-2.0 * np.exp(- ((x - mu) / width)**2.0)


# Actually make the magnitudes.
magnitudes = np.array([gauss(frequencies, cf) for cf in center_frequency])
magnitudes = magnitudes[np.newaxis, ...]
magnitudes /= magnitudes.max(axis=-1, keepdims=True)

# Filter.
audio_out = fir_filter(audio_trimmed, magnitudes)

# Listen.
play(audio_out)
specplot(audio_out)
_ = plt.matshow(np.rot90(magnitudes[0]), aspect='auto')
plt.title('Frequency Response')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.xticks([])
_ = plt.yticks([])