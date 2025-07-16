# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import spectrogram
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(x1, x2):

    # Load audio files
    audio1, sr1 = librosa.load(x1)
    audio2, sr2 = librosa.load(x2)

    # Resample if necessary
    if sr1 != sr2:
        audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        sr = sr1
    else:
        sr = sr1

    # Compute spectrograms
    n_fft = 2048
    hop_length = 512
    spec1 = librosa.stft(audio1, n_fft=n_fft, hop_length=hop_length)
    spec2 = librosa.stft(audio2, n_fft=n_fft, hop_length=hop_length)

    # Convert to decibel scale
    spec1_db = librosa.amplitude_to_db(np.abs(spec1), ref=np.max)
    spec2_db = librosa.amplitude_to_db(np.abs(spec2), ref=np.max)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot spectrogram 1
    img1 = librosa.display.specshow(spec1_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax1, cmap='viridis')
    ax1.set_title('Spectrogram 1')
    fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

    # Plot spectrogram 2
    img2 = librosa.display.specshow(spec2_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax2, cmap='viridis')
    ax2.set_title('Spectrogram 2')
    fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

    # Plot difference spectrogram
    spec_diff = spec1_db - spec2_db
    img3 = librosa.display.specshow(spec_diff, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax3, cmap='coolwarm')
    ax3.set_title('Difference (Spec1 - Spec2)')
    fig.colorbar(img3, ax=ax3, format='%+2.0f dB')

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

# def plot_spectrogram(x1, x2, sample_length, srate):

#     if not isinstance(x1, np.ndarray):
#         x1 = x1.numpy()
    
#     if not isinstance(x1, np.ndarray):
#         x2 = x2.numpy()

#     x1 = x1[:sample_length]
#     x2 = x2[:sample_length]

#     f1, t1, Sxx1 = spectrogram(x1, srate)
#     f2, t2, Sxx2 = spectrogram(x2, srate)

#     fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

#     breakpoint()
#     axs[0].pcolormesh(t1, f1, 10 * np.log10(Sxx1), shading='gouraud')
#     axs[0].set_title('Signal 1')
#     axs[0].set_ylabel('Frequency [Hz]')
#     axs[0].set_xlabel('Time [s]')

#     axs[1].pcolormesh(t2, f2, 10 * np.log10(Sxx2), shading='gouraud')
#     axs[1].set_title('Signal 2')
#     axs[1].set_xlabel('Time [s]')

#     plt.tight_layout()
#     plt.show()


    # plt.specgram(x1.numpy(), Fs=srate,)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [s]')
    # plt.title('Zoomed Low-Frequency Spectrogram')
    # plt.colorbar(label='Intensity [dB]')
    # plt.ylim(0, 20000)  # Zoom into 0–1000 Hz
    # plt.show()


if __name__ == "__main__":

    test_audio_1 = '/Users/michael/Desktop/wah dataset/bias fx auto wah/autowah_test_1/test1/one sample dry/233.wav'
    test_audio_2 = '/Users/michael/Desktop/wah dataset/bias fx auto wah/autowah_test_2/test1/one sample dry/233.wav'
    
    plot_spectrogram(test_audio_1, test_audio_2)