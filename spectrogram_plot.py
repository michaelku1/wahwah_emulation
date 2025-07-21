from utils.plot_spectrogram import plot_spectrogram

gt_path = "./results/test_single/mono/600/gt/233_42228.wav"
wet_path = "./results/test_single/mono/600/wet/233_42228.wav"
plot_spectrogram(gt_path, wet_path)