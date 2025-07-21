import torchaudio
from utils.plot_frequency_response import plot_measured_vs_gt_frequency_response
from utils.plot_spectrogram import plot_spectrogram
# dry_path = "./results/test_single/mono/600/dry/233_42228.wav"
# gt_path = "./results/test_single/mono/600/gt/233_42228.wav"
# wet_path = "./results/test_single/mono/600/wet/233_42228.wav"

# plot_measured_vs_gt_frequency_response(dry_path, wet_path, gt_path, save_path='Theoretical_vs_Measured_Frequency_Response.png')


#### spectrogram compare between dry and wet #### 
# dry_path = "./results/test_single/mono/600/dry/233_42228.wav"
# gt_path = "./results/test_single/mono/600/gt/233_42228.wav"
# wet_path = "./results/test_single/mono/600/wet/233_42228.wav"
# plot_spectrogram(dry_path, wet_path, save_path='Spectrogram_DRY_vs_WET.png')


#### spectrogram compare between dry and gt #### 
dry_path = "./results/test_single/mono/600/dry/233_42228.wav"
gt_path = "./results/test_single/mono/600/gt/233_42228.wav"
plot_spectrogram(dry_path, gt_path, save_path='Spectrogram_DRY_vs_GT.png')