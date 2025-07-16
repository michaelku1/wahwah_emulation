# import torch, torchaudio

# index = 10313
# n_samples = 2*44100
# silence_fraction_allowed = 0.2
# silence_threshold_energy = 4e-6

# # file_path_dry = '/Users/michael/Desktop/guitar data/EGDB subset/one sample dry/234.wav'
# file_path_wet = '/Users/michael/Desktop/ddsp_code/basics/fx_data/NA_WahFilter_7.5_Power_True_Bypass_False/234.wav'

# # audio_chunk, sr = torchaudio.load(file_path_dry, frame_offset=index, num_frames=n_samples,)
# audio_chunk, sr = torchaudio.load(file_path_wet, frame_offset=index, num_frames=n_samples,)

# window_size = int(silence_fraction_allowed * n_samples)
# hop_len = window_size // 4
# energy = audio_chunk ** 2
# unfolded = energy.unfold(dimension=-1, size=window_size, step=hop_len)
# mean_energies = torch.mean(unfolded, dim=-1, keepdim=False)
# n_silent = (mean_energies < silence_threshold_energy).sum().item()

# print(mean_energies)
# print(n_silent)
# print(n_silent > 0)

# # torchaudio.save("test_audio_dry.wav", audio_chunk , sr, backend='soundfile')
# torchaudio.save("test_audio_wet.wav", audio_chunk , sr, backend='soundfile')



