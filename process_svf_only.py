import os
import torch
import torchaudio
import yaml
from tqdm import tqdm
from models.processors.svf_biquads import StateVariableFilter
from models.processors.dasp_processor import wah_wah
from data_preprocess.random_chunk_preprocess import RandomAudioChunkDataset
from utils.plot_spectrogram import plot_spectrogram
from inference import ParameterNetwork

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_path = "./train_wh_parametric_eq_models_07-18/preset_config1_preset_config1_epoch_33_600.pt"
mount_point = "/mnt/gestalt/home/mku666"
test_input_paths_clean = os.path.join(mount_point, "wah_emulation/EGDB_test_subset/mono/dry")
test_input_paths_gt = os.path.join(mount_point, "wah_emulation/EGDB_test_subset/mono/wet")
config_path = "configs/config1.yml"
preset_path = "configs/preset_config1.yml"
sr = 44100
n_samples = 88200
bs = 1

out_dir = "./results/mono/svf_only"
os.makedirs(out_dir, exist_ok=True)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
preset = load_yaml(preset_path)
preset_values = torch.tensor([v for v in preset.values()], dtype=torch.float32).to(DEVICE)
preset_values = preset_values.unsqueeze(0)

dataset = RandomAudioChunkDataset(test_input_paths_clean, test_input_paths_gt, n_samples, sr)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=2, pin_memory=True)

svf = StateVariableFilter(num_filters=1, device=DEVICE)
wahwah = wah_wah(sr, processor_fn=svf.svf_biquads_processor_forward)

checkpoint = torch.load(model_path, map_location=DEVICE)
parameter_net = ParameterNetwork(preset_values.shape[-1], wahwah.num_params).to(DEVICE)
parameter_net.load_state_dict(checkpoint['parameter_net_state_dict'])
parameter_net.eval()
wahwah.eval = lambda: None  # dummy for compatibility

def process_svf_only(x, p_hat):
    return wahwah.process_normalized(x.view(-1, x.shape[-1]), p_hat)

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing audio (SVF only)"):
        (dry, dry_path, dry_idx), _ = batch
        dry = dry.to(DEVICE)
        dry = dry.view(-1, 1, dry.shape[-1])
        p_hat = parameter_net(preset_values)
        wet = process_svf_only(dry, p_hat)
        base = os.path.splitext(os.path.basename(dry_path[0]))[0]
        idx = dry_idx[0].item() if hasattr(dry_idx[0], 'item') else dry_idx[0]
        dry_out = os.path.join(out_dir, f"dry_{base}_{idx}.wav")
        wet_out = os.path.join(out_dir, f"wet_{base}_{idx}.wav")
        torchaudio.save(dry_out, dry.cpu().squeeze(1), sr, backend="soundfile")
        torchaudio.save(wet_out, wet.cpu().squeeze(1), sr, backend="soundfile")
        plot_spectrogram(dry_out, wet_out, save_path=wet_out.replace('.wav', '_spec.png'), name1="Dry", name2="Wet")