#!/usr/bin/env python3
"""
Simple script to load and listen to audio data using the existing dataset structure.
"""

import os
import torch
import torchaudio
import numpy as np
from data_preprocess.random_chunk_preprocess import RandomAudioChunkDataset

def main():
    # Use the same paths as in your training script
    mount_point = "/mnt/gestalt/home/mku666"
    
    train_input_paths_clean = os.path.join(mount_point, "EGDB_DI")
    train_input_paths_wet = os.path.join(mount_point, "NA_Wah/NA_WahFilter_7.5_Power_True_Bypass_False")
    
    # Check if directories exist
    if not os.path.isdir(train_input_paths_clean):
        print(f"Dry audio directory not found: {train_input_paths_clean}")
        return
    
    if not os.path.isdir(train_input_paths_wet):
        print(f"Wet audio directory not found: {train_input_paths_wet}")
        return
    
    # Create output directory
    output_dir = "./audio_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset parameters (same as training script)
    n_samples = 88200  # 2 seconds at 44.1kHz
    sr = 44100
    silence_fraction_allowed = 0.8
    silence_threshold_energy = 1e-6
    n_retries = 1
    end_buffer_n_samples = 0
    should_peak_norm = False
    num_examples_per_epoch = 1000
    
    print("Creating dataset...")
    dataset = RandomAudioChunkDataset(
        train_input_paths_clean,
        train_input_paths_wet,
        n_samples,
        num_examples_per_epoch=num_examples_per_epoch,
        sr=sr,
        silence_fraction_allowed=silence_fraction_allowed,
        silence_threshold_energy=silence_threshold_energy,
        n_retries=n_retries,
        end_buffer_n_samples=end_buffer_n_samples,
        should_peak_norm=should_peak_norm,
        seed=12345
    )
    
    print(f"Dataset created successfully!")
    print(f"Number of files: {len(dataset.input_paths_dry)}")
    
    # Load and save a few samples
    num_samples_to_save = 3
    
    for i in range(num_samples_to_save):
        print(f"\nProcessing sample {i+1}/{num_samples_to_save}")
        
        try:
            # Get a sample from the dataset
            (dry_audio, dry_file_path, dry_start_idx), (wet_audio, wet_file_path, wet_start_idx) = dataset[i]
            
            # Extract the actual audio data (remove the batch dimension)
            dry_audio = dry_audio.squeeze(0)  # Remove batch dimension
            wet_audio = wet_audio.squeeze(0)  # Remove batch dimension

            # breakpoint()
            
            print(f"Dry file: {os.path.basename(dry_file_path[0])}")
            print(f"Wet file: {os.path.basename(wet_file_path[0])}")
            print(f"Start index: {dry_start_idx}")
            print(f"Audio shape: {dry_audio.shape}")
            print(f"Duration: {dry_audio.shape[-1] / sr:.2f} seconds")
            
            # Save audio files
            dry_output = os.path.join(output_dir, f"dry_sample_{i+1}.wav")
            wet_output = os.path.join(output_dir, f"wet_sample_{i+1}.wav")
            
            torchaudio.save(dry_output, dry_audio, sr, backend='soundfile')
            torchaudio.save(wet_output, wet_audio, sr, backend='soundfile')
            
            print(f"Saved dry audio to: {dry_output}")
            print(f"Saved wet audio to: {wet_output}")
            
            # Print instructions for playing
            print(f"To play the audio, run:")
            print(f"  aplay {dry_output}")
            print(f"  aplay {wet_output}")
            print(f"  # or use your preferred audio player")
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            continue
    
    print(f"\nAll samples saved to: {output_dir}")

if __name__ == "__main__":
    main() 