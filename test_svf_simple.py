#!/usr/bin/env python3
"""
Simple test script for StateVariableFilter using audio data from EGDB_DI dataset.

This script demonstrates basic usage of the StateVariableFilter with a 2-second audio chunk.
"""

import torch
import torchaudio
import numpy as np
import os
import sys
from pathlib import Path

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent / "models"))

def load_audio_chunk_simple(file_path, chunk_duration=2.0, sample_rate=44100):
    """
    Load a 2-second chunk from an audio file using torchaudio.
    """
    try:
        # Load audio using torchaudio
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio = resampler(audio)
        
        # Extract 2-second chunk
        chunk_samples = int(chunk_duration * sample_rate)
        if audio.shape[1] < chunk_samples:
            # Pad with zeros if audio is shorter
            padding = chunk_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        else:
            # Take the middle chunk
            start_sample = (audio.shape[1] - chunk_samples) // 2
            audio = audio[:, start_sample:start_sample + chunk_samples]
        
        # Normalize
        audio = audio / torch.max(torch.abs(audio))
        
        # Add batch dimension: (batch, channels, samples)
        audio = audio.unsqueeze(0)  # (1, 1, samples)
        
        return audio, sample_rate
        
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def test_basic_svf():
    """
    Basic test of StateVariableFilter functionality.
    """
    print("=== Basic StateVariableFilter Test ===")
    
    # Find an audio file
    audio_file = "fx_data/NA_wah_75/mono/NA_WahFilter_7.5_Power_True_Bypass_False/233.wav"
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return
    
    print(f"Using audio file: {audio_file}")
    
    # Load 2-second audio chunk
    audio_chunk, sample_rate = load_audio_chunk_simple(audio_file, chunk_duration=2.0)
    
    if audio_chunk is None:
        print("Failed to load audio file.")
        return
    
    print(f"Loaded audio chunk: {audio_chunk.shape}, Sample rate: {sample_rate} Hz")
    
    try:
        # Import the StateVariableFilter
        from processors.svf_biquads import StateVariableFilter
        
        # Create SVF filter
        svf_filter = StateVariableFilter(num_filters=1, backend="fsm")
        
        # Create simple filter parameters for low-pass filter
        params = {
            "twoR": torch.tensor([0.5], dtype=torch.float32),
            "G": torch.tensor([0.3], dtype=torch.float32),
            "c_hp": torch.tensor([0.0], dtype=torch.float32),
            "c_bp": torch.tensor([0.0], dtype=torch.float32),
            "c_lp": torch.tensor([1.0], dtype=torch.float32),
        }
        
        print("Applying StateVariableFilter...")
        
        # Apply filter
        filtered_audio = svf_filter(audio_chunk, **params)
        
        print(f"Filter applied successfully!")
        print(f"Input shape: {audio_chunk.shape}")
        print(f"Output shape: {filtered_audio.shape}")
        
        # Save results
        output_dir = "test_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original
        original_path = os.path.join(output_dir, "svf_test_original.wav")
        torchaudio.save(original_path, audio_chunk.squeeze(0), sample_rate)
        
        # Save filtered
        filtered_path = os.path.join(output_dir, "svf_test_filtered.wav")
        torchaudio.save(filtered_path, filtered_audio.squeeze(0), sample_rate)
        
        print(f"Saved audio files to {output_dir}/")
        print(f"  Original: {original_path}")
        print(f"  Filtered: {filtered_path}")
        
        # Print some statistics
        print(f"\nAudio Statistics:")
        print(f"  Original - Max: {torch.max(torch.abs(audio_chunk)):.4f}, RMS: {torch.sqrt(torch.mean(audio_chunk**2)):.4f}")
        print(f"  Filtered - Max: {torch.max(torch.abs(filtered_audio)):.4f}, RMS: {torch.sqrt(torch.mean(filtered_audio**2)):.4f}")
        
    except Exception as e:
        print(f"Error during filter application: {e}")
        import traceback
        traceback.print_exc()

def test_multiple_filters():
    """
    Test different filter configurations.
    """
    print("\n=== Multiple Filter Configurations Test ===")
    
    # Load audio
    audio_file = "fx_data/NA_wah_75/mono/NA_WahFilter_7.5_Power_True_Bypass_False/233.wav"
    if not os.path.exists(audio_file):
        print("Audio file not found.")
        return
    
    audio_chunk, sample_rate = load_audio_chunk_simple(audio_file, chunk_duration=2.0)
    if audio_chunk is None:
        return
    
    try:
        from processors.svf_biquads import StateVariableFilter
        
        # Create SVF filter
        svf_filter = StateVariableFilter(num_filters=1, backend="fsm")
        
        # Test different filter configurations
        filter_configs = [
            ("lowpass", {
                "twoR": torch.tensor([0.5], dtype=torch.float32),
                "G": torch.tensor([0.3], dtype=torch.float32),
                "c_hp": torch.tensor([0.0], dtype=torch.float32),
                "c_bp": torch.tensor([0.0], dtype=torch.float32),
                "c_lp": torch.tensor([1.0], dtype=torch.float32),
            }),
            ("highpass", {
                "twoR": torch.tensor([0.5], dtype=torch.float32),
                "G": torch.tensor([0.3], dtype=torch.float32),
                "c_hp": torch.tensor([1.0], dtype=torch.float32),
                "c_bp": torch.tensor([0.0], dtype=torch.float32),
                "c_lp": torch.tensor([0.0], dtype=torch.float32),
            }),
            ("bandpass", {
                "twoR": torch.tensor([0.5], dtype=torch.float32),
                "G": torch.tensor([0.3], dtype=torch.float32),
                "c_hp": torch.tensor([0.0], dtype=torch.float32),
                "c_bp": torch.tensor([1.0], dtype=torch.float32),
                "c_lp": torch.tensor([0.0], dtype=torch.float32),
            }),
        ]
        
        output_dir = "test_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        for filter_name, params in filter_configs:
            print(f"\nTesting {filter_name} filter...")
            
            try:
                filtered_audio = svf_filter(audio_chunk, **params)
                
                # Save result
                filtered_path = os.path.join(output_dir, f"svf_{filter_name}.wav")
                torchaudio.save(filtered_path, filtered_audio.squeeze(0), sample_rate)
                
                print(f"✓ {filter_name} filter applied successfully")
                print(f"  Saved to: {filtered_path}")
                
            except Exception as e:
                print(f"✗ Error applying {filter_name} filter: {e}")
        
    except Exception as e:
        print(f"Error during multiple filter test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run basic test
    test_basic_svf()
    
    # Run multiple filter test
    test_multiple_filters()
    
    print("\n=== Test completed ===") 