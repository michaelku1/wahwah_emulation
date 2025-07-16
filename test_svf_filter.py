#!/usr/bin/env python3
"""
Test script for StateVariableFilter using audio data from EGDB_DI dataset.

This script demonstrates how to use the StateVariableFilter with different configurations:
- Low-pass filter
- High-pass filter  
- Band-pass filter
- Notch filter
- Wah-wah effect simulation

The script loads a 2-second chunk from an audio file and applies various filter configurations.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import sys
from pathlib import Path

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent / "models"))

from processors.svf_biquads import StateVariableFilter

def load_audio_chunk(file_path, chunk_duration=2.0, sample_rate=44100):
    """
    Load a 2-second chunk from an audio file.
    
    Args:
        file_path: Path to the audio file
        chunk_duration: Duration of the chunk in seconds
        sample_rate: Target sample rate
        
    Returns:
        torch.Tensor: Audio chunk of shape (1, samples)
    """
    try:
        # Load audio using soundfile for better compatibility
        audio, sr = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if necessary
        if sr != sample_rate:
            # Simple resampling (for production, use proper resampling)
            ratio = sample_rate / sr
            new_length = int(len(audio) * ratio)
            audio = np.interp(np.linspace(0, len(audio), new_length), 
                            np.arange(len(audio)), audio)
        
        # Extract 2-second chunk
        chunk_samples = int(chunk_duration * sample_rate)
        if len(audio) < chunk_samples:
            # Pad with zeros if audio is shorter
            audio = np.pad(audio, (0, chunk_samples - len(audio)), 'constant')
        else:
            # Take the middle chunk
            start_sample = (len(audio) - chunk_samples) // 2
            audio = audio[start_sample:start_sample + chunk_samples]
        
        # Convert to torch tensor and normalize
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
        
        # Add batch and channel dimensions: (batch, channels, samples)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        
        return audio_tensor, sample_rate
        
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def create_filter_parameters(filter_type, num_filters=1):
    """
    Create filter parameters for different filter types.
    
    Args:
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'notch', 'wah')
        num_filters: Number of filters to create
        
    Returns:
        dict: Dictionary containing filter parameters
    """
    if filter_type == "lowpass":
        # Low-pass filter: emphasize low frequencies
        return {
            "twoR": torch.tensor([0.5] * num_filters, dtype=torch.float32),
            "G": torch.tensor([0.3] * num_filters, dtype=torch.float32),
            "c_hp": torch.tensor([0.0] * num_filters, dtype=torch.float32),
            "c_bp": torch.tensor([0.0] * num_filters, dtype=torch.float32),
            "c_lp": torch.tensor([1.0] * num_filters, dtype=torch.float32),
        }
    
    elif filter_type == "highpass":
        # High-pass filter: emphasize high frequencies
        return {
            "twoR": torch.tensor([0.5] * num_filters, dtype=torch.float32),
            "G": torch.tensor([0.3] * num_filters, dtype=torch.float32),
            "c_hp": torch.tensor([1.0] * num_filters, dtype=torch.float32),
            "c_bp": torch.tensor([0.0] * num_filters, dtype=torch.float32),
            "c_lp": torch.tensor([0.0] * num_filters, dtype=torch.float32),
        }
    
    elif filter_type == "bandpass":
        # Band-pass filter: emphasize mid frequencies
        return {
            "twoR": torch.tensor([0.5] * num_filters, dtype=torch.float32),
            "G": torch.tensor([0.3] * num_filters, dtype=torch.float32),
            "c_hp": torch.tensor([0.0] * num_filters, dtype=torch.float32),
            "c_bp": torch.tensor([1.0] * num_filters, dtype=torch.float32),
            "c_lp": torch.tensor([0.0] * num_filters, dtype=torch.float32),
        }
    
    elif filter_type == "notch":
        # Notch filter: attenuate mid frequencies
        return {
            "twoR": torch.tensor([0.5] * num_filters, dtype=torch.float32),
            "G": torch.tensor([0.3] * num_filters, dtype=torch.float32),
            "c_hp": torch.tensor([0.5] * num_filters, dtype=torch.float32),
            "c_bp": torch.tensor([0.0] * num_filters, dtype=torch.float32),
            "c_lp": torch.tensor([0.5] * num_filters, dtype=torch.float32),
        }
    
    elif filter_type == "wah":
        # Wah-wah effect: varying band-pass filter
        return {
            "twoR": torch.tensor([0.3] * num_filters, dtype=torch.float32),
            "G": torch.tensor([0.6] * num_filters, dtype=torch.float32),
            "c_hp": torch.tensor([0.0] * num_filters, dtype=torch.float32),
            "c_bp": torch.tensor([1.0] * num_filters, dtype=torch.float32),
            "c_lp": torch.tensor([0.0] * num_filters, dtype=torch.float32),
        }
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

def plot_audio_comparison(original, filtered, title, sample_rate=44100):
    """
    Plot original vs filtered audio signals.
    
    Args:
        original: Original audio tensor
        filtered: Filtered audio tensor
        title: Plot title
        sample_rate: Audio sample rate
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # Time domain
    time = torch.arange(original.shape[-1]) / sample_rate
    
    ax1.plot(time, original.squeeze().numpy(), label='Original', alpha=0.7)
    ax1.plot(time, filtered.squeeze().numpy(), label='Filtered', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'{title} - Time Domain')
    ax1.legend()
    ax1.grid(True)
    
    # Frequency domain - Original
    fft_orig = torch.fft.rfft(original.squeeze())
    freqs = torch.fft.rfftfreq(original.shape[-1], 1/sample_rate)
    ax2.semilogx(freqs, 20 * torch.log10(torch.abs(fft_orig) + 1e-10))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Original - Frequency Domain')
    ax2.grid(True)
    ax2.set_xlim(20, sample_rate//2)
    
    # Frequency domain - Filtered
    fft_filt = torch.fft.rfft(filtered.squeeze())
    ax3.semilogx(freqs, 20 * torch.log10(torch.abs(fft_filt) + 1e-10))
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_title('Filtered - Frequency Domain')
    ax3.grid(True)
    ax3.set_xlim(20, sample_rate//2)
    
    plt.tight_layout()
    plt.show()

def save_audio_comparison(original, filtered, title, output_dir="test_outputs"):
    """
    Save original and filtered audio files.
    
    Args:
        original: Original audio tensor
        filtered: Filtered audio tensor
        title: Base name for the files
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original
    original_path = os.path.join(output_dir, f"{title}_original.wav")
    torchaudio.save(original_path, original, 44100)
    
    # Save filtered
    filtered_path = os.path.join(output_dir, f"{title}_filtered.wav")
    torchaudio.save(filtered_path, filtered, 44100)
    
    print(f"Saved audio files to {output_dir}/")
    print(f"  Original: {original_path}")
    print(f"  Filtered: {filtered_path}")

def test_svf_filter():
    """
    Main test function for StateVariableFilter.
    """
    print("=== StateVariableFilter Test Script ===")
    
    # Find an audio file from the EGDB_DI dataset
    audio_files = [
        "fx_data/NA_wah_75/mono/NA_WahFilter_7.5_Power_True_Bypass_False/233.wav",
        "fx_data/NA_wah_75/mono/NA_WahFilter_7.5_Power_True_Bypass_False/236.wav",
        "fx_data/NA_wah_75/mono/NA_WahFilter_7.5_Power_True_Bypass_False/238.wav",
        "fx_data/NA_wah_75/mono/NA_WahFilter_7.5_Power_True_Bypass_False/240.wav",
    ]
    
    # Find the first available audio file
    audio_file = None
    for file_path in audio_files:
        if os.path.exists(file_path):
            audio_file = file_path
            break
    
    if audio_file is None:
        print("No audio files found. Please check the file paths.")
        return
    
    print(f"Using audio file: {audio_file}")
    
    # Load 2-second audio chunk
    audio_chunk, sample_rate = load_audio_chunk(audio_file, chunk_duration=2.0)
    
    if audio_chunk is None:
        print("Failed to load audio file.")
        return
    
    print(f"Loaded audio chunk: {audio_chunk.shape}, Sample rate: {sample_rate} Hz")
    
    # Test different filter configurations
    filter_configs = [
        ("lowpass", "Low-Pass Filter"),
        ("highpass", "High-Pass Filter"),
        ("bandpass", "Band-Pass Filter"),
        ("notch", "Notch Filter"),
        ("wah", "Wah-Wah Effect"),
    ]
    
    # Create SVF filter
    svf_filter = StateVariableFilter(num_filters=1, backend="fsm")
    
    print("\nTesting different filter configurations...")
    
    for filter_type, filter_name in filter_configs:
        print(f"\n--- Testing {filter_name} ---")
        
        # Create filter parameters
        params = create_filter_parameters(filter_type)
        
        # Apply filter
        try:
            filtered_audio = svf_filter(audio_chunk, **params)
            
            # Plot comparison
            plot_audio_comparison(audio_chunk, filtered_audio, filter_name, sample_rate)
            
            # Save audio files
            save_audio_comparison(audio_chunk, filtered_audio, filter_type.lower().replace(" ", "_"))
            
            print(f"✓ {filter_name} applied successfully")
            
        except Exception as e:
            print(f"✗ Error applying {filter_name}: {e}")
    
    print("\n=== Test completed ===")

def test_parameter_sensitivity():
    """
    Test the sensitivity of filter parameters.
    """
    print("\n=== Parameter Sensitivity Test ===")
    
    # Load audio
    audio_file = "fx_data/NA_wah_75/mono/NA_WahFilter_7.5_Power_True_Bypass_False/233.wav"
    if not os.path.exists(audio_file):
        print("Audio file not found for sensitivity test.")
        return
    
    audio_chunk, sample_rate = load_audio_chunk(audio_file, chunk_duration=2.0)
    if audio_chunk is None:
        return
    
    # Create SVF filter
    svf_filter = StateVariableFilter(num_filters=1, backend="fsm")
    
    # Test different G values (affects cutoff frequency)
    g_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(len(g_values), 1, figsize=(12, 3*len(g_values)))
    if len(g_values) == 1:
        axes = [axes]
    
    for i, g_val in enumerate(g_values):
        params = {
            "twoR": torch.tensor([0.5], dtype=torch.float32),
            "G": torch.tensor([g_val], dtype=torch.float32),
            "c_hp": torch.tensor([0.0], dtype=torch.float32),
            "c_bp": torch.tensor([0.0], dtype=torch.float32),
            "c_lp": torch.tensor([1.0], dtype=torch.float32),
        }
        
        try:
            filtered_audio = svf_filter(audio_chunk, **params)
            
            # Plot frequency response
            fft_filt = torch.fft.rfft(filtered_audio.squeeze())
            freqs = torch.fft.rfftfreq(filtered_audio.shape[-1], 1/sample_rate)
            
            axes[i].semilogx(freqs, 20 * torch.log10(torch.abs(fft_filt) + 1e-10))
            axes[i].set_title(f'Low-pass filter with G={g_val}')
            axes[i].set_ylabel('Magnitude (dB)')
            axes[i].grid(True)
            axes[i].set_xlim(20, sample_rate//2)
            
        except Exception as e:
            print(f"Error with G={g_val}: {e}")
    
    axes[-1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run main test
    test_svf_filter()
    
    # Run parameter sensitivity test
    test_parameter_sensitivity() 