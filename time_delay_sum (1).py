import numpy as np
import soundfile as sf
import os
from scipy.signal import resample

# Microphone array geometry (meters)
mics = np.array([
    [-0.05, 0.00, 0.00],    # Mic1
    [0.05, 0.00, 0.00],     # Mic2
    [-0.08, 0.045, -0.04],  # Mic3
    [0.08, 0.045, -0.04],   # Mic4
])

# Source direction coordinates (meters)
source_pos = np.array([0, -0.06, 0])

# Speed of sound (m/s)
c = 343.0

# Input files and output path
input_files = [
    r"C:\Users\vikas saini\Downloads\Tr1.WAV",
    r"C:\Users\vikas saini\Downloads\Tr2.WAV",
    r"C:\Users\vikas saini\Downloads\Tr3.WAV",
    r"C:\Users\vikas saini\Downloads\Tr4.WAV",
]
output_path = os.path.join(os.path.dirname(input_files[0]), "beamformed_output.wAV")

def calculate_time_delays(mics, source, c):
    """Calculate time delays relative to first microphone"""
    distances = np.linalg.norm(mics - source, axis=1)
    return (distances - distances[0]) / c

def align_and_sum(signals, delays, fs):
    """Align signals using calculated delays and sum"""
    max_delay_samples = int(np.ceil(np.max(np.abs(delays)) * fs))
    padded_length = len(signals[0]) + 2 * max_delay_samples
    aligned = []
    
    for sig, delay in zip(signals, delays):
        # Convert delay to samples
        delay_samples = delay * fs
        # Fractional delay handling using resampling
        n = len(sig)
        new_length = int(np.round(n + delay_samples))
        resampled = resample(sig, new_length)
        
        # Trim/pad to match padded length
        if len(resampled) < padded_length:
            resampled = np.pad(resampled, (0, padded_length - len(resampled)))
        else:
            resampled = resampled[:padded_length]
            
        aligned.append(resampled)
    
    return np.sum(aligned, axis=0)

# Calculate time delays
delays = calculate_time_delays(mics, source_pos, c)

# Read input files
signals, fs = [], None
for f in input_files:
    data, sample_rate = sf.read(f)
    if fs is None:
        fs = sample_rate
    else:
        assert fs == sample_rate, "Sample rate mismatch"
    signals.append(data if data.ndim == 1 else data[:, 0])

# Process signals
output_signal = align_and_sum(signals, delays, fs)

# Normalize and save
output_signal /= np.max(np.abs(output_signal))
sf.write(output_path, output_signal, fs)

print(f"Beamformed audio saved to:\n{output_path}")

