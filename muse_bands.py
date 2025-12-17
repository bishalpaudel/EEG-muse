import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch
import time

# --- BAND SETTINGS ---
# Standard EEG bands
BANDS = {
    "Delta": [0.5, 4],
    "Theta": [4, 8],
    "Alpha": [8, 12],
    "Beta":  [12, 30],
    "Gamma": [30, 100]
}

# Buffer settings
BUFFER_LENGTH = 5  # Seconds of data to analyze (longer = smoother, more lag)
EPOCH_LENGTH = 1   # Seconds of data to update per step
OVERLAP_LENGTH = 0.8 # Overlap between epochs
sf = 256 # Sampling frequency (Muse default is usually 256Hz)

def get_band_power(data, sf, method='welch'):
    """Calculate the average power of the signal in specific frequency bands."""
    band_powers = {}
    
    # Calculate Power Spectral Density (PSD) using Welch's method
    freqs, psd = welch(data, sf, nperseg=sf*2)

    for band_name, (low, high) in BANDS.items():
        # Find indices corresponding to the frequency band
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Average power in this band
        band_power = np.mean(psd[idx_band])
        band_powers[band_name] = band_power
        
    return band_powers

def main():
    print("Looking for an EEG stream...")
    from stream_helper import resolve_stream; streams = resolve_stream(timeout=10)
    if len(streams) == 0:
        raise RuntimeError("Can't find EEG stream. Did you start BlueMuse or muselsl stream?")

    print("Start acquiring data...")
    inlet = StreamInlet(streams[0])
    
    # Initialize buffer
    buffer = np.zeros((int(sf * BUFFER_LENGTH), 4)) # 4 channels for Muse

    try:
        while True:
            # Pull a chunk of data
            chunk, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=int(sf * EPOCH_LENGTH))
            
            if chunk:
                chunk_np = np.array(chunk)
                
                # --- THE FIX IS HERE ---
                # If the data has more than 4 channels, take only the first 4
                if chunk_np.shape[1] > 4:
                    chunk_np = chunk_np[:, :4]
                # -----------------------

                # Roll buffer to make space for new data
                buffer = np.roll(buffer, -len(chunk_np), axis=0)
                buffer[-len(chunk_np):, :] = chunk_np

                # Average all 4 channels for a global brain view
                data_for_analysis = np.mean(buffer, axis=1)

                # Compute Band Powers
                powers = get_band_power(data_for_analysis, sf)

                # --- VISUALIZATION (Text-Based Bar Chart) ---
                print("\033[H\033[J") # Clear screen
                print("--- REAL-TIME BRAINWAVES ---")
                
                for band, value in powers.items():
                    # Log scale for better visualization
                    scaled_val = int(np.log1p(value) * 5) 
                    bar = "█" * scaled_val
                    print(f"{band:5} | {bar} ({value:.2f})")
                
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()
