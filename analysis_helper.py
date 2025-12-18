import numpy as np
from scipy.signal import welch

# --- CONFIGURATION & CONSTANTS ---
SF = 256  # Muse Sampling Frequency

# Band Definitions
BANDS = {
    'Delta(0.5-4Hz)': (0.5, 4),
    'Theta(4-8Hz)': (4, 8),
    'Alpha(8-13Hz)': (8, 13),
    'Beta(13-30Hz)': (13, 30),
    'Gamma(30-45Hz)': (30, 45)
}
BAND_NAMES = list(BANDS.keys())
WAVE_COLORS = ['#CC0000', '#9933CC', '#0099CC', '#669900', '#FF8A00']

def calculate_band_powers(eeg_data, sf=SF, left_right_coherence=False):
    """
    Performs FFT on the raw EEG data to get the power of Delta, Theta, etc.
    Input: (Samples x Sensors) - numpy array
    Output: List of 5 values (one per band) representing the 'Score' for that band.
    """
    # Check shape. If it's (Sensors, Samples), transpose it.
    # We expect (Samples, Sensors) normally from the buffer.
    # But welch expects (..., n_samples) ideally if we want to iterate easily, OR we can stick to our previous logic.
    
    # Actually, the previous implementation in live_graph.py took (Samples, 4) and did eeg_data.T
    # Let's handle generic input safely.
    
    data = np.array(eeg_data)
    if data.shape[0] < data.shape[1] and data.shape[0] <= 4:
        # It looks like (Sensors, Samples), let's keep it as is for Welch?
        # Welch expects axis=-1 by default.
        pass
    else:
        # It looks like (Samples, Sensors), transpose to (Sensors, Samples)
        data = data.T
        
    num_sensors = data.shape[0]
    num_samples = data.shape[1]
    
    if num_samples == 0:
        return [0]*5

    # Calculate Power Spectral Density (PSD) using Welch's method
    nperseg = min(num_samples, sf) 
    freqs, psd = welch(data, sf, nperseg=nperseg)
    
    # psd shape is (Sensors, Frequency Bins)
    
    band_powers = []
    for band_name in BAND_NAMES:
        low, high = BANDS[band_name]
        # Find frequencies in this band
        idx = np.logical_and(freqs >= low, freqs <= high)
        
        if left_right_coherence and num_sensors >= 4:
            # Left (TP9, AF7) vs Right (AF8, TP10)
            # Assuming standard order: TP9, AF7, AF8, TP10
            # Indices: 0, 1 vs 2, 3
            left_power = np.mean(psd[0:2, idx])
            right_power = np.mean(psd[2:4, idx])
            power = left_power - right_power
        else:
            # Average of all available sensors in this chunk
            power = np.mean(psd[:, idx])
        
        # Convert to Log scale (Bels)
        # Adding 1e-6 to avoid log(0)
        value = np.log10(power + 1e-6)
        band_powers.append(value)
        
    return band_powers
