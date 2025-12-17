import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# --- SETTINGS (Matched to your VBA) ---
FILE_PATH = 'muse_recording.csv'  # CHANGE THIS to your actual CSV filename
GRAPH_LEFT_RIGHT_COHERENCE = False # True: (Left - Right), False: Average of all 4
AVERAGE_TRENDLINE_PERIOD = 60      # Moving average window size
IGNORE_BLINKS = True               # Ignore '/muse/elements/blink'
IGNORE_JAW_CLENCH = False          # Ignore '/muse/elements/jaw_clench'
HEADBAND_STATUS_LIMIT = 4          # 1=Good, 2=OK, 4=Bad. Filter data >= this.

# Colors used in Mind Monitor/VBA (Red, Purple, Blue, Green, Orange)
WAVE_COLORS = {
    'Delta': '#CC0000',
    'Theta': '#9933CC',
    'Alpha': '#0099CC',
    'Beta':  '#669900',
    'Gamma': '#FF8A00'
}

def process_and_graph(csv_path):
    print(f"Loading {csv_path}...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: File not found. Please check the FILE_PATH setting.")
        return

    # Convert TimeStamp to datetime objects
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # 2. Filter Data (Cleaning)
    # Remove rows where HeadBandOn is 0 (False)
    if 'HeadBandOn' in df.columns:
        df = df[df['HeadBandOn'] == 1]

    # Filter based on Signal Quality (HSI)
    # Mind Monitor HSI columns: HSI_TP9, HSI_AF7, HSI_AF8, HSI_TP10
    hsi_cols = ['HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10']
    if all(col in df.columns for col in hsi_cols):
        # Keep rows where ALL sensors are better (less than) the limit
        condition = (df[hsi_cols] < HEADBAND_STATUS_LIMIT).all(axis=1)
        df = df[condition]

    # Remove rows with NaN in Delta_TP9 (cleaning empty data)
    df = df.dropna(subset=['Delta_TP9'])

    # 3. Calculate Bands
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    sensors = ['TP9', 'AF7', 'AF8', 'TP10']
    
    for band in bands:
        # Construct column names (e.g., Delta_TP9, Delta_AF7...)
        cols = [f"{band}_{s}" for s in sensors]
        
        # Check if columns exist
        if not all(c in df.columns for c in cols):
            continue

        if GRAPH_LEFT_RIGHT_COHERENCE:
            # (Left Average) - (Right Average)
            # Left: TP9, AF7 | Right: AF8, TP10
            left = df[[f"{band}_TP9", f"{band}_AF7"]].mean(axis=1)
            right = df[[f"{band}_AF8", f"{band}_TP10"]].mean(axis=1)
            df[f"{band}_Calc"] = left - right
        else:
            # Average of all 4 sensors
            df[f"{band}_Calc"] = df[cols].mean(axis=1)

        # Calculate Trendline (Moving Average)
        df[f"{band}_Trend"] = df[f"{band}_Calc"].rolling(window=AVERAGE_TRENDLINE_PERIOD).mean()

    # 4. Extract Events (Blinks, Jaw Clench, etc.)
    events = []
    if 'Elements' in df.columns:
        # Filter non-empty elements
        event_rows = df[df['Elements'].notna()]
        
        for idx, row in event_rows.iterrows():
            element_text = row['Elements']
            
            # Clean text (remove /muse/elements/ prefix)
            if element_text.startswith('/muse/elements/'):
                element_text = element_text.replace('/muse/elements/', '')
            
            # Apply filters
            if IGNORE_BLINKS and 'blink' in element_text:
                continue
            if IGNORE_JAW_CLENCH and 'jaw_clench' in element_text:
                continue
                
            events.append((row['TimeStamp'], element_text))

    # 5. Plotting
    print("Generating Graph...")
    fig, ax = plt.subplots(figsize=(12, 7))

    for band in bands:
        if f"{band}_Calc" not in df.columns: continue
        
        color = WAVE_COLORS[band]
        
        # Plot Raw Data (Thin, Transparent)
        ax.plot(df['TimeStamp'], df[f"{band}_Calc"], 
                color=color, alpha=0.15, linewidth=0.5, label='_nolegend_')
        
        # Plot Trendline (Thick, Solid)
        ax.plot(df['TimeStamp'], df[f"{band}_Trend"], 
                color=color, linewidth=2, label=band)

    # Add Event Markers
    y_min, y_max = ax.get_ylim()
    for time, label in events:
        # Add a small vertical line or annotation
        ax.annotate(label, 
                    xy=(time, y_max), 
                    xytext=(0, 10), textcoords='offset points',
                    rotation=90, fontsize=8, color='gray',
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

    # Formatting
    ax.legend(loc='upper right')
    title_suffix = "Left/Right Coherence" if GRAPH_LEFT_RIGHT_COHERENCE else "Average Absolute Brain Waves"
    ax.set_title(f"Mind Monitor - {title_suffix}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude (Bels)" if not GRAPH_LEFT_RIGHT_COHERENCE else "Diff")
    
    # Format X-axis time
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    process_and_graph(FILE_PATH)