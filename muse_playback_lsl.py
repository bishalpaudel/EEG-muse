import sys
import time
import pandas as pd
import pylsl
from pylsl import StreamInfo, StreamOutlet

def replay_eeg(csv_path):
    print(f"Loading recording: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Expected columns: TimeStamp, TP9, AF7, AF8, TP10
    # We need to extract the 4 EEG columns
    required_cols = ['TP9', 'AF7', 'AF8', 'TP10']
    
    # Check if we have the raw columns
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file must contain raw EEG columns: {required_cols}")
        print(f"Found columns: {list(df.columns)}")
        return

    data = df[required_cols].to_numpy() # Shape: (samples, 4)
    
    # Create LSL Outlet
    # name='MusePlayback' helps distinguish from real 'Muse'
    info = StreamInfo('MusePlayback', 'EEG', 4, 256, 'float32', 'PlaybackSource')
    outlet = StreamOutlet(info)
    
    print(f"Started LSL Replay Stream for {csv_path}")
    print(f"Total Samples: {len(data)}")
    print("Press Ctrl+C to stop manually.")
    
    # Replay loop
    # 256 Hz = ~3.9ms per sample. 
    # Sending sample-by-sample in python might be slow/jittery.
    # Better to send in chunks (e.g. 32 samples ~ 8 times per second)
    
    chunk_size = 32
    sample_rate = 256
    sleep_time = chunk_size / sample_rate # ~0.125s
    
    idx = 0
    while idx < len(data):
        chunk = data[idx : idx + chunk_size]
        
        # pylsl expects list of lists (or valid numpy array logic, but list of lists is safest for py3)
        # However, push_chunk accepts numpy arrays if properly typed? 
        # Let's convert to list for safety with pylsl bindings.
        chunk_list = chunk.tolist()
        
        outlet.push_chunk(chunk_list)
        
        idx += chunk_size
        time.sleep(sleep_time)
        
        if idx % 2560 == 0:
            print(f"Replayed {idx} / {len(data)} samples...")

    print("Replay finished. Restarting loop...")
    # Loop forever so the graph doesn't die
    while True:
        idx = 0
        while idx < len(data):
            chunk = data[idx : idx + chunk_size]
            outlet.push_chunk(chunk.tolist())
            idx += chunk_size
            time.sleep(sleep_time)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python muse_playback_lsl.py <csv_file>")
        sys.exit(1)
        
    replay_eeg(sys.argv[1])
