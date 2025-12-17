import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch
import threading
import time
from datetime import datetime

class MuseRecorder:
    def __init__(self, filename):
        self.filename = filename
        if not self.filename.endswith('.csv'):
            self.filename += '.csv'
            
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.daemon = True # Kill if main app dies
        self.thread.start()
        print(f"Recording started: {self.filename}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print(f"Recording stopped.")
            
    def _record_loop(self):
        print("Looking for stream for recorder...")
        streams = resolve_byprop('type', 'EEG', timeout=5)
        if not streams:
            print("Recorder: No EEG stream found.")
            self.running = False
            return
            
        inlet = StreamInlet(streams[0])
        print("Recorder: Stream connected.")
        
        # Raw Data Format
        sensors = ['TP9', 'AF7', 'AF8', 'TP10']
        columns = ['TimeStamp'] + sensors
        
        recorded_data = []
        last_flush_time = time.time()
        
        while self.running:
            # Pull available data
            chunk, timestamps = inlet.pull_chunk(timeout=0.2)
            
            if chunk:
                # Timestamps: We'll use system time to be safe/simple for CSV, 
                # or we can use the LSL timestamps if we want relative time.
                # For playback, we just need the data sequence. 
                # Let's use current wall clock time for the record.
                
                chunk_np = np.array(chunk)
                current_time = datetime.now()
                
                # Check channel count
                if chunk_np.shape[1] > 4:
                    chunk_np = chunk_np[:, :4]
                    
                for i in range(len(chunk_np)):
                    row = [current_time] + chunk_np[i].tolist()
                    recorded_data.append(row)
            
            # Periodically write to disk (every 3 seconds)
            if time.time() - last_flush_time > 3.0 and len(recorded_data) > 0:
                self._save_to_csv(recorded_data, columns, mode='a')
                recorded_data = [] 
                last_flush_time = time.time()
                
            time.sleep(0.01)
            
        # Final flush
        if len(recorded_data) > 0:
            self._save_to_csv(recorded_data, columns, mode='a')
            
    def _save_to_csv(self, data, columns, mode='w'):
        df = pd.DataFrame(data, columns=columns)
        
        # If file doesn't exist, include header. If it does (and we are appending), exclude header.
        import os
        header = not os.path.exists(self.filename)
        
        # mode='a' appends, but we need to ensure we don't duplicate headers if file exists
        df.to_csv(self.filename, mode=mode, header=header, index=False)
