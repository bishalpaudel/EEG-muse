import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, sosfilt, sosfilt_zi
import pandas as pd
import sys

# --- CONFIGURATION ---
SF = 256             # Sampling Frequency
WINDOW_SECONDS = 30  # View window (seconds)
BUFFER_SIZE = SF * WINDOW_SECONDS 
SMOOTHING_FREQ = 0.1 # 0.1Hz = Very smooth, slow moving lines

# Colors for the bands
BANDS = [
    ('Delta(0.5-4Hz)', 0.5, 4,  'r'),   # Red
    ('Theta(4-8Hz)', 4,   8,  'y'),   # Yellow
    ('Alpha(8-13Hz)', 8,   13, 'g'),   # Green
    ('Beta(13-30Hz)',  13,  30, 'c'),   # Cyan
    ('Gamma(30-44Hz)', 30,  44, 'm')    # Magenta
]

# Map physical sensors to the grid layout
# Indices: 0=TP9, 1=AF7, 2=AF8, 3=TP10
SENSOR_LAYOUT = [
    # (Name, Index, Row, Col)
    ("Left Forehead (AF7)",  1, 1, 0),
    ("Right Forehead (AF8)", 2, 1, 1),
    ("Left Ear (TP9)",       0, 2, 0),
    ("Right Ear (TP10)",     3, 2, 1)
]

class Filter:
    """Helper to maintain filter state (zi) for real-time processing."""
    def __init__(self, sos):
        self.sos = sos
        self.zi = sosfilt_zi(self.sos)
    
    def process(self, data):
        filtered_data, self.zi = sosfilt(self.sos, data, zi=self.zi)
        return filtered_data

def create_bandpass(low, high, sf):
    sos = butter(3, [low/(sf/2), high/(sf/2)], btype='band', output='sos')
    return Filter(sos)

def create_lowpass(cutoff, sf):
    sos = butter(1, cutoff/(sf/2), btype='low', output='sos')
    return Filter(sos)

class SignalProcessor:
    """
    Handles the math pipeline for ONE signal source (a sensor or the average).
    Pipeline: Raw -> Bandpass -> Rectify -> Log -> Lowpass -> Buffer
    """
    def __init__(self):
        self.bp_filters = []
        self.lp_filters = []
        self.buffers = []
        
        for _ in BANDS:
            # 1. Bandpass Filter (Isolate the wave type)
            self.bp_filters.append(create_bandpass(_[1], _[2], SF))
            # 2. Lowpass Filter (Smooth the envelope)
            self.lp_filters.append(create_lowpass(SMOOTHING_FREQ, SF))
            # 3. Data Buffer
            self.buffers.append(np.zeros(BUFFER_SIZE))

    def process_and_store(self, new_data_chunk):
        """Input: 1D array of new samples."""
        n_new = len(new_data_chunk)
        
        for i in range(len(BANDS)):
            # Step A: Bandpass
            wave = self.bp_filters[i].process(new_data_chunk)
            
            # Step B: Rectify & Log (The "Lazy Log" squashing)
            # We use log1p to avoid log(0) errors and compress spikes
            wave = np.log1p(np.abs(wave))
            
            # Step C: Smooth
            envelope = self.lp_filters[i].process(wave)
            
            # Step D: Store in buffer
            self.buffers[i] = np.roll(self.buffers[i], -n_new)
            self.buffers[i][-n_new:] = envelope
            
        return self.buffers

class Quadrants(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Muse Master Dashboard (Average + Quadrants)")
        self.resize(1600, 1000)
        
        # 1. Setup UI Layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget)
        
        # 2. Connect to Stream - OPTIONAL NOW (Only if no file loaded)
        # We don't block here anymore. We wait for check_stream or file load.
        # But for backward compatibility with standalone run, we can try.
        self.init_plots()

    def load_stream(self):
        print("Looking for stream...")
        from pylsl import StreamInlet, resolve_byprop
        from stream_helper import resolve_stream; streams = resolve_stream(timeout=5)
        if streams:
            self.inlet = StreamInlet(streams[0])
            self.x_axis = np.linspace(-WINDOW_SECONDS, 0, BUFFER_SIZE)
            
            # Reset X Ranges for all plots (in case coming from static playback)
            # Global Avg
            self.graph_widget.getItem(0, 0).setXRange(-WINDOW_SECONDS, 0)
            # Individual Sensors
            for _, _, r, c in SENSOR_LAYOUT:
                 self.graph_widget.getItem(r, c).setXRange(-WINDOW_SECONDS, 0)

            # Start Update Loop
            self.timer = QtCore.QTimer()
            self.timer.setInterval(33) # ~30 FPS
            self.timer.timeout.connect(self.update)
            self.timer.start()
        else:
             print("No stream found.")

    def load_static_file(self, csv_path):
        """Loads a static CSV file and displays the whole thing."""
        print(f"Quadrants Loading: {csv_path}")
        if hasattr(self, 'timer'):
            self.timer.stop()
            
        try:
            df = pd.read_csv(csv_path)
            
            # Extract Data (Expected 4 cols)
            # Try finding standard columns or just take 1-4
            cols = ['TP9', 'AF7', 'AF8', 'TP10']
            if all(c in df.columns for c in cols):
                data = df[cols].values
            else:
                data = df.iloc[:, 1:5].values # Assume timestamp is 0, then 4 sensors
            
            # Process Full Data
            # Create a time axis based on SF
            total_samples = len(data)
            duration = total_samples / SF
            t_axis = np.linspace(0, duration, total_samples)
            
            # Reuse logic? We need to iterate over bands.
            # But SignalProcessor works on chunks. 
            # We effectively want to filter the WHOLE signal.
            
            for i, (name, idx, r, c) in enumerate(SENSOR_LAYOUT):
                sensor_data = data[:, idx]
                
                # We can use scipy.signal.sosfilt on the whole array
                # For each band...
                for b_i, (_, low, high, _) in enumerate(BANDS):
                    # 1. Bandpass
                    sos_bp = butter(3, [low/(SF/2), high/(SF/2)], btype='band', output='sos')
                    wave = sosfilt(sos_bp, sensor_data)
                    
                    # 2. Rectify + Log
                    wave = np.log1p(np.abs(wave))
                    
                    # 3. Lowpass
                    sos_lp = butter(1, SMOOTHING_FREQ/(SF/2), btype='low', output='sos')
                    envelope = sosfilt(sos_lp, wave)
                    
                    # Update Curve
                    self.curves[idx][b_i].setData(t_axis, envelope)
                    self.graph_widget.getItem(r, c).setXRange(0, duration)
            
            # Global Average (calculate average of envelopes or envelope of averages?)
            # Dashboard logic was: Average RAW data, then process.
            # Let's match that.
            avg_data = np.mean(data, axis=1)
            for b_i, (_, low, high, _) in enumerate(BANDS):
                sos_bp = butter(3, [low/(SF/2), high/(SF/2)], btype='band', output='sos')
                wave = sosfilt(sos_bp, avg_data)
                wave = np.log1p(np.abs(wave))
                sos_lp = butter(1, SMOOTHING_FREQ/(SF/2), btype='low', output='sos')
                envelope = sosfilt(sos_lp, wave)
                
                self.curves['avg'][b_i].setData(t_axis, envelope)
                self.graph_widget.getItem(0, 0).setXRange(0, duration)
                
        except Exception as e:
            print(f"Error loading file: {e}")
            import traceback
            traceback.print_exc()

    def init_plots(self):
        """Initializes the processors and plot curves."""
        # 3. Initialize Processors & Plots
        self.processors = {} # Stores the math engines
        self.curves = {}     # Stores the visual lines
        
        # --- A. Setup Global Average Plot (Top Row) ---
        avg_plot = self.graph_widget.addPlot(row=0, col=0, colspan=2, title="GLOBAL AVERAGE (All Sensors)")
        avg_plot.setLabel('left', 'Activity (Log)')
        avg_plot.showGrid(x=True, y=True, alpha=0.3)
        avg_plot.setYRange(0, 6)
        avg_plot.setXRange(-WINDOW_SECONDS, 0)
        avg_plot.addLegend(offset=(10, 10))
        
        self.processors['avg'] = SignalProcessor()
        self.curves['avg'] = []
        for name, _, _, color in BANDS:
            c = avg_plot.plot(pen=pg.mkPen(color, width=3), name=name)
            self.curves['avg'].append(c)

        # --- B. Setup Individual Sensor Plots (Grid) ---
        for name, idx, r, c in SENSOR_LAYOUT:
            # Create Plot
            p = self.graph_widget.addPlot(row=r, col=c, title=name)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setYRange(0, 6)
            p.setXRange(-WINDOW_SECONDS, 0)
            
            # Create Processor & Curves
            self.processors[idx] = SignalProcessor()
            self.curves[idx] = []
            for _, _, _, color in BANDS:
                # Thinner lines for the smaller graphs
                c = p.plot(pen=pg.mkPen(color, width=2))
                self.curves[idx].append(c)

    def update(self):
        # Pull data
        chunk, _ = self.inlet.pull_chunk(timeout=0.0)
        
        if chunk:
            chunk_np = np.array(chunk)
            
            # 1. Handle Channel Count (Force 4)
            if chunk_np.shape[1] > 4:
                chunk_np = chunk_np[:, :4]
                
            # 2. Update Global Average
            # Calculate mean across the 4 sensors for this chunk
            global_avg_data = np.mean(chunk_np, axis=1)
            
            avg_buffers = self.processors['avg'].process_and_store(global_avg_data)
            for i in range(len(BANDS)):
                self.curves['avg'][i].setData(self.x_axis, avg_buffers[i])

            # 3. Update Individual Sensors
            for _, idx, _, _ in SENSOR_LAYOUT:
                # Extract specific sensor column
                sensor_data = chunk_np[:, idx]
                
                sensor_buffers = self.processors[idx].process_and_store(sensor_data)
                for i in range(len(BANDS)):
                    self.curves[idx][i].setData(self.x_axis, sensor_buffers[i])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Quadrants()
    # If running standalone, assume we want Live mode immediately
    window.load_stream()
    window.show()
    sys.exit(app.exec())