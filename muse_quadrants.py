import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, sosfilt, sosfilt_zi
import sys

# --- CONFIGURATION ---
SF = 256             # Sampling Frequency
WINDOW_SECONDS = 30  # View window
BUFFER_SIZE = SF * WINDOW_SECONDS 
SMOOTHING_FREQ = 0.1 # Very slow, smooth visualization (Lazy Log)

# Channel Mapping for Muse (Standard LSL order: TP9, AF7, AF8, TP10)
# We map them to grid positions (Row, Col)
# AF7 (Front Left) -> (0, 0) | Index 1
# AF8 (Front Right)-> (0, 1) | Index 2
# TP9 (Back Left)  -> (1, 0) | Index 0
# TP10 (Back Right)-> (1, 1) | Index 3
SENSOR_MAP = [
    {"name": "Left Forehead (AF7)",  "idx": 1, "pos": (0, 0)},
    {"name": "Right Forehead (AF8)", "idx": 2, "pos": (0, 1)},
    {"name": "Left Ear (TP9)",       "idx": 0, "pos": (1, 0)},
    {"name": "Right Ear (TP10)",     "idx": 3, "pos": (1, 1)},
]

BANDS = [
    ('Delta(0.5-4Hz)', 0.5, 4,  'r'),
    ('Theta(4-8Hz)', 4,   8,  'y'),
    ('Alpha(8-13Hz)', 8,   13, 'g'),
    ('Beta(13-30Hz)',  13,  30, 'c')
]

class Filter:
    """Helper to maintain filter state."""
    def __init__(self, sos):
        self.sos = sos
        self.zi = sosfilt_zi(self.sos)
    
    def process(self, data):
        # We need to update zi to keep the filter continuous across chunks
        filtered_data, self.zi = sosfilt(self.sos, data, zi=self.zi)
        return filtered_data

def create_bandpass(low, high, sf):
    sos = butter(3, [low/(sf/2), high/(sf/2)], btype='band', output='sos')
    return Filter(sos)

def create_lowpass(cutoff, sf):
    sos = butter(1, cutoff/(sf/2), btype='low', output='sos')
    return Filter(sos)

class SensorProcessor:
    """Handles the math (Filters -> Log -> Smooth) for ONE single sensor."""
    def __init__(self):
        self.bp_filters = []
        self.lp_filters = []
        self.buffers = []
        
        for _ in BANDS:
            # Bandpass to isolate wave
            self.bp_filters.append(create_bandpass(_[1], _[2], SF))
            # Lowpass to smooth envelope
            self.lp_filters.append(create_lowpass(SMOOTHING_FREQ, SF))
            # Data buffer for plotting
            self.buffers.append(np.zeros(BUFFER_SIZE))

    def process(self, new_data_chunk):
        """Input: 1D array of new samples for this sensor."""
        n_new = len(new_data_chunk)
        
        for i in range(len(BANDS)):
            # 1. Bandpass
            wave = self.bp_filters[i].process(new_data_chunk)
            # 2. Rectify & Log (The "Lazy Log" Magic)
            wave = np.log1p(np.abs(wave))
            # 3. Smooth
            envelope = self.lp_filters[i].process(wave)
            
            # 4. Store
            self.buffers[i] = np.roll(self.buffers[i], -n_new)
            self.buffers[i][-n_new:] = envelope
            
        return self.buffers

class QuadrantVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Muse Quadrant Analysis (Front/Back/Left/Right)")
        self.resize(1400, 900)
        
        # UI Layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget)
        
        # Connect Stream
        print("Looking for stream...")
        from stream_helper import resolve_stream; streams = resolve_stream(timeout=5)
        if not streams:
            print("No stream found! Run 'muselsl stream' first.")
            sys.exit(1)
        self.inlet = StreamInlet(streams[0])

        self.x_axis = np.linspace(-WINDOW_SECONDS, 0, BUFFER_SIZE)
        
        # Initialize Processors and Plots
        self.processors = {}  # Key: index (0-3), Value: SensorProcessor
        self.curves = {}      # Key: index (0-3), Value: List of 4 curves
        
        for sensor in SENSOR_MAP:
            idx = sensor['idx']
            row, col = sensor['pos']
            name = sensor['name']
            
            # 1. Create Processor for this sensor
            self.processors[idx] = SensorProcessor()
            
            # 2. Create Plot Area
            plot = self.graph_widget.addPlot(row=row, col=col, title=name)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setYRange(0, 6) # Fixed Log scale
            plot.setXRange(-WINDOW_SECONDS, 0)
            if idx == 1: plot.addLegend(offset=(10, 10)) # Only add legend to one to save space
            
            # 3. Create Curves (Lines)
            self.curves[idx] = []
            for band_name, low, high, color in BANDS:
                curve = plot.plot(pen=pg.mkPen(color, width=2), name=band_name)
                self.curves[idx].append(curve)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(33) # 30 FPS
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        chunk, _ = self.inlet.pull_chunk(timeout=0.0)
        
        if chunk:
            chunk_np = np.array(chunk)
            
            # Trim extra channels
            if chunk_np.shape[1] > 4:
                chunk_np = chunk_np[:, :4]

            # Process each sensor individually
            for sensor in SENSOR_MAP:
                idx = sensor['idx']
                
                # Extract data for this specific sensor (Column `idx`)
                sensor_data = chunk_np[:, idx]
                
                # Run the math pipeline
                processed_buffers = self.processors[idx].process(sensor_data)
                
                # Update the 4 lines for this sensor
                for i in range(4):
                    self.curves[idx][i].setData(self.x_axis, processed_buffers[i])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = QuadrantVisualizer()
    window.show()
    sys.exit(app.exec())