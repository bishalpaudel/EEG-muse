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

# SMOOTHING SETTINGS
# 0.1Hz = Very slow, smooth changes (approx 10-second rolling average)
SMOOTHING_FREQ = 0.1 

BANDS = [
    # Name, Low, High, Color
    ('Delta (0.5-4Hz)', 0.5, 4,  'r'),   # Red
    ('Theta (4-8Hz)', 4,   8,  'y'),   # Yellow
    ('Alpha (8-13Hz)', 8,   13, 'g'),   # Green
    ('Beta (13-30Hz)', 13,  30, 'c')    # Cyan
]

class Filter:
    """Helper to filter data using Second-Order Sections (SOS) for stability at low freqs."""
    def __init__(self, sos):
        self.sos = sos
        self.zi = sosfilt_zi(self.sos)
    
    def process(self, data):
        filtered_data, self.zi = sosfilt(self.sos, data, zi=self.zi)
        return filtered_data

def create_bandpass_filter(low, high, sf):
    nyq = 0.5 * sf
    sos = butter(3, [low/nyq, high/nyq], btype='band', output='sos')
    return Filter(sos)

def create_lowpass_filter(cutoff, sf):
    nyq = 0.5 * sf
    # Use 1st order filter for a gentle exponential moving average effect
    sos = butter(1, cutoff/nyq, btype='low', output='sos')
    return Filter(sos)

class LazyLogVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Brainwave Activity (Log Scale + Smooth)")
        self.resize(1200, 600)
        
        # UI Setup
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget)
        
        # Plot Setup
        self.plot = self.graph_widget.addPlot(title="Brainwave Activity (Logarithmic)")
        self.plot.setLabel('left', 'Activity Level (Log)')
        self.plot.setLabel('bottom', 'Time (Seconds)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()
        
        self.x_axis = np.linspace(-WINDOW_SECONDS, 0, BUFFER_SIZE)
        self.plot.setXRange(-WINDOW_SECONDS, 0)
        
        # A Log scale usually results in values between 0 and 5 for EEG
        self.plot.setYRange(0, 6) 

        # Initialize Pipelines
        self.curves = []
        self.bp_filters = [] 
        self.lp_filters = [] 
        self.buffers = []

        # Connect Stream
        print("Looking for stream...")
        from stream_helper import resolve_stream; streams = resolve_stream(timeout=5)
        if not streams:
            print("No stream found! Run 'muselsl stream' first.")
            sys.exit(1)
        self.inlet = StreamInlet(streams[0])

        for name, low, high, color in BANDS:
            curve = self.plot.plot(pen=pg.mkPen(color, width=3), name=name)
            self.curves.append(curve)
            
            self.bp_filters.append(create_bandpass_filter(low, high, SF))
            self.lp_filters.append(create_lowpass_filter(SMOOTHING_FREQ, SF))
            self.buffers.append(np.zeros(BUFFER_SIZE))

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(33) 
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        chunk, _ = self.inlet.pull_chunk(timeout=0.0)
        
        if chunk:
            chunk_np = np.array(chunk)
            
            if chunk_np.shape[1] > 4:
                chunk_np = chunk_np[:, :4]
                
            # Global Average
            global_signal = np.mean(chunk_np, axis=1)
            n_new = len(global_signal)
            
            for i in range(len(BANDS)):
                # 1. Isolate Band
                band_signal = self.bp_filters[i].process(global_signal)
                
                # 2. Rectify (Absolute Value)
                rectified_signal = np.abs(band_signal)
                
                # 3. Logarithm (The "Squasher")
                # We add 1 so we don't try to calculate log(0)
                # This compresses huge spikes down significantly
                log_signal = np.log1p(rectified_signal)
                
                # 4. Smooth (The "Slower")
                smooth_envelope = self.lp_filters[i].process(log_signal)
                
                # Update Buffer
                self.buffers[i] = np.roll(self.buffers[i], -n_new)
                self.buffers[i][-n_new:] = smooth_envelope
                
                # Update Plot
                self.curves[i].setData(self.x_axis, self.buffers[i])

if __name__ == '__main__':
    # Connect Stream
    print("Looking for stream...")
    from stream_helper import resolve_stream; streams = resolve_stream(timeout=5)
    if not streams:
        print("No stream found! Run 'muselsl stream' first.")
        sys.exit(1)
    
    app = QtWidgets.QApplication(sys.argv)
    
    # Initialize Stream Inlet here to pass to window if needed, 
    # but the class handles it fine globally in this script style.
    window = LazyLogVisualizer()
    window.inlet = StreamInlet(streams[0]) # Inject inlet
    
    window.show()
    sys.exit(app.exec())