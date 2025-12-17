import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, lfilter, lfilter_zi
import sys

# --- CONFIGURATION ---
SF = 256             # Sampling Frequency
WINDOW_SECONDS = 30  # View window
BUFFER_SIZE = SF * WINDOW_SECONDS 

# SMOOTHING FACTOR
# Lower = Smoother/Slower lines. Higher = More responsive/Jittery.
# 1.0Hz is a good balance for visualizing "Flow"
SMOOTHING_FREQ = 1.0 

BANDS = [
    # Name, Low, High, Color
    ('Delta', 0.5, 4,  'r'),   # Red
    ('Theta', 4,   8,  'y'),   # Yellow
    ('Alpha', 8,   13, 'g'),   # Green
    ('Beta',  13,  30, 'c')    # Cyan
]

class Filter:
    """Helper to filter data in real-time."""
    def __init__(self, b, a):
        self.b = b
        self.a = a
        self.zi = lfilter_zi(self.b, self.a)
    
    def process(self, data):
        filtered_data, self.zi = lfilter(self.b, self.a, data, zi=self.zi)
        return filtered_data

def create_bandpass_filter(low, high, sf):
    nyq = 0.5 * sf
    b, a = butter(3, [low/nyq, high/nyq], btype='band')
    return Filter(b, a)

def create_lowpass_filter(cutoff, sf):
    nyq = 0.5 * sf
    b, a = butter(3, cutoff/nyq, btype='low')
    return Filter(b, a)

class SmoothEnvelopeVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Brainwave Power (Envelopes)")
        self.resize(1200, 600)
        
        # UI Setup
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget)
        
        # Connect to Stream
        print("Looking for stream...")
        from stream_helper import resolve_stream; streams = resolve_stream(timeout=5)
        if not streams:
            print("No stream found! Run 'muselsl stream' first.")
            sys.exit(1)
        self.inlet = StreamInlet(streams[0])

        # Plot Setup
        self.plot = self.graph_widget.addPlot(title="Brainwave Strength (30s History)")
        self.plot.setLabel('left', 'Strength (uV)')
        self.plot.setLabel('bottom', 'Time (Seconds)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()
        
        self.x_axis = np.linspace(-WINDOW_SECONDS, 0, BUFFER_SIZE)
        self.plot.setXRange(-WINDOW_SECONDS, 0)
        self.plot.setYRange(0, 20) # Range starts at 0 since envelopes are positive

        # Initialize Pipelines
        self.curves = []
        self.bp_filters = [] # Bandpass filters (to isolate Alpha/Beta/etc)
        self.lp_filters = [] # Lowpass filters (to smooth the line)
        self.buffers = []

        for name, low, high, color in BANDS:
            # 1. The Line
            curve = self.plot.plot(pen=pg.mkPen(color, width=3), name=name)
            self.curves.append(curve)
            
            # 2. The Math (Bandpass -> Lowpass)
            self.bp_filters.append(create_bandpass_filter(low, high, SF))
            self.lp_filters.append(create_lowpass_filter(SMOOTHING_FREQ, SF))
            
            # 3. The Data Storage
            self.buffers.append(np.zeros(BUFFER_SIZE))

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(33) # ~30 FPS
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        chunk, _ = self.inlet.pull_chunk(timeout=0.0)
        
        if chunk:
            chunk_np = np.array(chunk)
            
            if chunk_np.shape[1] > 4:
                chunk_np = chunk_np[:, :4]
                
            # Average sensors for global signal
            global_signal = np.mean(chunk_np, axis=1)
            n_new = len(global_signal)
            
            for i in range(len(BANDS)):
                # Step 1: Isolate the Band (e.g., get only Alpha)
                band_signal = self.bp_filters[i].process(global_signal)
                
                # Step 2: Rectify (Make all values positive)
                rectified_signal = np.abs(band_signal)
                
                # Step 3: Smooth (Filter out the jitter)
                smooth_envelope = self.lp_filters[i].process(rectified_signal)
                
                # Update Buffer
                self.buffers[i] = np.roll(self.buffers[i], -n_new)
                self.buffers[i][-n_new:] = smooth_envelope
                
                # Update Plot
                self.curves[i].setData(self.x_axis, self.buffers[i])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = SmoothEnvelopeVisualizer()
    window.show()
    sys.exit(app.exec())