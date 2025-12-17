import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, lfilter, lfilter_zi
import sys

# --- CONFIGURATION ---
BUFFER_SIZE = 1000   # View window size
SF = 256             # Sampling Frequency (Muse default)
# Define the bands we want to see as smooth waves
BANDS = [
    ('Delta (0.5-4Hz)', 0.5, 4, 'r'),   # Red
    ('Theta (4-8Hz)',   4,   8, 'y'),   # Yellow
    ('Alpha (8-13Hz)',  8,   13, 'g'),  # Green
    ('Beta (13-30Hz)',  13,  30, 'c')   # Cyan
]

class RealTimeFilter:
    """Helper to filter data in real-time using a butterworth filter."""
    def __init__(self, lowcut, highcut, sf, order=3):
        nyq = 0.5 * sf
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype='band')
        # Initial filter state (zi) to allow continuous streaming without "jumps"
        self.zi = lfilter_zi(self.b, self.a)
    
    def process(self, new_data):
        # Filter the new chunk of data, updating the filter state (zi)
        filtered_data, self.zi = lfilter(self.b, self.a, new_data, zi=self.zi)
        return filtered_data

class SmoothWaveVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Real-Time Smooth Brainwaves")
        self.resize(1000, 800)
        
        # UI Layout
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

        # Setup Plots & Filters
        self.plots = []
        self.curves = []
        self.filters = []
        self.buffers = []

        # We will average all 4 sensors into 1 signal for a cleaner "Global Brain" view
        # and then split that 1 signal into 4 frequency bands.
        
        for i, (name, low, high, color) in enumerate(BANDS):
            # Create Plot
            p = self.graph_widget.addPlot(row=i, col=0)
            p.setLabel('left', name)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setRange(yRange=(-20, 20)) # Fixed range keeps waves centered
            
            # Create Curve
            curve = p.plot(pen=pg.mkPen(color, width=2))
            self.plots.append(p)
            self.curves.append(curve)
            
            # Create Filter for this band
            self.filters.append(RealTimeFilter(low, high, SF))
            
            # Create Data Buffer
            self.buffers.append(np.zeros(BUFFER_SIZE))

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(20) # 50 FPS
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        # Get data
        chunk, _ = self.inlet.pull_chunk(timeout=0.0)
        
        if chunk:
            chunk_np = np.array(chunk)
            
            # Handle extra channels (take first 4)
            if chunk_np.shape[1] > 4:
                chunk_np = chunk_np[:, :4]
                
            # Average the 4 sensors to get one "Global" brain signal
            # This reduces noise and makes the sine waves look much cleaner
            global_signal = np.mean(chunk_np, axis=1)
            
            n_new = len(global_signal)
            
            # Process each band
            for i in range(len(BANDS)):
                # Apply the bandpass filter to the new data
                smooth_chunk = self.filters[i].process(global_signal)
                
                # Roll buffer and add new smoothed data
                self.buffers[i] = np.roll(self.buffers[i], -n_new)
                self.buffers[i][-n_new:] = smooth_chunk
                
                # Update visual
                self.curves[i].setData(self.buffers[i])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = SmoothWaveVisualizer()
    window.show()
    sys.exit(app.exec())