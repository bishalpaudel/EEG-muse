import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, lfilter, lfilter_zi
import sys

# --- CONFIGURATION ---
SF = 256             # Muse sampling frequency
WINDOW_SECONDS = 1  # Show 30 seconds of history
BUFFER_SIZE = SF * WINDOW_SECONDS 

# Define the bands
BANDS = [
    # Name, Low, High, Color
    ('Delta', 0.5, 4,  'r'),   # Red
    ('Theta', 4,   8,  'y'),   # Yellow
    ('Alpha', 8,   13, 'g'),   # Green
    ('Beta',  13,  30, 'c')    # Cyan
]

class RealTimeFilter:
    """Helper to filter data in real-time using a butterworth filter."""
    def __init__(self, lowcut, highcut, sf, order=3):
        nyq = 0.5 * sf
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype='band')
        self.zi = lfilter_zi(self.b, self.a)
    
    def process(self, new_data):
        filtered_data, self.zi = lfilter(self.b, self.a, new_data, zi=self.zi)
        return filtered_data

class AllWavesVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Real-Time Brainwaves (30s Window)")
        self.resize(1200, 600)
        
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

        # Create ONE Single Plot
        self.plot = self.graph_widget.addPlot(title="Combined Brainwaves")
        self.plot.setLabel('left', 'Amplitude (uV)')
        self.plot.setLabel('bottom', 'Time (Seconds)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend() # Add the legend
        
        # Set X-Axis to show -30 to 0 seconds
        self.x_axis = np.linspace(-WINDOW_SECONDS, 0, BUFFER_SIZE)
        self.plot.setXRange(-WINDOW_SECONDS, 0)
        self.plot.setYRange(-30, 30) # Fixed vertical range for stability

        # Setup Curves and Filters
        self.curves = []
        self.filters = []
        self.buffers = []

        for name, low, high, color in BANDS:
            # Add curve to the single plot
            curve = self.plot.plot(pen=pg.mkPen(color, width=2), name=name)
            self.curves.append(curve)
            
            # Create Filter
            self.filters.append(RealTimeFilter(low, high, SF))
            
            # Create Buffer
            self.buffers.append(np.zeros(BUFFER_SIZE))

        # Timer (30 FPS is enough for this slow view)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(33) 
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        # Get data
        chunk, _ = self.inlet.pull_chunk(timeout=0.0)
        
        if chunk:
            chunk_np = np.array(chunk)
            
            # Handle extra channels
            if chunk_np.shape[1] > 4:
                chunk_np = chunk_np[:, :4]
                
            # Average sensors for Global Brain Signal
            global_signal = np.mean(chunk_np, axis=1)
            n_new = len(global_signal)
            
            # Process each band
            for i in range(len(BANDS)):
                smooth_chunk = self.filters[i].process(global_signal)
                
                # Roll buffer
                self.buffers[i] = np.roll(self.buffers[i], -n_new)
                self.buffers[i][-n_new:] = smooth_chunk
                
                # Update visual
                self.curves[i].setData(self.x_axis, self.buffers[i])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AllWavesVisualizer()
    window.show()
    sys.exit(app.exec())