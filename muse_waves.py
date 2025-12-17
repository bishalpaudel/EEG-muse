import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_byprop
import sys

# --- CONFIGURATION ---
BUFFER_SIZE = 1000  # Number of data points to show on screen (approx 4 seconds)
COLORS = ['r', 'g', 'b', 'y']  # Red, Green, Blue, Yellow lines for 4 sensors
LABELS = ['TP9 (Left Ear)', 'AF7 (Left Forehead)', 'AF8 (Right Forehead)', 'TP10 (Right Ear)']

class EEGVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 1. Setup the UI Window
        self.setWindowTitle("Real-Time Muse EEG")
        self.resize(1200, 800)
        
        # Create the main layout container
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # Create the PyQtGraph plotting area
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget)
        
        # 2. Connect to LSL Stream
        print("Looking for an EEG stream...")
        from stream_helper import resolve_stream; streams = resolve_stream(timeout=5)
        if len(streams) == 0:
            print("Error: Could not find stream. Make sure 'muselsl stream' is running in another terminal.")
            sys.exit(1)
            
        self.inlet = StreamInlet(streams[0])
        print("Stream found!")

        # 3. Initialize Data Buffers and Plots
        self.data_buffer = np.zeros((BUFFER_SIZE, 4))
        self.curves = []
        
        # Create 4 stacked plots (one for each sensor)
        for i in range(4):
            plot = self.graph_widget.addPlot(row=i, col=0)
            
            # Labeling
            plot.setLabel('left', LABELS[i])
            if i == 3:
                plot.setLabel('bottom', 'Time (Samples)')
            
            # visual settings to make it look cool
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setYRange(700, 900) # Approximate range for Muse raw data (uV)
            # You might need to adjust this range or use plot.enableAutoRange()
            
            # Create the line object
            curve = plot.plot(pen=COLORS[i])
            self.curves.append(curve)

        # 4. Setup Update Timer (runs every 20ms)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(20) 
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        # Pull all available data from the stream
        chunk, timestamp = self.inlet.pull_chunk(timeout=0.0)
        
        if chunk:
            chunk_np = np.array(chunk)
            
            # Handle the 5-channel vs 4-channel issue
            if chunk_np.shape[1] > 4:
                chunk_np = chunk_np[:, :4]
            
            # Get the number of new samples
            n_new = len(chunk_np)
            
            # Shift the old data to the left
            self.data_buffer = np.roll(self.data_buffer, -n_new, axis=0)
            
            # Insert the new data at the end
            self.data_buffer[-n_new:, :] = chunk_np
            
            # Update the 4 lines on the screen
            for i in range(4):
                self.curves[i].setData(self.data_buffer[:, i])

if __name__ == '__main__':
    # Start the Qt Application
    app = QtWidgets.QApplication(sys.argv)
    window = EEGVisualizer()
    window.show()
    sys.exit(app.exec())
