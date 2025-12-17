import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch
import sys

# --- SETTINGS ---
GRAPH_LEFT_RIGHT_COHERENCE = False  # True: (Left - Right), False: Average
AVERAGE_TRENDLINE_PERIOD = 30       # Moving average window (smaller = faster response)
WINDOW_SECONDS = 60                 # View history
UPDATE_FPS = 10                     # Updates per second
SF = 256                            # Muse Sampling Frequency

# Band Definitions
BANDS = {'Delta(0.5-4Hz)': (0.5, 4), 'Theta(4-8Hz)': (4, 8), 'Alpha(8-13Hz)': (8, 13), 'Beta(13-30Hz)': (13, 30), 'Gamma(30-45Hz)': (30, 45)}
WAVE_COLORS = ['#CC0000', '#9933CC', '#0099CC', '#669900', '#FF8A00']
BAND_NAMES = list(BANDS.keys())

class LiveMuseGraph(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-Time Muse Analysis (Calculated from Raw)")
        self.resize(1400, 800)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget)

        # --- 1. Connect to EEG Stream ---
        print("Looking for EEG stream...")
        from stream_helper import resolve_stream; streams = resolve_stream(timeout=5)
        if not streams:
            raise RuntimeError("Could not find EEG stream. Run 'muselsl stream' first.")
        
        self.inlet_eeg = StreamInlet(streams[0])
        print("EEG Stream found!")
        
        # Try to find optional Marker stream (Blinks)
        self.inlet_markers = None
        marker_streams = resolve_byprop('type', 'Markers', timeout=1)
        if marker_streams:
            self.inlet_markers = StreamInlet(marker_streams[0])
            print("Marker stream found.")

        # --- 2. Setup Buffers ---
        # Buffer for raw audio to calculate FFT (1 second window for stability)
        self.raw_buffer = np.zeros((SF * 2, 4)) 
        
        # Buffer for the plotted graph
        buffer_size = WINDOW_SECONDS * UPDATE_FPS
        self.x_axis = np.linspace(-WINDOW_SECONDS, 0, buffer_size)
        
        # Storage for calculated band powers
        self.data_buffers = [pd.Series(np.zeros(buffer_size)) for _ in range(5)]
        self.trend_buffers = [np.zeros(buffer_size) for _ in range(5)]

        # --- 3. Setup Plot ---
        self.plot = self.graph_widget.addPlot()
        self.plot.addLegend(offset=(10, 10))
        self.plot.setXRange(-WINDOW_SECONDS, 0)
        self.plot.setLabel('left', 'Power (Log Scale)')
        self.plot.setLabel('bottom', 'Time (Seconds)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        self.curves_raw = []
        self.curves_trend = []
        
        for i, name in enumerate(BAND_NAMES):
            color = WAVE_COLORS[i]
            # Dotted raw line
            c_raw = self.plot.plot(pen=pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DotLine), name=f"{name} (Raw)")
            # Solid trend line
            c_trend = self.plot.plot(pen=pg.mkPen(color, width=3), name=f"{name} (Trend)")
            self.curves_raw.append(c_raw)
            self.curves_trend.append(c_trend)

        # --- 4. Start Update Loop ---
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(1000 / UPDATE_FPS))
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def calculate_band_powers(self, eeg_data):
        """
        Performs FFT on the raw EEG data to get the power of Delta, Theta, etc.
        Input: (Samples x 4 Sensors)
        Output: List of 5 values (one per band) representing the 'Score' for that band.
        """
        # Calculate Power Spectral Density (PSD) using Welch's method
        # Transpose data so channels are rows
        nperseg = min(len(eeg_data), SF) # 1-second window
        freqs, psd = welch(eeg_data.T, SF, nperseg=nperseg)
        
        # psd shape is now (4 Sensors, Frequency Bins)
        
        band_powers = []
        for band_name in BAND_NAMES:
            low, high = BANDS[band_name]
            # Find frequencies in this band
            idx = np.logical_and(freqs >= low, freqs <= high)
            
            if GRAPH_LEFT_RIGHT_COHERENCE:
                # Left (TP9, AF7) vs Right (AF8, TP10)
                # Assuming standard order: TP9, AF7, AF8, TP10
                left_power = np.mean(psd[0:2, idx])
                right_power = np.mean(psd[2:4, idx])
                power = left_power - right_power
            else:
                # Average of all 4 sensors
                power = np.mean(psd[:, idx])
            
            # Convert to Log scale (Bels) to match Mind Monitor style
            # Adding 1e-6 to avoid log(0)
            band_powers.append(np.log10(power + 1e-6))
            
        return band_powers

    def update(self):
        # 1. Get new Raw EEG data
        # We assume 1/10th of a second has passed, so we expect ~25 samples
        chunk, _ = self.inlet_eeg.pull_chunk(timeout=0.0)
        
        if chunk:
            chunk_np = np.array(chunk)
            if chunk_np.shape[1] > 4:
                chunk_np = chunk_np[:, :4]
            
            # Add to raw buffer (scroll logic)
            self.raw_buffer = np.roll(self.raw_buffer, -len(chunk_np), axis=0)
            self.raw_buffer[-len(chunk_np):, :] = chunk_np
            
            # 2. Calculate the Band Powers from the Raw Buffer
            # We calculate based on the last 1 second of data for stability
            new_powers = self.calculate_band_powers(self.raw_buffer)
            
            # 3. Update Plots
            for i in range(5):
                # Update Pandas Buffer
                self.data_buffers[i] = pd.concat([
                    self.data_buffers[i].iloc[1:], 
                    pd.Series([new_powers[i]])
                ])
                
                # Calculate Trendline
                trend = self.data_buffers[i].rolling(window=AVERAGE_TRENDLINE_PERIOD).mean()
                
                # Update Visuals
                self.curves_raw[i].setData(self.x_axis, self.data_buffers[i].values)
                self.curves_trend[i].setData(self.x_axis, trend.values)

        # 4. Check for Markers (Optional)
        if self.inlet_markers:
            marker, _ = self.inlet_markers.pull_sample(timeout=0.0)
            if marker:
                # Draw a vertical line for the marker
                line = pg.InfiniteLine(pos=0, angle=90, pen='gray')
                self.plot.addItem(line)
                # Note: We aren't scrolling markers here to keep code simple, 
                # but they will appear at the "Now" (0) mark.

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = LiveMuseGraph()
    window.show()
    sys.exit(app.exec())