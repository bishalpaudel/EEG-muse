import sys
import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from analysis_helper import BANDS, BAND_NAMES, WAVE_COLORS, SF, calculate_band_powers

# Extended ComboBox for searchable file lists (Reusing the one from dashboard if possible, or re-defining)
# To avoid circular imports, let's redefine a simple version or just standard ComboBox for now, 
# or copy the ExtendedComboBox class. For simplicity, standard ComboBox is fine, or we can copy the class.
# Let's use standard QComboBox for now to keep it simple, as the lists are simple.

class CompareWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brainwave Comparison Tool")
        self.resize(1400, 900)
        
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        self.init_ui()
        self.init_plots()

    def init_ui(self):
        # --- Top Controls ---
        control_group = QtWidgets.QGroupBox("Comparison Settings")
        control_layout = QtWidgets.QHBoxLayout()
        
        # File A
        file_a_layout = QtWidgets.QVBoxLayout()
        file_a_layout.addWidget(QtWidgets.QLabel("Recording A (Solid Line):"))
        self.combo_file_a = QtWidgets.QComboBox()
        file_a_layout.addWidget(self.combo_file_a)
        control_layout.addLayout(file_a_layout)
        
        # File B
        file_b_layout = QtWidgets.QVBoxLayout()
        file_b_layout.addWidget(QtWidgets.QLabel("Recording B (Dashed Line):"))
        self.combo_file_b = QtWidgets.QComboBox()
        file_b_layout.addWidget(self.combo_file_b)
        control_layout.addLayout(file_b_layout)
        
        # Channels
        chan_layout = QtWidgets.QVBoxLayout()
        chan_layout.addWidget(QtWidgets.QLabel("Channels to Analyze:"))
        
        self.chan_checks = {}
        # Layout: TP9, AF7, AF8, TP10
        # Indices in Muse CSV are usually TP9(1), AF7(2), AF8(3), TP10(4) - actually column indices vary.
        # We'll rely on name matching or column order. 
        # LiveGraph.py assumed cols or just took 1:5.
        # Let's stick to standard names.
        
        check_hbox = QtWidgets.QHBoxLayout()
        for chan in ['TP9', 'AF7', 'AF8', 'TP10']:
            chk = QtWidgets.QCheckBox(chan)
            chk.setChecked(True) # Default all on
            self.chan_checks[chan] = chk
            check_hbox.addWidget(chk)
            
        chan_layout.addLayout(check_hbox)
        control_layout.addLayout(chan_layout)
        
        # Action Buttons
        btn_layout = QtWidgets.QVBoxLayout()
        self.btn_compare = QtWidgets.QPushButton("Compare")
        self.btn_compare.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.btn_compare.clicked.connect(self.run_comparison)
        btn_layout.addWidget(self.btn_compare)
        
        control_layout.addLayout(btn_layout)
        
        control_group.setLayout(control_layout)
        self.layout.addWidget(control_group)
        
        # Refresh Files
        self.refresh_file_list()

    def init_plots(self):
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget)
        
        self.plots = {}
        self.curves_a = {}
        self.curves_b = {}
        
        # Create 5 vertical plots, one for each band
        for i, name in enumerate(BAND_NAMES):
            p = self.graph_widget.addPlot(row=i, col=0, title=f"{name} Power")
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel('left', 'Log Power')
            
            # Link X axes
            if i > 0:
                p.setXLink(self.plots[BAND_NAMES[0]])
            
            self.plots[name] = p
            
            # Use distinct high-contrast colors for comparison
            # File A = Cyan
            self.curves_a[name] = p.plot(pen=pg.mkPen('#00FFFF', width=2), name="File A")
            
            # File B = Magenta
            self.curves_b[name] = p.plot(pen=pg.mkPen('#FF00FF', width=2), name="File B")

    def refresh_file_list(self):
        start_dir = os.path.join(os.getcwd(), 'recordings')
        if not os.path.exists(start_dir):
            os.makedirs(start_dir)
            
        files = [f for f in os.listdir(start_dir) if f.endswith('.csv')]
        files.sort(reverse=True)
        
        self.combo_file_a.clear()
        self.combo_file_a.addItems(files)
        
        self.combo_file_b.clear()
        self.combo_file_b.addItems(files)
        # Select second file for B if available
        if len(files) > 1:
            self.combo_file_b.setCurrentIndex(1)

    def load_and_process(self, filename):
        """Loads a CSV and returns the band power trends (dict of name -> array)."""
        start_dir = os.path.join(os.getcwd(), 'recordings')
        path = os.path.join(start_dir, filename)
        
        if not os.path.exists(path):
            return None, None
            
        try:
            df = pd.read_csv(path)
            
            # Determine channels
            # Helper assumes input (Samples, Sensors)
            # We need to filter based on checked boxes.
            
            # Map checkboxes to columns
            # Standard Muse order in many CSVs: timestamp, TP9, AF7, AF8, TP10, Right AUX...
            # Or named columns.
            
            target_cols = []
            if 'TP9' in df.columns: # Named columns
                for chan in ['TP9', 'AF7', 'AF8', 'TP10']:
                    if self.chan_checks[chan].isChecked():
                        target_cols.append(chan)
            else:
                # Fallback indices: 1, 2, 3, 4
                idx_map = {'TP9':1, 'AF7':2, 'AF8':3, 'TP10':4}
                cols = df.columns
                for chan in ['TP9', 'AF7', 'AF8', 'TP10']:
                     if self.chan_checks[chan].isChecked():
                        target_cols.append(cols[idx_map[chan]])
                        
            if not target_cols:
                return None, None # No channels selected
                
            data = df[target_cols].values
            
            # Sliding Window Analysis
            # We must match the LiveGraph logic: 1s window, 10FPS (0.1s step)
            step_size = int(SF / 10) # 10 FPS
            window_size = SF
            
            n_samples = len(data)
            duration = n_samples / SF
            
            band_series = {name: [] for name in BAND_NAMES}
            timestamps = []
            
            for start in range(0, n_samples - window_size, step_size):
                end = start + window_size
                chunk = data[start:end]
                
                # Use helper
                powers = calculate_band_powers(chunk, sf=SF, left_right_coherence=False)
                # Note: Comparison doesn't support left-right coherence toggle yet, assumes avg of selected
                
                for i, name in enumerate(BAND_NAMES):
                    band_series[name].append(powers[i])
                
                timestamps.append(start / SF)
                
            return timestamps, band_series
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None, None

    def run_comparison(self):
        file_a = self.combo_file_a.currentText()
        file_b = self.combo_file_b.currentText()
        
        if not file_a or not file_b:
            return

        print(f"Comparing {file_a} vs {file_b}")
        
        # Process A
        t_a, data_a = self.load_and_process(file_a)
        
        # Process B
        t_b, data_b = self.load_and_process(file_b)
        
        if t_a is None or t_b is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Could not process one or both files.")
            return

        # Plot
        for name in BAND_NAMES:
            self.curves_a[name].setData(t_a, data_a[name])
            self.curves_b[name].setData(t_b, data_b[name])
            
            # Adjust ranges
            max_t = max(t_a[-1], t_b[-1])
            self.plots[name].setXRange(0, max_t)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = CompareWindow()
    window.show()
    sys.exit(app.exec())
