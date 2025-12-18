import sys
import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from analysis_helper import BANDS, BAND_NAMES, WAVE_COLORS, SF, calculate_band_powers
from compare_stat import StatisticalAnalyzer

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
        self.init_tabs()
        
        # Analyzer
        self.analyzer = StatisticalAnalyzer()
        self.current_data_a = None # Store processed data for stats
        self.current_data_b = None

    def init_ui(self):
        # --- Top Controls ---
        control_group = QtWidgets.QGroupBox("Comparison Settings")
        control_layout = QtWidgets.QHBoxLayout()
        
        # File A
        file_a_layout = QtWidgets.QVBoxLayout()
        file_a_layout.addWidget(QtWidgets.QLabel("Recording A:"))
        self.combo_file_a = QtWidgets.QComboBox()
        file_a_layout.addWidget(self.combo_file_a)
        control_layout.addLayout(file_a_layout)
        
        # File B
        file_b_layout = QtWidgets.QVBoxLayout()
        file_b_layout.addWidget(QtWidgets.QLabel("Recording B:"))
        self.combo_file_b = QtWidgets.QComboBox()
        file_b_layout.addWidget(self.combo_file_b)
        control_layout.addLayout(file_b_layout)
        
        # Channels
        chan_layout = QtWidgets.QVBoxLayout()
        chan_layout.addWidget(QtWidgets.QLabel("Channels:"))
        
        self.chan_checks = {}
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

    def init_tabs(self):
        self.tabs = QtWidgets.QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # Tab 1: Visual Trends
        self.tab_visual = QtWidgets.QWidget()
        self.visual_layout = QtWidgets.QVBoxLayout(self.tab_visual)
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.visual_layout.addWidget(self.graph_widget)
        self.tabs.addTab(self.tab_visual, "Visual Trends")
        
        # Tab 2: Statistical Analysis
        self.tab_stats = QtWidgets.QWidget()
        self.stats_layout = QtWidgets.QVBoxLayout(self.tab_stats)
        self.tabs.addTab(self.tab_stats, "Statistical Analysis")
        
        self.init_visual_plots()
        self.init_stats_ui()

    def init_visual_plots(self):
        self.plots = {}
        self.curves_a = {}
        self.curves_b = {}
        
        # Create 5 vertical plots
        for i, name in enumerate(BAND_NAMES):
            p = self.graph_widget.addPlot(row=i, col=0, title=f"{name} Power")
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel('left', 'Log Power')
            
            if i > 0:
                p.setXLink(self.plots[BAND_NAMES[0]])
            
            self.plots[name] = p
            
            # File A = Cyan
            self.curves_a[name] = p.plot(pen=pg.mkPen('#00FFFF', width=2), name="File A")
            # File B = Magenta
            self.curves_b[name] = p.plot(pen=pg.mkPen('#FF00FF', width=2), name="File B")

    def init_stats_ui(self):
        # 1. Band Selection
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel("Select Band to Analyze:"))
        self.combo_stats_band = QtWidgets.QComboBox()
        self.combo_stats_band.addItems(BAND_NAMES)
        self.combo_stats_band.currentIndexChanged.connect(self.update_stats_view)
        hbox.addWidget(self.combo_stats_band)
        hbox.addStretch()
        self.stats_layout.addLayout(hbox)
        
        # 2. Text Output
        self.stats_text = QtWidgets.QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_layout.addWidget(self.stats_text)
        
        # 3. Histogram Plot
        self.stats_plot = pg.PlotWidget(title="Distribution Comparison")
        self.stats_plot.showGrid(x=True, y=True, alpha=0.3)
        self.stats_layout.addWidget(self.stats_plot)

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
            
            # Channel filtering
            target_cols = []
            if 'TP9' in df.columns: # Named columns
                for chan in ['TP9', 'AF7', 'AF8', 'TP10']:
                    if self.chan_checks[chan].isChecked():
                        target_cols.append(chan)
            else:
                idx_map = {'TP9':1, 'AF7':2, 'AF8':3, 'TP10':4}
                cols = df.columns
                for chan in ['TP9', 'AF7', 'AF8', 'TP10']:
                     if self.chan_checks[chan].isChecked():
                        target_cols.append(cols[idx_map[chan]])
                        
            if not target_cols:
                return None, None 
                
            data = df[target_cols].values
            
            step_size = int(SF / 10) # 10 FPS
            window_size = SF
            n_samples = len(data)
            
            band_series = {name: [] for name in BAND_NAMES}
            timestamps = []
            
            for start in range(0, n_samples - window_size, step_size):
                end = start + window_size
                chunk = data[start:end]
                powers = calculate_band_powers(chunk, sf=SF, left_right_coherence=False)
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

        # Time Balancing: Crop to shorter duration
        min_len = min(len(t_a), len(t_b))
        
        # Truncate A
        t_a = t_a[:min_len]
        for name in BAND_NAMES:
            data_a[name] = data_a[name][:min_len]
            
        # Truncate B
        t_b = t_b[:min_len]
        for name in BAND_NAMES:
            data_b[name] = data_b[name][:min_len]

        print(f"Comparison cropped to {t_a[-1]:.1f} seconds (Length: {min_len})")

        # Store data for stats
        self.current_data_a = data_a
        self.current_data_b = data_b
        
        # 1. Update Visuals
        for name in BAND_NAMES:
            self.curves_a[name].setData(t_a, data_a[name])
            self.curves_b[name].setData(t_b, data_b[name])
            
            # Adjust ranges
            self.plots[name].setXRange(0, t_a[-1])
            
        # 2. Update Stats (initially whatever is selected)
        self.update_stats_view()

    def update_stats_view(self):
        if self.current_data_a is None or self.current_data_b is None:
            return
            
        band_name = self.combo_stats_band.currentText()
        
        # Get raw arrays for this band
        raw_a = self.current_data_a[band_name]
        raw_b = self.current_data_b[band_name]
        
        # Analyze
        res = self.analyzer.compare_bands(band_name, raw_a, raw_b)
        
        if not res['valid']:
            self.stats_text.setText(f"Error: {res['error']}")
            self.stats_plot.clear()
            return
            
        # Update Text
        html = f"""
        <h3>Statistical Analysis: {band_name}</h3>
        <ul>
            <li><b>File A Mean:</b> {res['mean_a']:.4f}</li>
            <li><b>File B Mean:</b> {res['mean_b']:.4f}</li>
            <li><b>Difference:</b> {res['pct_change']:+.2f}%</li>
            <li><b>P-Value:</b> {res['p_value']:.5f}</li>
        </ul>
        <p><b>Conclusion:</b> {res['conclusion']}</p>
        """
        self.stats_text.setHtml(html)
        
        # Update Histogram
        self.stats_plot.clear()
        
        # Simple Histogram using BarGraphItem or plain plot logic
        # For simplicity, let's use numpy histogram and plot steps
        
        y_a, x_a = np.histogram(res['clean_a'], bins=30, density=True)
        y_b, x_b = np.histogram(res['clean_b'], bins=30, density=True)
        
        # Plot as step curves
        # File A (Cyan)
        self.stats_plot.plot(x_a, y_a, stepMode="center", fillLevel=0, brush=(0, 255, 255, 100), pen='#00FFFF', name="File A")
        # File B (Magenta)
        self.stats_plot.plot(x_b, y_b, stepMode="center", fillLevel=0, brush=(255, 0, 255, 100), pen='#FF00FF', name="File B")
        self.stats_plot.addLegend()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = CompareWindow()
    window.show()
    sys.exit(app.exec())
