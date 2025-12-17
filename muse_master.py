import sys
import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import resolve_byprop

# Import the individual dashboard classes
# We use try-except blocks to import them to handle potential missing file errors smoothly
try:
    from muse_dashboard import MasterDashboard
    from muse_live_graph import LiveMuseGraph
    from muse_quadrants import QuadrantVisualizer
    from muse_waves import EEGVisualizer
    from muse_smooth_waves import SmoothWaveVisualizer
    from muse_envelope import SmoothEnvelopeVisualizer
    from muse_lazy_log import LazyLogVisualizer
    from muse_all_in_one import AllWavesVisualizer
    from muse_recorder import MuseRecorder
    import mind_monotor # Importing the visualization script
    import os
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ... (Imports remain the same, but we need subprocess)
import subprocess

class ModeSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Muse Master - Select Mode")
        self.resize(400, 200)
        self.mode = None
        self.filename = None
        
        layout = QtWidgets.QVBoxLayout(self)
        
        lbl = QtWidgets.QLabel("Select Operation Mode:")
        font = lbl.font()
        font.setPointSize(14)
        lbl.setFont(font)
        if hasattr(QtCore.Qt, 'AlignmentFlag'):
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        else:
            lbl.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(lbl)
        
        btn_live = QtWidgets.QPushButton("Live Stream")
        btn_live.setMinimumHeight(50)
        btn_live.clicked.connect(self.select_live)
        layout.addWidget(btn_live)
        
        btn_play = QtWidgets.QPushButton("Playback Recording")
        btn_play.setMinimumHeight(50)
        btn_play.clicked.connect(self.select_playback)
        layout.addWidget(btn_play)
        
    def select_live(self):
        self.mode = "LIVE"
        self.accept()
        
    def select_playback(self):
        # Ask for file immediately
        start_dir = os.path.join(os.getcwd(), 'recordings')
        if not os.path.exists(start_dir):
            start_dir = os.getcwd()
            
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open recorded Raw CSV', start_dir, "CSV Files (*.csv)")
        if fname:
            self.mode = "PLAYBACK"
            self.filename = fname
            self.accept()

class MuseMasterLauncher(QtWidgets.QMainWindow):
    def __init__(self, mode, playback_file=None):
        super().__init__()
        
        self.mode = mode
        self.playback_file = playback_file
        self.playback_process = None
        
        # Set Environment Variable to tell views which stream to target
        if self.mode == "PLAYBACK":
            os.environ['MUSE_STREAM_NAME'] = 'MusePlayback'
        else:
            if 'MUSE_STREAM_NAME' in os.environ:
                del os.environ['MUSE_STREAM_NAME']

        title_suffix = "(Live)" if mode == "LIVE" else "(Playback)"
        self.setWindowTitle(f"Brainwave Master Dashboard {title_suffix}")
        self.resize(1600, 1000)
        
        # Main Layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # --- Recording Controls (Only relevant in Live Mode usually, but user might want to re-record playback?) ---
        # Let's keep it simple: Record only in Live Mode.
        if self.mode == "LIVE":
            self.recorder = None
            control_layout = QtWidgets.QHBoxLayout()
            
            # Record Button
            self.btn_record = QtWidgets.QPushButton("Start Recording")
            self.btn_record.setCheckable(True)
            self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
            self.btn_record.clicked.connect(self.toggle_recording)
            control_layout.addWidget(self.btn_record)
            
            self.layout.addLayout(control_layout)
        
        # --------------------------
        
        # status label
        status_msg = "Checking for EEG stream..." if mode == "LIVE" else f"Starting playback: {os.path.basename(playback_file)}..."
        self.status_label = QtWidgets.QLabel(status_msg)
        self.layout.addWidget(self.status_label)
        
        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.layout.addWidget(self.tabs)
        self.tabs.hide() 
        
        # Timer to check for stream (works for both live and playback streams)
        self.check_stream_timer = QtCore.QTimer()
        self.check_stream_timer.setInterval(1000)
        self.check_stream_timer.timeout.connect(self.check_stream)
        
        # Launch playback if needed
        if self.mode == "PLAYBACK":
            self.start_playback_subprocess()
            
        self.check_stream_timer.start()
        self.loaded = False

    def start_playback_subprocess(self):
        # Path to the playback script
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(curr_dir, 'muse_playback_lsl.py')
        
        cmd = [sys.executable, script_path, self.playback_file]
        print(f"Launching playback: {cmd}")
        self.playback_process = subprocess.Popen(cmd)
        
    def closeEvent(self, event):
        # Ensure we kill the subprocess when closing the window
        if self.playback_process:
            print("Terminating playback process...")
            self.playback_process.terminate()
        super().closeEvent(event)

    def toggle_recording(self):
        is_recording = self.btn_record.isChecked()
        
        if is_recording:
            # Start Recording
            filename, ok = QtWidgets.QInputDialog.getText(self, "Save Recording", "Enter filename for recording:")
            
            if ok and filename:
                # Validate filename
                filename = "".join(c for c in filename if c.isalnum() or c in (' ', '_', '-')).strip()
                if not filename:
                    filename = "recording"
                    
                # Create recordings folder
                save_dir = os.path.join(os.getcwd(), 'recordings')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                # Timestamp to avoid overwrites
                timestamp = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd_HH-mm-ss")
                full_name = f"{filename}_{timestamp}.csv"
                full_path = os.path.join(save_dir, full_name)
                
                self.recorder = MuseRecorder(full_path)
                self.recorder.start()
                
                self.btn_record.setText(f"Recording... ({full_name})")
                self.btn_record.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; padding: 10px;")
            else:
                self.btn_record.setChecked(False)
        else:
            # Stop Recording
            if self.recorder:
                self.recorder.stop()
                self.recorder = None
            
            self.btn_record.setText("Start Recording")
            self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")

    def check_stream(self):
        if self.loaded: return
        
        target = os.environ.get('MUSE_STREAM_NAME')
        if target:
            streams = resolve_byprop('name', target, timeout=1)
        else:
            streams = resolve_byprop('type', 'EEG', timeout=1)
            
        if streams:
            self.status_label.setText("EEG Stream Found! Loading Dashboards...")
            self.check_stream_timer.stop()
            self.load_tabs()
        else:
            msg = "No EEG stream found. " 
            msg += "Please run 'muselsl stream'..." if self.mode == "LIVE" else "Waiting for playback script..."
            self.status_label.setText(msg)

    def load_tabs(self):
        try:
            # We instantiate the windows and add them as tabs.
            # QMainWindow can be added to QTabWidget.
            
            # Tab 1: Dashboard (Avg + Quadrants)
            self.tabs.addTab(MasterDashboard(), "Master Dashboard")
            
            # Tab 2: Live Graph (FFT Trends)
            self.tabs.addTab(LiveMuseGraph(), "Live Analysis")
            
            # Tab 3: Quadrants
            self.tabs.addTab(QuadrantVisualizer(), "Quadrants")
            
            # Tab 4: Smooth Waves
            self.tabs.addTab(SmoothWaveVisualizer(), "Smooth Waves")
            
            # Tab 5: Envelopes
            self.tabs.addTab(SmoothEnvelopeVisualizer(), "Envelopes")
            
            # Tab 6: Lazy Log (Smooth Log)
            self.tabs.addTab(LazyLogVisualizer(), "Lazy Log") 

            # Tab 7: Combined Waves
            self.tabs.addTab(AllWavesVisualizer(), "Combined Waves")

            # Tab 8: Raw Waves
            self.tabs.addTab(EEGVisualizer(), "Raw Monitor")
            
            self.tabs.show()
            self.status_label.hide() # Hide status once loaded
            self.loaded = True
            
        except SystemExit:
            # Catch modules exiting if they fail their own internal checks
            self.status_label.setText("Error: One of the modules failed to initialize (Sub-module exited).")
            self.status_label.show()
        except Exception as e:
            self.status_label.setText(f"Error loading tabs: {e}")
            self.status_label.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # Show Mode Selection Logic
    dialog = ModeSelectionDialog()
    if dialog.exec() == 1: # 1 is QDialog.DialogCode.Accepted
        window = MuseMasterLauncher(dialog.mode, dialog.filename)
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)
