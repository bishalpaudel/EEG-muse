import sys
import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import resolve_byprop

# Import the individual dashboard classes
# We use try-except blocks to import them to handle potential missing file errors smoothly
try:
    from quadrants import Quadrants
    from live_graph import LiveMuseGraph
    from muse_recorder import MuseRecorder
    import os
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ... (Imports remain the same, but we need subprocess)
import subprocess

# Helper Class for Searchable ComboBox
class ExtendedComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(ExtendedComboBox, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setEditable(True)

        # add a filter model to filter matching items
        self.pFilterModel = QtCore.QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.model())

        # add a completer, which uses the filter model
        self.completer = QtWidgets.QCompleter(self.pFilterModel, self)
        # always show all (filtered) completions
        self.completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self.setCompleter(self.completer)

        # connect signals
        self.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
        self.completer.activated.connect(self.on_completer_activated)

    def on_completer_activated(self, text):
        if text:
            index = self.findText(text)
            self.setCurrentIndex(index)
            # self.activated[str].emit(self.itemText(index))

    def setModel(self, model):
        super(ExtendedComboBox, self).setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super(ExtendedComboBox, self).setModelColumn(column)


class MuseMasterLauncher(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.mode = "LIVE"
        self.playback_file = None
        self.playback_process = None
        self.recorder = None
        
        # Ensure we start fresh with environment variables
        if 'MUSE_STREAM_NAME' in os.environ:
            del os.environ['MUSE_STREAM_NAME']

        self.setWindowTitle("Brainwave Master Dashboard (Live)")
        self.resize(1600, 1000)
        
        # Main Layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # --- Top Control Bar ---
        control_layout = QtWidgets.QHBoxLayout()
        
        # 1. Record Button
        self.btn_record = QtWidgets.QPushButton("Start Recording")
        self.btn_record.setCheckable(True)
        self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_record.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.btn_record)
        
        # 2. Playback Controls (Dropdown + Play Button)
        playback_layout = QtWidgets.QHBoxLayout()
        
        self.combo_files = ExtendedComboBox()
        self.combo_files.setMinimumWidth(300)
        self.combo_files.setStyleSheet("padding: 8px;")
        self.update_file_list()
        playback_layout.addWidget(self.combo_files)
        
        self.btn_play = QtWidgets.QPushButton("Load & Play")
        self.btn_play.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.btn_play.clicked.connect(self.on_play_clicked)
        playback_layout.addWidget(self.btn_play)
        
        self.btn_live = QtWidgets.QPushButton("Switch to Live")
        self.btn_live.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        self.btn_live.clicked.connect(self.switch_to_live)
        self.btn_live.hide() # Hidden by default, shown when in Playback
        playback_layout.addWidget(self.btn_live)
        
        self.btn_refresh = QtWidgets.QPushButton("↻")
        self.btn_refresh.setToolTip("Refresh File List")
        self.btn_refresh.setMaximumWidth(40)
        self.btn_refresh.setStyleSheet("padding: 10px;")
        self.btn_refresh.clicked.connect(self.update_file_list)
        playback_layout.addWidget(self.btn_refresh)
        
        control_layout.addLayout(playback_layout)
        
        self.layout.addLayout(control_layout)
        
        # status label
        self.status_label = QtWidgets.QLabel("Checking for EEG stream...")
        self.layout.addWidget(self.status_label)
        
        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.layout.addWidget(self.tabs)
        self.tabs.hide() 
        
        # Timer to check for stream
        self.check_stream_timer = QtCore.QTimer()
        self.check_stream_timer.setInterval(1000)
        self.check_stream_timer.timeout.connect(self.check_stream)
        self.check_stream_timer.start()
        
        self.loaded = False

    def closeEvent(self, event):
        # Ensure we kill the subprocess when closing the window
        if self.playback_process:
            print("Terminating playback process...")
            self.playback_process.terminate()
        super().closeEvent(event)

    def update_file_list(self):
        self.combo_files.clear()
        
        start_dir = os.path.join(os.getcwd(), 'recordings')
        if not os.path.exists(start_dir):
            os.makedirs(start_dir)
            
        files = [f for f in os.listdir(start_dir) if f.endswith('.csv')]
        files.sort(reverse=True) # Show newest first
        self.combo_files.addItems(files)

    def on_play_clicked(self):
        filename = self.combo_files.currentText()
        if not filename:
            return
            
        start_dir = os.path.join(os.getcwd(), 'recordings')
        full_path = os.path.join(start_dir, filename)
        
        if os.path.exists(full_path):
            self.switch_to_playback(full_path)
        else:
            QtWidgets.QMessageBox.warning(self, "File Not Found", f"Could not find file: {filename}")

    def switch_to_playback(self, filename):
        print(f"Switching to static playback: {filename}")
        self.mode = "PLAYBACK"
        self.playback_file = filename
        self.setWindowTitle(f"Brainwave Master Dashboard (Playback: {os.path.basename(filename)})")
        
        # 1. Stop Live/Stream Checks
        self.check_stream_timer.stop()
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
        
        # 2. Update UI
        self.status_label.setText(f"Loaded Static Recording: {os.path.basename(filename)}")
        self.status_label.show()
        self.btn_record.setEnabled(False) # Disable recording in playback logic
        self.btn_record.setText("Recording Disabled")
        self.btn_record.setStyleSheet("background-color: #9E9E9E; color: white; padding: 10px;")
        
        # 3. Load Tabs (if not already loaded)
        if not self.loaded:
            self.load_tabs()
        
        # 4. Tell Tabs to Load Static File
        # Iterate through tabs and call `load_static_file` if it exists
        for i in range(self.tabs.count()):
            widget = self.tabs.widget(i)
            if hasattr(widget, 'load_static_file'):
                widget.load_static_file(filename)
                
        self.status_label.hide()
        self.tabs.show()
        
        # 5. UI Toggles
        self.btn_live.show()
        # Optional: Disable play button if we want to force switching to live first, 
        # but users might want to switch file-to-file directly.
        
    def switch_to_live(self):
        print("Switching BACK to Live Mode...")
        self.mode = "LIVE"
        self.setWindowTitle("Brainwave Master Dashboard (Live)")
        
        # 1. UI Updates
        self.btn_live.hide()
        self.status_label.setText("Resuming Live Stream...")
        self.status_label.show()
        
        self.btn_record.setEnabled(True)
        self.btn_record.setText("Start Recording")
        self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        
        # 2. Tell Tabs to Resume Stream
        # If tabs aren't loaded yet, we just start the timer, 
        # but if they ARE loaded, we need to call load_stream()
        
        if self.loaded:
            for i in range(self.tabs.count()):
                widget = self.tabs.widget(i)
                if hasattr(widget, 'load_stream'):
                    # Force reload of stream
                    widget.load_stream()
            self.status_label.hide()
        else:
            # If not loaded, restart the check timer
            self.check_stream_timer.start()

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
        
        streams = resolve_byprop('type', 'EEG', timeout=1)
            
        if streams:
            self.status_label.setText("EEG Stream Found! Loading Dashboards...")
            self.check_stream_timer.stop()
            self.load_tabs()
        else:
            self.status_label.setText("No EEG stream found. Please run 'muselsl stream'...")

    def load_tabs(self):
        try:
            # We instantiate the windows and add them as tabs.
            # QMainWindow can be added to QTabWidget.
            
            # Tab 1: Dashboard (Avg + Quadrants)
            quad = Quadrants()
            self.tabs.addTab(quad, "Quadrants")
            
            # Tab 2: Live Graph (FFT Trends)
            live = LiveMuseGraph()
            self.tabs.addTab(live, "Live Analysis")
            
            # If in Live mode, we need to explicitly start the streams
            if self.mode == "LIVE":
                quad.load_stream()
                live.load_stream()
                        
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
    window = MuseMasterLauncher()
    window.show()
    sys.exit(app.exec())
