# Muse EEG Master Dashboard

A comprehensive Python-based dashboard for visualizing, recording, and replaying EEG data from Muse headbands.

![Main Dashboard](https://imgur.com/H3EiLwk.png)
![Live Analysis](https://imgur.com/tQzOi6J.png)

## Overview

This application provides a powerful interface to analyze brainwave data in real-time. It connects to an LSL (Lab Streaming Layer) stream provided by tools like `muselsl` and offers a variety of visualization modules, from raw wave monitoring to processed band power trends (Alpha, Beta, Theta, Delta, Gamma).

Crucially, it supports **two modes of operation**:
1.  **Live Stream**: Visualize and record data directly from a Muse headband.
2.  **Playback Mode**: Replay previously recorded raw data as if it were a live stream to review sessions using the same visualization tools.

## Features

*   **Dual Modes**: Seamlessly switch between Live and Playback.
*   **Raw Data Recording**: Saves high-resolution raw EEG data (TP9, AF7, AF8, TP10) for future analysis or replay.
*   **Modular Visualizations**:
    *   **Master Dashboard**: Global averaging and quadrant-based views.
    *   **Live Analysis**: Real-time FFT-based band power trends.
    *   **Raw Monitor**: Inspect signal quality and raw voltage levels.
    *   **Quadrants**: brain activity heatmap by region.
    *   **Lazy Log**: Smoothed, slow-moving trend logs.
    *   **Envelopes**: Smoothed amplitude envelopes.
*   **Robust Connectivity**: "Hot-pluggable" architecture that waits for streams to appear.

## Prerequisites

### Hardware
*   **Muse Headband**: Muse 2 or Muse S recommended.
*   **Bluetooth Dongle** (Optional but recommended): BLED112 for stable connection on some OSs.

### Software
*   **Python 3.8+**
*   **muselsl**: This project *consumes* an LSL stream. You need `muselsl` (or BlueMuse on Windows) running separately to broadcast the data from the headset.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/muse-project.git
    cd muse-project
    ```

2.  Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Start the LSL Stream
Before running the dashboard in **Live Mode**, you must establish a connection to the headset.

Open a terminal and run:
```bash
muselsl stream
```
*Wait until you see "Stream Created" output.*

### 2. Run the Dashboard
Open a second terminal and launch the application:
```bash
python dashboard.py
```

### 3. Select Mode
A dialog will appear asking you to choose a mode:

*   **Live Stream**: Connects to the active `muselsl` stream.
    *   *Recording*: Click the "Start Recording" button to save raw data to the `recordings/` directory.
*   **Playback Recording**: Select a previously recorded `.csv` file.
    *   The app will launch a simulate stream (`muse_playback_lsl.py`) in the background and connect the dashboard to it.

### 4. Compare Recordings
The dashboard includes a powerful **Comparison Tool** to analyze two sessions side-by-side.

1.  Click the purple **"Compare"** button in the dashboard.
2.  **Select Files**: Choose two recordings to compare.
3.  **Visual Trends**: View high-contrast plots (Cyan vs Magenta) of band power over time.
4.  **Statistical Analysis**: Switch to the "Stats" tab to perform a Welch's t-test, calculating significance (p-value), mean differences, and viewing distribution histograms.

![Comparison Visuals](https://imgur.com/1pHNtC5.png)
![Comparison Stats](https://imgur.com/1ce8ErX.png)

## File Structure
*   `dashboard.py`: **Entry point**. Handles the main window, live/playback modes, and recording.
*   `compare_window.py`: The UI for comparing two recordings (visuals + stats).
*   `compare_stat.py`: Statistical logic (Welch's t-test, outlier removal) for the comparison tool.
*   `analysis_helper.py`: Shared signal processing logic (FFT/Band Power calculations).
*   `muse_recorder.py`: Logic for saving the raw LSL stream to CSV.
*   `muse_playback_lsl.py`: Script that acts as a mock device, replaying a CSV file to LSL.
*   `quadrants.py`: Visualization for quadrant-based brain activity.
*   `live_graph.py`: Real-time band power trend graph.

## Troubleshooting

*   **"No EEG stream found"**: Ensure `muselsl stream` is running in a separate terminal and the headset is connected.
*   **Playback connecting to Live device**: The app attempts to use unique stream names to prevent this. Ensure you launch Playback mode from the `muse_master.py` dialog, not manually.
*   **Qt/Alignment Errors**: Ensure you have a compatible version of PyQt/PySide. This project is tested with standard `PyQt6` or `PyQt5`.

## License

[MIT License](LICENSE)
