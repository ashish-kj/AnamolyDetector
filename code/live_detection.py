#!/usr/bin/env python3
"""
Live Pipeline Leak Detection System with Web Dashboard
=====================================================

This system streams stored CSV data to simulate live pipeline monitoring
and provides real-time leak detection with a web-based dashboard.

Author: Pipeline Monitoring System
Version: 1.0
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pipeline_leak_detector import PipelineLeakDetector
except ImportError:
    # Fallback if running from different directory
    import importlib.util
    spec = importlib.util.spec_from_file_location("pipeline_leak_detector", 
                                                 os.path.join(os.path.dirname(__file__), "pipeline_leak_detector.py"))
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    PipelineLeakDetector = pipeline_module.PipelineLeakDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveDataStreamer:
    """Streams stored CSV data to simulate live pipeline monitoring"""
    
    def __init__(self, data_directory: str = "TestData"):
        self.data_directory = data_directory
        self.csv_files = self._get_csv_files()
        self.current_file_index = 0
        self.current_row_index = 0
        self.streaming = False
        self.stream_interval = 1.0  # seconds between data points
        self.leak_detector = PipelineLeakDetector()
        
        # Data buffers for real-time analysis
        self.data_buffer = []
        self.buffer_size = 50  # Keep last 50 samples for analysis
        self.distances = None
        
        # Anomaly storage
        self.anomaly_log_file = f"anomaly_log_{datetime.now().strftime('%Y%m%d')}.json"
        self.anomaly_storage = []
        self._load_existing_anomalies()
        
        logger.info(f"Initialized LiveDataStreamer with {len(self.csv_files)} files")
    
    def _get_csv_files(self) -> List[str]:
        """Get list of CSV files sorted by timestamp in filename"""
        if not os.path.exists(self.data_directory):
            logger.error(f"Data directory {self.data_directory} not found")
            return []
        
        csv_files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
        # Sort by timestamp in filename
        csv_files.sort()
        return [os.path.join(self.data_directory, f) for f in csv_files]
    
    def _extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract timestamp from DAS filename format"""
        try:
            # Expected format: DAS_Test_20250805_100000_325(36991)_Raw.csv
            parts = os.path.basename(filename).split('_')
            if len(parts) >= 4:
                date_str = parts[2]  # 20250805
                time_str = parts[3]  # 100000
                
                # Parse date and time
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])
                
                return datetime(year, month, day, hour, minute, second)
        except Exception as e:
            logger.warning(f"Could not extract timestamp from {filename}: {e}")
        
        return None
    
    def _calculate_stream_interval(self) -> float:
        """Calculate streaming interval based on file timestamps"""
        if len(self.csv_files) < 2:
            return 1.0
        
        timestamps = []
        for file_path in self.csv_files[:5]:  # Check first 5 files
            ts = self._extract_timestamp_from_filename(file_path)
            if ts:
                timestamps.append(ts)
        
        if len(timestamps) >= 2:
            # Calculate average interval between files
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            logger.info(f"Calculated stream interval: {avg_interval} seconds")
            return max(0.1, min(avg_interval, 10.0))  # Clamp between 0.1 and 10 seconds
        
        return 1.0
    
    def _load_existing_anomalies(self):
        """Load existing anomalies from log file"""
        if os.path.exists(self.anomaly_log_file):
            try:
                with open(self.anomaly_log_file, 'r') as f:
                    self.anomaly_storage = json.load(f)
                logger.info(f"Loaded {len(self.anomaly_storage)} existing anomalies")
            except Exception as e:
                logger.error(f"Error loading anomalies: {e}")
                self.anomaly_storage = []
    
    def _save_anomaly(self, anomaly):
        """Save anomaly to local storage"""
        try:
            # Add to memory storage
            self.anomaly_storage.append(anomaly)
            
            # Save to file
            with open(self.anomaly_log_file, 'w') as f:
                json.dump(self.anomaly_storage, f, indent=2, default=str)
            
            # Keep only last 10000 anomalies to prevent file from getting too large
            if len(self.anomaly_storage) > 10000:
                self.anomaly_storage = self.anomaly_storage[-10000:]
                with open(self.anomaly_log_file, 'w') as f:
                    json.dump(self.anomaly_storage, f, indent=2, default=str)
                    
        except Exception as e:
            logger.error(f"Error saving anomaly: {e}")
    
    def start_streaming(self) -> None:
        """Start streaming data"""
        if self.streaming:
            logger.warning("Streaming already started")
            return
        
        self.streaming = True
        self.stream_interval = self._calculate_stream_interval()
        logger.info(f"Starting data stream with interval: {self.stream_interval}s")
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._stream_data, daemon=True)
        self.stream_thread.start()
        logger.info("Started streaming thread")
    
    def stop_streaming(self) -> None:
        """Stop streaming data"""
        self.streaming = False
        logger.info("Stopped data streaming")
        
        # Reset streaming state
        self.current_file_index = 0
        self.current_row_index = 0
    
    def _stream_data(self) -> None:
        """Main streaming loop"""
        logger.info("Starting data streaming loop")
        while self.streaming and self.current_file_index < len(self.csv_files):
            # Check streaming status at the start of each iteration
            if not self.streaming:
                logger.info("Streaming stopped by user")
                break
                
            try:
                # Load current file if needed
                current_file = self.csv_files[self.current_file_index]
                
                if self.current_row_index == 0:
                    logger.info(f"Loading file: {os.path.basename(current_file)}")
                    df = pd.read_csv(current_file)
                    self.current_data = df
                    
                    # Extract distances from column names
                    distance_cols = [col for col in df.columns if col != 'Time(ms)/Distance(m)']
                    self.distances = np.array([float(col) for col in distance_cols])
                
                # Get current row
                if self.current_row_index < len(self.current_data):
                    row = self.current_data.iloc[self.current_row_index]
                    timestamp = datetime.now()
                    
                    # Extract measurement data
                    measurements = row.iloc[1:].values  # Skip time column
                    
                    # Create data point
                    data_point = {
                        'timestamp': timestamp.isoformat(),
                        'file_index': self.current_file_index,
                        'filename': os.path.basename(current_file),
                        'row_index': self.current_row_index,
                        'measurements': measurements.tolist(),
                        'distances': self.distances.tolist() if self.distances is not None else []
                    }
                    
                    # Add to buffer
                    self.data_buffer.append(data_point)
                    if len(self.data_buffer) > self.buffer_size:
                        self.data_buffer.pop(0)
                    
                    # Emit data to web clients
                    socketio.emit('new_data', data_point)
                    
                    # Check for anomalies if we have enough data
                    if len(self.data_buffer) >= 10:  # Need at least 10 samples
                        anomalies = self._detect_anomalies()
                        if anomalies:
                            for anomaly in anomalies:
                                # Save to local storage
                                self._save_anomaly(anomaly)
                                
                                # Only emit high-confidence anomalies to reduce noise
                                if anomaly['confidence'] >= 0.3:  # Minimum threshold for display
                                    socketio.emit('anomaly_detected', anomaly)
                                    logger.warning(f"Anomaly detected: {anomaly['type']} at {anomaly['position_km']:.3f}km (confidence: {anomaly['confidence']:.3f})")
                    
                    self.current_row_index += 1
                else:
                    # Move to next file
                    self.current_file_index += 1
                    self.current_row_index = 0
                    logger.info(f"Completed file {self.current_file_index}/{len(self.csv_files)}")
                
                    time.sleep(self.stream_interval)
                    
                    # Check if streaming was stopped during sleep
                    if not self.streaming:
                        logger.info("Streaming stopped during sleep")
                        break
                    
                except Exception as e:
                    logger.error(f"Error in streaming loop: {e}")
                    time.sleep(1.0)
        
        logger.info("Streaming completed")
        self.streaming = False
    
    def _detect_anomalies(self) -> List[Dict]:
        """Detect anomalies in current data buffer"""
        if len(self.data_buffer) < 10:
            return []
        
        try:
            # Extract recent measurements for analysis
            recent_data = np.array([point['measurements'] for point in self.data_buffer[-10:]])
            
            # Quick anomaly detection using statistical methods
            anomalies = []
            
            # Method 1: Z-score based detection (more selective)
            z_scores = np.abs((recent_data[-1] - np.mean(recent_data[:-1], axis=0)) / 
                            (np.std(recent_data[:-1], axis=0) + 1e-6))
            
            # Find locations with high z-scores (increased threshold)
            anomaly_indices = np.where(z_scores > 4.0)[0]
            
            for idx in anomaly_indices:
                if idx < len(self.distances):
                    anomaly = {
                        'timestamp': self.data_buffer[-1]['timestamp'],
                        'type': 'Statistical Anomaly',
                        'position_m': self.distances[idx],
                        'position_km': self.distances[idx] / 1000,
                        'confidence': min(1.0, z_scores[idx] / 6.0),
                        'severity': 'HIGH' if z_scores[idx] > 6.0 else 'MEDIUM' if z_scores[idx] > 5.0 else 'LOW',
                        'z_score': float(z_scores[idx]),
                        'measurement_value': float(recent_data[-1][idx]),
                        'baseline_mean': float(np.mean(recent_data[:-1], axis=0)[idx]),
                        'reason': f'Z-score of {z_scores[idx]:.2f} exceeds threshold of 4.0'
                    }
                    anomalies.append(anomaly)
            
            # Method 2: Sudden amplitude change detection
            if len(self.data_buffer) >= 5:
                recent_5 = np.array([point['measurements'] for point in self.data_buffer[-5:]])
                amplitude_changes = np.abs(recent_5[-1] - np.mean(recent_5[:-1], axis=0))
                threshold = np.std(recent_5[:-1], axis=0) * 3.5  # More selective threshold
                
                sudden_change_indices = np.where(amplitude_changes > threshold)[0]
                
                for idx in sudden_change_indices:
                    if idx < len(self.distances) and idx not in anomaly_indices:
                        change_magnitude = amplitude_changes[idx] / (threshold[idx] + 1e-6)
                        anomaly = {
                            'timestamp': self.data_buffer[-1]['timestamp'],
                            'type': 'Sudden Amplitude Change',
                            'position_m': self.distances[idx],
                            'position_km': self.distances[idx] / 1000,
                            'confidence': min(1.0, change_magnitude / 4.0),
                            'severity': 'HIGH' if change_magnitude > 5.0 else 'MEDIUM' if change_magnitude > 4.0 else 'LOW',
                            'change_magnitude': float(change_magnitude),
                            'measurement_value': float(recent_5[-1][idx]),
                            'baseline_mean': float(np.mean(recent_5[:-1], axis=0)[idx]),
                            'reason': f'Sudden amplitude change of {change_magnitude:.2f}x baseline variation'
                        }
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return []
    
    def get_status(self) -> Dict:
        """Get current streaming status"""
        return {
            'streaming': self.streaming,
            'current_file': os.path.basename(self.csv_files[self.current_file_index]) if self.current_file_index < len(self.csv_files) else None,
            'current_file_index': self.current_file_index,
            'total_files': len(self.csv_files),
            'current_row': self.current_row_index,
            'buffer_size': len(self.data_buffer),
            'stream_interval': self.stream_interval
        }

# Flask application setup
import os
template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
app = Flask(__name__, template_folder=template_folder)
app.config['SECRET_KEY'] = 'pipeline_monitoring_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global streamer instance
streamer = None

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get streaming status"""
    if streamer:
        return jsonify(streamer.get_status())
    return jsonify({'error': 'Streamer not initialized'})

@app.route('/api/start', methods=['POST'])
def start_streaming():
    """Start data streaming"""
    global streamer
    if not streamer:
        streamer = LiveDataStreamer()
    
    if not streamer.streaming:
        streamer.start_streaming()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/api/stop', methods=['POST'])
def stop_streaming():
    """Stop data streaming"""
    if streamer and streamer.streaming:
        streamer.stop_streaming()
        logger.info("Stop command received via API")
        return jsonify({'status': 'stopped', 'message': 'Streaming stopped successfully'})
    return jsonify({'status': 'not_running', 'message': 'No active streaming to stop'})

@app.route('/api/anomalies')
def get_anomalies():
    """Get anomaly statistics and recent anomalies"""
    if not streamer:
        return jsonify({'error': 'Streamer not initialized'})
    
    total_anomalies = len(streamer.anomaly_storage)
    recent_anomalies = streamer.anomaly_storage[-100:] if streamer.anomaly_storage else []
    
    # Calculate statistics
    if streamer.anomaly_storage:
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        type_counts = {}
        
        for anomaly in streamer.anomaly_storage:
            severity_counts[anomaly.get('severity', 'LOW')] += 1
            anomaly_type = anomaly.get('type', 'Unknown')
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
    else:
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        type_counts = {}
    
    return jsonify({
        'total_anomalies': total_anomalies,
        'recent_anomalies': recent_anomalies,
        'severity_counts': severity_counts,
        'type_counts': type_counts,
        'log_file': streamer.anomaly_log_file
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('status', {'message': 'Connected to pipeline monitoring system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

def create_dashboard_template():
    """Create the HTML dashboard template"""
    # Create template directory relative to the script's parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, '..', 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Leak Detection - Live Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .controls {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .status-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .status-card h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .status-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            word-wrap: break-word;
            overflow-wrap: break-word;
            max-width: 100%;
            line-height: 1.2;
        }
        .status-value.filename {
            font-size: 14px;
            font-weight: normal;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .alerts-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        .alert {
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 5px solid;
        }
        .alert-high {
            background-color: #ffebee;
            border-left-color: #f44336;
        }
        .alert-medium {
            background-color: #fff3e0;
            border-left-color: #ff9800;
        }
        .alert-low {
            background-color: #f3e5f5;
            border-left-color: #9c27b0;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        .btn-primary {
            background-color: #667eea;
            color: white;
        }
        .btn-danger {
            background-color: #f44336;
            color: white;
        }
        .btn:hover {
            opacity: 0.8;
        }
        .streaming-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-left: 10px;
        }
        .streaming-active {
            background-color: #4caf50;
            animation: pulse 2s infinite;
        }
        .streaming-inactive {
            background-color: #f44336;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛢️ Pipeline Leak Detection System</h1>
        <p>Live DAS Monitoring Dashboard</p>
    </div>

    <div class="controls">
        <button id="startBtn" class="btn btn-primary" onclick="startStreaming()">Start Monitoring</button>
        <button id="stopBtn" class="btn btn-danger" onclick="stopStreaming()">Stop Monitoring</button>
        <span id="streamingStatus">Status: Not Connected</span>
        <span id="streamingIndicator" class="streaming-indicator streaming-inactive"></span>
        
        <div style="margin-top: 15px;">
            <label><strong>Alert Filters:</strong></label>
            <label><input type="checkbox" id="filterHigh" checked> HIGH Severity</label>
            <label><input type="checkbox" id="filterMedium" checked> MEDIUM Severity</label>
            <label><input type="checkbox" id="filterLow"> LOW Severity</label>
            <label style="margin-left: 20px;"><strong>Min Confidence:</strong></label>
            <input type="range" id="confidenceSlider" min="0" max="1" step="0.1" value="0.5">
            <span id="confidenceValue">0.5</span>
        </div>
    </div>

    <div class="status-grid">
        <div class="status-card">
            <h3>Current File</h3>
            <div id="currentFile" class="status-value filename">-</div>
        </div>
        <div class="status-card">
            <h3>Progress</h3>
            <div id="progress" class="status-value">0/0</div>
        </div>
        <div class="status-card">
            <h3>Data Points</h3>
            <div id="dataPoints" class="status-value">0</div>
        </div>
        <div class="status-card">
            <h3>Anomalies (Shown/Total)</h3>
            <div id="anomalyCount" class="status-value">0/0</div>
        </div>
    </div>

    <div class="chart-container">
        <h3>Live Pipeline Signal</h3>
        <div id="signalChart" style="height: 400px;"></div>
    </div>

    <div class="alerts-container">
        <h3>🚨 Anomaly Alerts</h3>
        <div id="alertsList"></div>
    </div>

    <script>
        const socket = io();
        let signalData = [];
        let anomalies = [];
        let allAnomalies = []; // Store all anomalies for filtering
        let maxDataPoints = 100;
        
        // Filtering settings
        let filterSettings = {
            showHigh: true,
            showMedium: true,
            showLow: false,
            minConfidence: 0.5
        };

        // Initialize chart
        let signalTrace = {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines',
            name: 'Pipeline Signal',
            line: {color: '#667eea'}
        };

        let layout = {
            title: 'Live Pipeline Acoustic Signal',
            xaxis: {title: 'Time'},
            yaxis: {title: 'Signal Amplitude'},
            showlegend: true
        };

        Plotly.newPlot('signalChart', [signalTrace], layout);

        // Socket event handlers
        socket.on('connect', function() {
            document.getElementById('streamingStatus').textContent = 'Status: Connected';
            updateStatus();
        });

        socket.on('new_data', function(data) {
            // Update signal chart with average signal
            const avgSignal = data.measurements.reduce((a, b) => a + b, 0) / data.measurements.length;
            const timestamp = new Date(data.timestamp);
            
            signalTrace.x.push(timestamp);
            signalTrace.y.push(avgSignal);
            
            // Keep only recent data points
            if (signalTrace.x.length > maxDataPoints) {
                signalTrace.x.shift();
                signalTrace.y.shift();
            }
            
            Plotly.redraw('signalChart');
            
            // Update data points counter
            document.getElementById('dataPoints').textContent = signalData.length + 1;
            signalData.push(data);
        });

        socket.on('anomaly_detected', function(anomaly) {
            allAnomalies.push(anomaly);
            
            // Check if anomaly passes filters
            if (shouldShowAnomaly(anomaly)) {
                anomalies.push(anomaly);
                addAnomalyAlert(anomaly);
            }
            
            // Update counts (show filtered vs total)
            document.getElementById('anomalyCount').textContent = `${anomalies.length}/${allAnomalies.length}`;
        });
        
        function shouldShowAnomaly(anomaly) {
            // Check severity filter
            if (anomaly.severity === 'HIGH' && !filterSettings.showHigh) return false;
            if (anomaly.severity === 'MEDIUM' && !filterSettings.showMedium) return false;
            if (anomaly.severity === 'LOW' && !filterSettings.showLow) return false;
            
            // Check confidence filter
            if (anomaly.confidence < filterSettings.minConfidence) return false;
            
            return true;
        }

        function addAnomalyAlert(anomaly) {
            const alertsList = document.getElementById('alertsList');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${anomaly.severity.toLowerCase()}`;
            
            const timestamp = new Date(anomaly.timestamp).toLocaleString();
            alertDiv.innerHTML = `
                <strong>${anomaly.severity} SEVERITY ANOMALY</strong><br>
                <strong>Type:</strong> ${anomaly.type}<br>
                <strong>Location:</strong> ${anomaly.position_km.toFixed(3)} km<br>
                <strong>Time:</strong> ${timestamp}<br>
                <strong>Confidence:</strong> ${(anomaly.confidence * 100).toFixed(1)}%<br>
                <strong>Reason:</strong> ${anomaly.reason}
            `;
            
            alertsList.insertBefore(alertDiv, alertsList.firstChild);
            
            // Keep only recent alerts
            while (alertsList.children.length > 10) {
                alertsList.removeChild(alertsList.lastChild);
            }
        }

        function startStreaming() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log('Started streaming:', data);
                    updateStatus();
                });
        }

        function stopStreaming() {
            console.log('Stop button clicked');
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log('Stop response:', data);
                    if (data.status === 'stopped') {
                        document.getElementById('streamingStatus').textContent = 'Status: Stopped';
                        document.getElementById('streamingIndicator').className = 'streaming-indicator streaming-inactive';
                    }
                    updateStatus();
                })
                .catch(error => {
                    console.error('Error stopping stream:', error);
                });
        }

        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error('Status error:', data.error);
                        return;
                    }
                    
                    updateFilename(data.current_file);
                    document.getElementById('progress').textContent = `${data.current_file_index}/${data.total_files}`;
                    
                    const indicator = document.getElementById('streamingIndicator');
                    const status = document.getElementById('streamingStatus');
                    
                    if (data.streaming) {
                        indicator.className = 'streaming-indicator streaming-active';
                        status.textContent = 'Status: Streaming';
                    } else {
                        indicator.className = 'streaming-indicator streaming-inactive';
                        status.textContent = 'Status: Stopped';
                    }
                });
        }

        // Filter control event handlers
        document.getElementById('filterHigh').addEventListener('change', function(e) {
            filterSettings.showHigh = e.target.checked;
            applyFilters();
        });
        
        document.getElementById('filterMedium').addEventListener('change', function(e) {
            filterSettings.showMedium = e.target.checked;
            applyFilters();
        });
        
        document.getElementById('filterLow').addEventListener('change', function(e) {
            filterSettings.showLow = e.target.checked;
            applyFilters();
        });
        
        document.getElementById('confidenceSlider').addEventListener('input', function(e) {
            filterSettings.minConfidence = parseFloat(e.target.value);
            document.getElementById('confidenceValue').textContent = e.target.value;
            applyFilters();
        });
        
        function applyFilters() {
            // Clear current display
            anomalies = [];
            const alertsList = document.getElementById('alertsList');
            alertsList.innerHTML = '';
            
            // Re-filter all anomalies
            allAnomalies.forEach(anomaly => {
                if (shouldShowAnomaly(anomaly)) {
                    anomalies.push(anomaly);
                    addAnomalyAlert(anomaly);
                }
            });
            
            // Update count
            document.getElementById('anomalyCount').textContent = `${anomalies.length}/${allAnomalies.length}`;
        }
        
        function updateFilename(filename) {
            // Truncate long filenames for display
            const maxLength = 25;
            if (filename && filename.length > maxLength) {
                const truncated = filename.substring(0, maxLength) + '...';
                document.getElementById('currentFile').textContent = truncated;
                document.getElementById('currentFile').title = filename; // Show full name on hover
            } else {
                document.getElementById('currentFile').textContent = filename || '-';
            }
        }

        // Update status periodically
        setInterval(updateStatus, 2000);
        updateStatus();
    </script>
</body>
</html>'''
    
    with open(os.path.join(template_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info("Created dashboard template")

def main():
    """Main function to run the live detection system"""
    print("🛢️ LIVE PIPELINE LEAK DETECTION SYSTEM")
    print("=" * 50)
    
    # Create dashboard template
    create_dashboard_template()
    
    # Initialize global streamer
    global streamer
    streamer = LiveDataStreamer()
    
    print(f"✅ System initialized with {len(streamer.csv_files)} data files")
    print(f"📊 Dashboard available at: http://localhost:5000")
    print(f"🔄 Stream interval: {streamer._calculate_stream_interval():.1f} seconds")
    print("\n🚀 Starting web server...")
    print("   - Open your browser to http://localhost:5000")
    print("   - Click 'Start Monitoring' to begin live detection")
    print("   - Press Ctrl+C to stop the server")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        if streamer:
            streamer.stop_streaming()

if __name__ == "__main__":
    main()
