# Pipeline Leak Detection System using DAS Technology

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A comprehensive real-time pipeline leak detection system using **Distributed Acoustic Sensing (DAS)** technology with fiber optic monitoring. This system provides advanced anomaly detection, live web dashboard, and real-time alerting for oil pipeline monitoring.

## 🌟 Features

### 🔍 **Advanced Leak Detection**
- **Multiple Detection Algorithms**: Statistical, ML-based, frequency domain, and physics-based methods
- **Real-time Processing**: Optimized for low-latency leak detection
- **High Accuracy**: Ensemble methods combining multiple detection approaches
- **Pipeline-Specific**: Tailored algorithms for oil pipeline leak signatures

### 🌐 **Live Web Dashboard**
- **Real-time Monitoring**: Live data streaming with WebSocket communication
- **Interactive Visualizations**: Professional charts and status indicators
- **Immediate Alerts**: Color-coded anomaly notifications with detailed context
- **Responsive Design**: Modern web interface suitable for control rooms

### 📊 **Comprehensive Analysis**
- **Data Exploration**: Temporal and spatial pattern analysis
- **Baseline Establishment**: Normal operation profiling
- **Performance Metrics**: Confidence scoring and severity classification
- **Detailed Reporting**: Comprehensive leak detection reports

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ashish-kj/AnamolyDetector.git
cd AnamolyDetector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Test the System
```bash
python test_system.py
```

### 4. Start Live Detection Dashboard
```bash
python code/live_detection.py
```
Then open your browser to: **http://localhost:5000**

## 📁 Project Structure

```
AnamolyDetector/
├── code/
│   ├── analyze.py                    # Basic data analysis
│   ├── pipeline_leak_detector.py     # Advanced leak detection
│   ├── live_detection.py            # Live web dashboard
│   ├── realtime_pipeline_monitor.py # Batch monitoring
│   └── data_explorer.py             # Comprehensive data exploration
├── TestData/                        # Sample DAS data files
│   ├── DAS_Test_*.csv               # Pipeline monitoring data
├── templates/                       # Web dashboard templates (auto-generated)
├── PIPELINE_LEAK_DETECTION_GUIDE.md # Comprehensive user guide
├── requirements.txt                 # Python dependencies
├── test_system.py                  # System validation
└── README.md                       # This file
```

## 🛠️ System Components

| Component | Purpose | Best For |
|-----------|---------|----------|
| `analyze.py` | Basic data analysis | First-time data exploration |
| `pipeline_leak_detector.py` | Specialized leak detection | Detailed leak analysis |
| `live_detection.py` | **Live web dashboard** | **Real-time operations** |
| `realtime_pipeline_monitor.py` | Batch monitoring | Processing multiple files |
| `data_explorer.py` | Pattern analysis | Understanding data characteristics |

## 🔧 Detection Methods

### 1. **Pressure Wave Detection**
- Identifies pressure waves traveling at pipeline velocity (~1200 m/s)
- Detects sudden pressure changes characteristic of leaks
- Validates wave propagation patterns

### 2. **Spatial Correlation Analysis**
- Analyzes correlation breakdown at leak locations
- Identifies points behaving differently from neighbors
- Detects localized disturbances

### 3. **Frequency Domain Analysis**
- Focuses on leak-specific frequencies (1-100 Hz)
- Power spectral density analysis
- Identifies elevated leak-frequency content

### 4. **Machine Learning Detection**
- Isolation Forest for outlier pattern detection
- Trained on normal operation baselines
- Complementary to physics-based methods

### 5. **Statistical Methods**
- Z-score and IQR-based anomaly detection
- Real-time threshold monitoring
- Confidence scoring and severity classification

## 📊 Live Dashboard Features

### Real-time Capabilities
- ✅ Live data streaming from DAS sensors
- ✅ Automatic time interval detection
- ✅ Real-time anomaly detection
- ✅ Immediate web-based alerts

### Professional Interface
- ✅ Modern, responsive web design
- ✅ Interactive charts and visualizations
- ✅ Status monitoring and progress tracking
- ✅ Color-coded alert management

### Alert System
- 🔴 **HIGH Severity**: Immediate investigation required
- 🟠 **MEDIUM Severity**: Increased monitoring needed
- 🟣 **LOW Severity**: Minor anomaly detected

## 🎯 Usage Examples

### Basic Data Analysis
```python
python code/analyze.py
```

### Advanced Leak Detection
```python
python code/pipeline_leak_detector.py
```

### Live Monitoring Dashboard
```python
python code/live_detection.py
# Open http://localhost:5000 in your browser
```

## 📈 Data Format

The system works with DAS (Distributed Acoustic Sensing) data in CSV format:
- **Time samples**: Typically 1000 measurements per file
- **Spatial points**: ~10,000 locations along pipeline (~40km coverage)
- **Sampling rate**: 1 Hz (configurable)
- **File format**: `DAS_Test_YYYYMMDD_HHMMSS_*.csv`

## 🔧 Configuration

### Key Parameters (adjustable in code)
```python
# Detection thresholds
ALERT_THRESHOLD = 0.6        # Confidence threshold for alerts
LEAK_FREQUENCY_RANGE = (1, 100)  # Hz - leak frequency range
PRESSURE_WAVE_SPEED = 1200   # m/s - typical for oil pipelines
SPATIAL_CLUSTERING = 100     # m - group nearby detections
```

## 🚨 Safety & Operational Notes

- ⚠️ This system is a **monitoring aid**, not a safety system
- 🔍 High-confidence detections should trigger immediate field investigation
- 📋 Always follow your company's leak response procedures
- 🔄 Regular calibration and validation are essential
- 📊 System performance should be validated with known events

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for oil pipeline monitoring using DAS technology
- Designed for real-time operational environments
- Incorporates industry best practices for leak detection
- Optimized for fiber optic acoustic sensing systems

## 📞 Support

For questions, issues, or feature requests:
- 📧 Create an issue in this repository
- 📖 Read the comprehensive guide: `PIPELINE_LEAK_DETECTION_GUIDE.md`
- 🧪 Run system tests: `python test_system.py`

---

**🛢️ Professional Pipeline Monitoring • 🔍 Real-time Leak Detection • 🌐 Modern Web Interface**
