# Pipeline Leak Detection System using DAS Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import find_peaks, butter, filtfilt, spectrogram
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class PipelineLeakDetector:
    def __init__(self, pipeline_length_km=40.88, sampling_rate_hz=1000):
        """
        Initialize Pipeline Leak Detector for DAS-based monitoring
        
        Args:
            pipeline_length_km: Total pipeline length in kilometers
            sampling_rate_hz: Data sampling rate in Hz
        """
        self.pipeline_length_km = pipeline_length_km
        self.sampling_rate_hz = sampling_rate_hz
        self.leak_signatures = []
        self.baseline_profile = None
        self.detection_results = {}
        
        # Pipeline-specific parameters
        self.leak_frequency_range = (1, 100)  # Hz - typical leak frequencies
        self.pressure_wave_speed = 1200  # m/s - typical for oil pipelines
        self.min_leak_duration = 5  # samples - minimum leak duration
        self.spatial_correlation_threshold = 0.7
        
    def load_pipeline_data(self, file_path):
        """Load DAS data for pipeline monitoring"""
        print(f"Loading pipeline data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Extract measurement columns (distance points along pipeline)
        distance_cols = [col for col in df.columns if col != 'Time(ms)/Distance(m)']
        data = df[distance_cols].values
        
        # Convert column names to distances in meters
        distances = np.array([float(col) for col in distance_cols])
        
        print(f"Pipeline data loaded: {data.shape[0]} time samples, {len(distances)} spatial points")
        print(f"Pipeline coverage: {distances[0]:.1f}m to {distances[-1]:.1f}m ({(distances[-1]-distances[0])/1000:.1f} km)")
        
        return data, distances
    
    def preprocess_for_leaks(self, data, method='leak_optimized'):
        """Preprocess data specifically for leak detection"""
        print(f"Preprocessing data for leak detection using {method} method...")
        
        if method == 'leak_optimized':
            # 1. Remove DC component (baseline drift)
            data_processed = data - np.mean(data, axis=0)
            
            # 2. Apply bandpass filter for leak frequencies
            nyquist = self.sampling_rate_hz / 2
            low_freq = self.leak_frequency_range[0] / nyquist
            high_freq = min(self.leak_frequency_range[1] / nyquist, 0.95)
            
            if data.shape[0] > 10:  # Need enough samples for filtering
                b, a = butter(4, [low_freq, high_freq], btype='band')
                for i in range(data_processed.shape[1]):
                    data_processed[:, i] = filtfilt(b, a, data_processed[:, i])
            
            # 3. Normalize by spatial variance to handle distance-dependent attenuation
            spatial_std = np.std(data_processed, axis=0)
            spatial_std[spatial_std == 0] = 1  # Avoid division by zero
            data_processed = data_processed / spatial_std
            
        elif method == 'differential':
            # Differential processing to highlight changes
            data_processed = np.diff(data, axis=0)
            # Pad to maintain original shape
            data_processed = np.vstack([data_processed, data_processed[-1:]])
            
        else:
            # Standard normalization
            scaler = StandardScaler()
            data_processed = scaler.fit_transform(data)
            
        return data_processed
    
    def detect_pressure_waves(self, data, distances):
        """Detect pressure wave signatures characteristic of leaks"""
        print("Detecting pressure wave signatures...")
        
        leak_candidates = []
        wave_velocities = []
        
        # Look for pressure waves propagating along the pipeline
        for t in range(5, data.shape[0] - 5):  # Leave buffer for analysis
            time_slice = data[t-2:t+3, :]  # 5-sample window around time t
            
            # Calculate spatial gradient to find wave fronts
            spatial_gradient = np.gradient(np.mean(time_slice, axis=0))
            
            # Find peaks in spatial gradient (potential wave fronts)
            peaks, properties = find_peaks(np.abs(spatial_gradient), 
                                         height=np.std(spatial_gradient) * 2,
                                         distance=10)  # Minimum 10 points apart
            
            if len(peaks) >= 2:  # Need at least 2 peaks for wave velocity calculation
                # Calculate apparent wave velocity
                for i in range(len(peaks) - 1):
                    distance_diff = distances[peaks[i+1]] - distances[peaks[i]]
                    # Assume 1 sample time difference (can be refined)
                    apparent_velocity = abs(distance_diff) * self.sampling_rate_hz
                    
                    # Check if velocity is reasonable for pressure waves
                    if 500 <= apparent_velocity <= 2000:  # m/s - reasonable range
                        leak_candidates.append({
                            'time': t,
                            'position_m': distances[peaks[i]],
                            'velocity_ms': apparent_velocity,
                            'amplitude': properties['peak_heights'][i] if i < len(properties['peak_heights']) else 0,
                            'confidence': min(1.0, properties['peak_heights'][i] / np.max(spatial_gradient))
                        })
        
        return leak_candidates
    
    def detect_spatial_anomalies(self, data, distances):
        """Detect spatially localized anomalies that could indicate leaks"""
        print("Detecting spatial anomalies...")
        
        spatial_anomalies = []
        
        # Calculate spatial correlation matrix
        spatial_corr = np.corrcoef(data.T)  # Correlation between spatial points
        
        # Expected correlation should decrease with distance
        for i in range(len(distances)):
            # Get correlations with nearby points
            nearby_indices = np.where(np.abs(distances - distances[i]) < 1000)[0]  # Within 1km
            
            if len(nearby_indices) > 3:
                nearby_corr = spatial_corr[i, nearby_indices]
                expected_corr = np.mean(nearby_corr)
                
                # Look for points with unexpectedly low correlation (potential leak location)
                if expected_corr < self.spatial_correlation_threshold:
                    # Calculate anomaly strength
                    anomaly_strength = 1.0 - expected_corr
                    
                    # Additional check: look for amplitude anomalies at this location
                    location_signal = data[:, i]
                    signal_strength = np.std(location_signal)
                    mean_signal_strength = np.mean([np.std(data[:, j]) for j in nearby_indices])
                    
                    amplitude_anomaly = signal_strength / mean_signal_strength if mean_signal_strength > 0 else 1.0
                    
                    if amplitude_anomaly > 1.5:  # 50% higher than nearby points
                        spatial_anomalies.append({
                            'position_m': distances[i],
                            'correlation_anomaly': anomaly_strength,
                            'amplitude_anomaly': amplitude_anomaly,
                            'confidence': min(1.0, (anomaly_strength + amplitude_anomaly - 1.0) / 2.0)
                        })
        
        return spatial_anomalies
    
    def detect_frequency_anomalies(self, data, distances):
        """Detect frequency-domain anomalies characteristic of leaks"""
        print("Analyzing frequency domain for leak signatures...")
        
        frequency_anomalies = []
        
        # Analyze frequency content at each spatial location
        for i in range(0, len(distances), 50):  # Sample every 50th point for efficiency
            location_signal = data[:, i]
            
            if len(location_signal) >= 64:  # Need enough samples for FFT
                # Compute power spectral density
                freqs, psd = signal.welch(location_signal, fs=self.sampling_rate_hz, 
                                        nperseg=min(64, len(location_signal)//4))
                
                # Focus on leak-relevant frequencies
                leak_freq_mask = (freqs >= self.leak_frequency_range[0]) & (freqs <= self.leak_frequency_range[1])
                leak_power = np.sum(psd[leak_freq_mask])
                total_power = np.sum(psd)
                
                # Calculate leak frequency ratio
                leak_ratio = leak_power / total_power if total_power > 0 else 0
                
                # Check for dominant frequencies in leak range
                if leak_ratio > 0.3:  # 30% of power in leak frequencies
                    # Find dominant frequency
                    dominant_freq_idx = np.argmax(psd[leak_freq_mask])
                    dominant_freq = freqs[leak_freq_mask][dominant_freq_idx]
                    
                    frequency_anomalies.append({
                        'position_m': distances[i],
                        'dominant_frequency_hz': dominant_freq,
                        'leak_power_ratio': leak_ratio,
                        'confidence': min(1.0, leak_ratio)
                    })
        
        return frequency_anomalies
    
    def machine_learning_detection(self, data):
        """Use ML methods for leak detection"""
        print("Applying machine learning leak detection...")
        
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.05,  # Expect ~5% leaks
                                   random_state=42,
                                   n_estimators=100)
        
        # Fit and predict
        predictions = iso_forest.fit_predict(data)
        anomaly_scores = -iso_forest.decision_function(data)  # Higher = more anomalous
        
        # Extract anomalous time points
        anomaly_times = np.where(predictions == -1)[0]
        
        ml_detections = []
        for t in anomaly_times:
            # Find spatial location with highest anomaly contribution
            time_slice = data[t, :]
            max_anomaly_idx = np.argmax(np.abs(time_slice))
            
            ml_detections.append({
                'time': t,
                'spatial_index': max_anomaly_idx,
                'anomaly_score': anomaly_scores[t],
                'confidence': min(1.0, (anomaly_scores[t] - np.mean(anomaly_scores)) / np.std(anomaly_scores))
            })
        
        return ml_detections
    
    def ensemble_leak_detection(self, data, distances):
        """Combine multiple detection methods for robust leak detection"""
        print("\n" + "="*60)
        print("ENSEMBLE PIPELINE LEAK DETECTION")
        print("="*60)
        
        # Run all detection methods
        pressure_waves = self.detect_pressure_waves(data, distances)
        spatial_anomalies = self.detect_spatial_anomalies(data, distances)
        frequency_anomalies = self.detect_frequency_anomalies(data, distances)
        ml_detections = self.machine_learning_detection(data)
        
        # Combine results by spatial proximity
        all_detections = []
        
        # Process pressure wave detections
        for detection in pressure_waves:
            all_detections.append({
                'position_m': detection['position_m'],
                'type': 'pressure_wave',
                'confidence': detection['confidence'],
                'details': detection
            })
        
        # Process spatial anomalies
        for detection in spatial_anomalies:
            all_detections.append({
                'position_m': detection['position_m'],
                'type': 'spatial_anomaly',
                'confidence': detection['confidence'],
                'details': detection
            })
        
        # Process frequency anomalies
        for detection in frequency_anomalies:
            all_detections.append({
                'position_m': detection['position_m'],
                'type': 'frequency_anomaly',
                'confidence': detection['confidence'],
                'details': detection
            })
        
        # Process ML detections (need to convert spatial index to position)
        for detection in ml_detections:
            if detection['spatial_index'] < len(distances):
                all_detections.append({
                    'position_m': distances[detection['spatial_index']],
                    'type': 'ml_anomaly',
                    'confidence': max(0, detection['confidence']),
                    'details': detection
                })
        
        # Cluster nearby detections (within 100m)
        clustered_leaks = self.cluster_detections(all_detections, cluster_distance=100)
        
        return clustered_leaks, {
            'pressure_waves': pressure_waves,
            'spatial_anomalies': spatial_anomalies,
            'frequency_anomalies': frequency_anomalies,
            'ml_detections': ml_detections
        }
    
    def cluster_detections(self, detections, cluster_distance=100):
        """Cluster nearby detections to avoid duplicate leak reports"""
        if not detections:
            return []
        
        # Sort by position
        detections.sort(key=lambda x: x['position_m'])
        
        clustered = []
        current_cluster = [detections[0]]
        
        for detection in detections[1:]:
            # Check if close to current cluster
            cluster_center = np.mean([d['position_m'] for d in current_cluster])
            
            if abs(detection['position_m'] - cluster_center) <= cluster_distance:
                current_cluster.append(detection)
            else:
                # Finalize current cluster
                cluster_summary = self.summarize_cluster(current_cluster)
                clustered.append(cluster_summary)
                
                # Start new cluster
                current_cluster = [detection]
        
        # Don't forget the last cluster
        if current_cluster:
            cluster_summary = self.summarize_cluster(current_cluster)
            clustered.append(cluster_summary)
        
        return clustered
    
    def summarize_cluster(self, cluster):
        """Summarize a cluster of detections into a single leak report"""
        positions = [d['position_m'] for d in cluster]
        confidences = [d['confidence'] for d in cluster]
        types = [d['type'] for d in cluster]
        
        # Calculate cluster statistics
        avg_position = np.mean(positions)
        max_confidence = np.max(confidences)
        detection_count = len(cluster)
        
        # Calculate combined confidence (multiple methods increase confidence)
        type_diversity = len(set(types))
        combined_confidence = min(1.0, max_confidence * (1 + 0.2 * (type_diversity - 1)))
        
        return {
            'leak_position_m': avg_position,
            'leak_position_km': avg_position / 1000,
            'confidence': combined_confidence,
            'detection_count': detection_count,
            'detection_types': list(set(types)),
            'position_uncertainty_m': np.std(positions) if len(positions) > 1 else 0,
            'severity': 'HIGH' if combined_confidence > 0.8 else 'MEDIUM' if combined_confidence > 0.5 else 'LOW',
            'details': cluster
        }
    
    def visualize_leak_detection(self, data, distances, leak_results, filename_prefix="leak_detection"):
        """Create comprehensive visualizations for leak detection results"""
        print("Creating leak detection visualizations...")
        
        clustered_leaks, individual_results = leak_results
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Pipeline Leak Detection Results', fontsize=16, fontweight='bold')
        
        # 1. Pipeline overview with detected leaks
        ax1 = axes[0, 0]
        
        # Plot average signal along pipeline
        avg_signal = np.mean(data, axis=0)
        ax1.plot(distances/1000, avg_signal, 'b-', alpha=0.7, label='Average Signal')
        
        # Mark detected leaks
        for leak in clustered_leaks:
            color = 'red' if leak['severity'] == 'HIGH' else 'orange' if leak['severity'] == 'MEDIUM' else 'yellow'
            ax1.axvline(leak['leak_position_km'], color=color, linestyle='--', alpha=0.8, linewidth=2)
            ax1.text(leak['leak_position_km'], np.max(avg_signal) * 0.9, 
                    f"LEAK\n{leak['severity']}\n{leak['confidence']:.2f}", 
                    ha='center', va='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        ax1.set_xlabel('Pipeline Distance (km)')
        ax1.set_ylabel('Average Signal Amplitude')
        ax1.set_title('Pipeline Overview with Leak Locations')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Time-distance heatmap
        ax2 = axes[0, 1]
        
        # Sample data for visualization (every 10th spatial point)
        sample_indices = range(0, len(distances), 10)
        sample_distances = distances[sample_indices] / 1000
        sample_data = data[:, sample_indices]
        
        im = ax2.imshow(sample_data.T, aspect='auto', cmap='seismic', 
                       extent=[0, data.shape[0], sample_distances[-1], sample_distances[0]])
        ax2.set_xlabel('Time Sample')
        ax2.set_ylabel('Pipeline Distance (km)')
        ax2.set_title('Time-Distance Heatmap (Sampled)')
        plt.colorbar(im, ax=ax2, label='Signal Amplitude')
        
        # Mark leak positions
        for leak in clustered_leaks:
            ax2.axhline(leak['leak_position_km'], color='white', linestyle='-', linewidth=2)
        
        # 3. Detection method comparison
        ax3 = axes[1, 0]
        
        methods = ['Pressure Wave', 'Spatial Anomaly', 'Frequency Anomaly', 'ML Detection']
        method_counts = [
            len(individual_results['pressure_waves']),
            len(individual_results['spatial_anomalies']),
            len(individual_results['frequency_anomalies']),
            len(individual_results['ml_detections'])
        ]
        
        bars = ax3.bar(methods, method_counts, color=['red', 'orange', 'green', 'blue'], alpha=0.7)
        ax3.set_ylabel('Number of Detections')
        ax3.set_title('Detection Method Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, method_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        # 4. Leak summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if clustered_leaks:
            # Create summary table
            table_data = []
            headers = ['Position (km)', 'Severity', 'Confidence', 'Methods', 'Uncertainty (m)']
            
            for leak in clustered_leaks:
                row = [
                    f"{leak['leak_position_km']:.2f}",
                    leak['severity'],
                    f"{leak['confidence']:.3f}",
                    f"{leak['detection_count']} types",
                    f"±{leak['position_uncertainty_m']:.1f}"
                ]
                table_data.append(row)
            
            # Create table
            table = ax4.table(cellText=table_data, colLabels=headers, 
                            cellLoc='center', loc='center', bbox=[0, 0.3, 1, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Color code by severity
            for i, leak in enumerate(clustered_leaks):
                color = 'lightcoral' if leak['severity'] == 'HIGH' else 'lightyellow' if leak['severity'] == 'MEDIUM' else 'lightgreen'
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor(color)
            
            ax4.set_title('Detected Leaks Summary', pad=20)
        else:
            ax4.text(0.5, 0.5, 'No leaks detected', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=16, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            ax4.set_title('Leak Detection Results', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return f'{filename_prefix}_comprehensive.png'
    
    def generate_leak_report(self, leak_results, filename):
        """Generate a comprehensive leak detection report"""
        clustered_leaks, individual_results = leak_results
        
        report = []
        report.append("="*80)
        report.append("PIPELINE LEAK DETECTION REPORT")
        report.append("="*80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data File: {filename}")
        report.append(f"Pipeline Length: {self.pipeline_length_km:.2f} km")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY:")
        report.append("-" * 20)
        if clustered_leaks:
            high_severity = sum(1 for leak in clustered_leaks if leak['severity'] == 'HIGH')
            medium_severity = sum(1 for leak in clustered_leaks if leak['severity'] == 'MEDIUM')
            low_severity = sum(1 for leak in clustered_leaks if leak['severity'] == 'LOW')
            
            report.append(f"🚨 TOTAL LEAKS DETECTED: {len(clustered_leaks)}")
            report.append(f"   • HIGH severity: {high_severity}")
            report.append(f"   • MEDIUM severity: {medium_severity}")
            report.append(f"   • LOW severity: {low_severity}")
            
            if high_severity > 0:
                report.append("⚠️  IMMEDIATE ACTION REQUIRED for HIGH severity leaks!")
        else:
            report.append("✅ NO LEAKS DETECTED - Pipeline appears normal")
        
        report.append("")
        
        # Detailed Results
        if clustered_leaks:
            report.append("DETAILED LEAK LOCATIONS:")
            report.append("-" * 30)
            
            for i, leak in enumerate(clustered_leaks, 1):
                report.append(f"\nLEAK #{i}:")
                report.append(f"  Position: {leak['leak_position_km']:.3f} km (±{leak['position_uncertainty_m']:.1f}m)")
                report.append(f"  Severity: {leak['severity']}")
                report.append(f"  Confidence: {leak['confidence']:.3f}")
                report.append(f"  Detection Methods: {', '.join(leak['detection_types'])}")
                report.append(f"  Supporting Evidence: {leak['detection_count']} independent detections")
        
        # Method Performance
        report.append(f"\nDETECTION METHOD PERFORMANCE:")
        report.append("-" * 35)
        report.append(f"  Pressure Wave Analysis: {len(individual_results['pressure_waves'])} detections")
        report.append(f"  Spatial Correlation Analysis: {len(individual_results['spatial_anomalies'])} detections")
        report.append(f"  Frequency Domain Analysis: {len(individual_results['frequency_anomalies'])} detections")
        report.append(f"  Machine Learning Analysis: {len(individual_results['ml_detections'])} detections")
        
        # Recommendations
        report.append(f"\nRECOMMENDations:")
        report.append("-" * 15)
        if clustered_leaks:
            high_confidence_leaks = [l for l in clustered_leaks if l['confidence'] > 0.7]
            if high_confidence_leaks:
                report.append("1. IMMEDIATE INSPECTION required for high-confidence leak locations")
                for leak in high_confidence_leaks:
                    report.append(f"   • Inspect around {leak['leak_position_km']:.3f} km mark")
            
            report.append("2. Increase monitoring frequency in detected leak areas")
            report.append("3. Consider pressure testing in affected pipeline sections")
            report.append("4. Review maintenance records for detected locations")
        else:
            report.append("1. Continue normal monitoring schedule")
            report.append("2. System is operating within normal parameters")
        
        report.append("")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        report_filename = f"leak_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_filename, 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\n📄 Report saved as: {report_filename}")
        
        return report_filename
    
    def analyze_pipeline_file(self, file_path):
        """Complete pipeline leak analysis for a single file"""
        print(f"\n{'='*80}")
        print(f"PIPELINE LEAK ANALYSIS: {os.path.basename(file_path)}")
        print(f"{'='*80}")
        
        # Load data
        data, distances = self.load_pipeline_data(file_path)
        
        # Preprocess for leak detection
        processed_data = self.preprocess_for_leaks(data, method='leak_optimized')
        
        # Run ensemble detection
        leak_results = self.ensemble_leak_detection(processed_data, distances)
        
        # Create visualizations
        viz_file = self.visualize_leak_detection(processed_data, distances, leak_results, 
                                               filename_prefix=f"leak_analysis_{os.path.basename(file_path).split('.')[0]}")
        
        # Generate report
        report_file = self.generate_leak_report(leak_results, os.path.basename(file_path))
        
        return leak_results, viz_file, report_file

def main():
    """Main function for pipeline leak detection"""
    # Initialize detector
    detector = PipelineLeakDetector(pipeline_length_km=40.88, sampling_rate_hz=1000)
    
    # Analyze first file as example
    test_data_folder = "TestData"
    csv_files = [f for f in os.listdir(test_data_folder) if f.endswith('.csv')]
    
    if csv_files:
        # Analyze the first file
        first_file = os.path.join(test_data_folder, csv_files[0])
        print(f"🔍 Analyzing {first_file} for pipeline leaks...")
        
        leak_results, viz_file, report_file = detector.analyze_pipeline_file(first_file)
        
        print(f"\n✅ Pipeline leak analysis complete!")
        print(f"📊 Visualization: {viz_file}")
        print(f"📄 Report: {report_file}")
        
        # Quick summary
        clustered_leaks, _ = leak_results
        if clustered_leaks:
            print(f"\n🚨 ALERT: {len(clustered_leaks)} potential leak(s) detected!")
            for i, leak in enumerate(clustered_leaks, 1):
                print(f"   Leak {i}: {leak['leak_position_km']:.3f} km - {leak['severity']} severity")
        else:
            print(f"\n✅ No leaks detected - Pipeline appears normal")
            
    else:
        print("❌ No CSV files found in TestData folder!")

if __name__ == "__main__":
    main()
