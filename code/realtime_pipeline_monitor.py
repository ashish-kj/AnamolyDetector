# Real-time Pipeline Leak Monitoring System
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
from datetime import datetime, timedelta
import json
import warnings
from pipeline_leak_detector import PipelineLeakDetector
warnings.filterwarnings('ignore')

class RealtimePipelineMonitor:
    def __init__(self, pipeline_length_km=40.88, alert_threshold=0.7):
        """
        Real-time pipeline monitoring system
        
        Args:
            pipeline_length_km: Total pipeline length
            alert_threshold: Confidence threshold for leak alerts
        """
        self.pipeline_length_km = pipeline_length_km
        self.alert_threshold = alert_threshold
        self.detector = PipelineLeakDetector(pipeline_length_km)
        
        # Monitoring state
        self.monitoring_active = False
        self.alert_history = []
        self.baseline_established = False
        self.baseline_data = None
        
        # Alert configuration
        self.alert_cooldown = 300  # 5 minutes between repeat alerts for same location
        self.last_alerts = {}  # Track last alert time for each location
        
    def establish_baseline(self, baseline_files):
        """Establish baseline from normal operation data"""
        print("Establishing pipeline baseline from normal operation data...")
        
        baseline_data = []
        for file_path in baseline_files:
            if os.path.exists(file_path):
                data, distances = self.detector.load_pipeline_data(file_path)
                processed_data = self.detector.preprocess_for_leaks(data)
                baseline_data.append(processed_data)
                print(f"  Added baseline: {os.path.basename(file_path)}")
        
        if baseline_data:
            # Calculate statistical baseline
            combined_baseline = np.concatenate(baseline_data, axis=0)
            self.baseline_data = {
                'mean': np.mean(combined_baseline, axis=0),
                'std': np.std(combined_baseline, axis=0),
                'percentile_95': np.percentile(combined_baseline, 95, axis=0),
                'percentile_5': np.percentile(combined_baseline, 5, axis=0),
                'distances': distances
            }
            self.baseline_established = True
            print(f"✅ Baseline established from {len(baseline_data)} files")
        else:
            print("❌ No baseline files found!")
    
    def analyze_current_data(self, file_path):
        """Analyze current data against baseline and detect leaks"""
        if not os.path.exists(file_path):
            return None
            
        # Load and process current data
        data, distances = self.detector.load_pipeline_data(file_path)
        processed_data = self.detector.preprocess_for_leaks(data)
        
        # Run leak detection
        leak_results = self.detector.ensemble_leak_detection(processed_data, distances)
        clustered_leaks, individual_results = leak_results
        
        # Compare with baseline if available
        baseline_anomalies = []
        if self.baseline_established:
            current_stats = {
                'mean': np.mean(processed_data, axis=0),
                'std': np.std(processed_data, axis=0)
            }
            
            # Check for significant deviations from baseline
            mean_deviation = np.abs(current_stats['mean'] - self.baseline_data['mean'])
            std_threshold = self.baseline_data['std'] * 3  # 3-sigma threshold
            
            anomalous_locations = np.where(mean_deviation > std_threshold)[0]
            
            for loc_idx in anomalous_locations:
                if loc_idx < len(distances):
                    baseline_anomalies.append({
                        'position_m': distances[loc_idx],
                        'deviation_magnitude': mean_deviation[loc_idx] / std_threshold[loc_idx] if std_threshold[loc_idx] > 0 else 0,
                        'type': 'baseline_deviation'
                    })
        
        analysis_result = {
            'timestamp': datetime.now(),
            'filename': os.path.basename(file_path),
            'clustered_leaks': clustered_leaks,
            'individual_results': individual_results,
            'baseline_anomalies': baseline_anomalies,
            'total_detections': len(clustered_leaks)
        }
        
        return analysis_result
    
    def generate_alert(self, leak_info, analysis_result):
        """Generate alert for detected leak"""
        current_time = datetime.now()
        position_km = leak_info['leak_position_km']
        
        # Check alert cooldown
        location_key = f"{position_km:.2f}"
        if location_key in self.last_alerts:
            time_since_last = (current_time - self.last_alerts[location_key]).seconds
            if time_since_last < self.alert_cooldown:
                return None  # Skip alert due to cooldown
        
        # Create alert
        alert = {
            'alert_id': f"LEAK_{current_time.strftime('%Y%m%d_%H%M%S')}_{position_km:.3f}",
            'timestamp': current_time,
            'severity': leak_info['severity'],
            'confidence': leak_info['confidence'],
            'position_km': position_km,
            'position_uncertainty_m': leak_info['position_uncertainty_m'],
            'detection_methods': leak_info['detection_types'],
            'source_file': analysis_result['filename'],
            'requires_immediate_action': leak_info['severity'] == 'HIGH' and leak_info['confidence'] > 0.8
        }
        
        # Update last alert time
        self.last_alerts[location_key] = current_time
        
        # Add to alert history
        self.alert_history.append(alert)
        
        return alert
    
    def process_alerts(self, analysis_result):
        """Process and generate alerts for current analysis"""
        alerts = []
        
        for leak in analysis_result['clustered_leaks']:
            if leak['confidence'] >= self.alert_threshold:
                alert = self.generate_alert(leak, analysis_result)
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def save_alert_log(self, alerts):
        """Save alerts to log file"""
        if not alerts:
            return
            
        log_filename = f"pipeline_alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Load existing alerts if file exists
        existing_alerts = []
        if os.path.exists(log_filename):
            try:
                with open(log_filename, 'r') as f:
                    existing_alerts = json.load(f, default=str)
            except:
                existing_alerts = []
        
        # Add new alerts
        for alert in alerts:
            # Convert datetime to string for JSON serialization
            alert_copy = alert.copy()
            alert_copy['timestamp'] = alert['timestamp'].isoformat()
            existing_alerts.append(alert_copy)
        
        # Save updated alerts
        with open(log_filename, 'w') as f:
            json.dump(existing_alerts, f, indent=2, default=str)
        
        print(f"💾 Alerts saved to {log_filename}")
    
    def print_alerts(self, alerts):
        """Print alerts to console"""
        if not alerts:
            return
            
        print("\n" + "🚨" * 20 + " LEAK ALERTS " + "🚨" * 20)
        
        for alert in alerts:
            print(f"\n⚠️  ALERT: {alert['alert_id']}")
            print(f"   Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Location: {alert['position_km']:.3f} km (±{alert['position_uncertainty_m']:.1f}m)")
            print(f"   Severity: {alert['severity']} (Confidence: {alert['confidence']:.3f})")
            print(f"   Detection Methods: {', '.join(alert['detection_methods'])}")
            
            if alert['requires_immediate_action']:
                print(f"   🚨 IMMEDIATE ACTION REQUIRED! 🚨")
            
            print(f"   Source: {alert['source_file']}")
        
        print("\n" + "🚨" * 52)
    
    def create_monitoring_dashboard(self, analysis_results):
        """Create a monitoring dashboard visualization"""
        if not analysis_results:
            return
            
        latest_result = analysis_results[-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Pipeline Monitoring Dashboard - {latest_result["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=14, fontweight='bold')
        
        # 1. Current leak status
        ax1 = axes[0, 0]
        if latest_result['clustered_leaks']:
            positions = [leak['leak_position_km'] for leak in latest_result['clustered_leaks']]
            severities = [leak['severity'] for leak in latest_result['clustered_leaks']]
            confidences = [leak['confidence'] for leak in latest_result['clustered_leaks']]
            
            colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow'}
            for pos, sev, conf in zip(positions, severities, confidences):
                ax1.scatter(pos, conf, c=colors[sev], s=200, alpha=0.8, edgecolors='black')
                ax1.text(pos, conf + 0.05, f'{sev}\n{pos:.2f}km', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlim(0, self.pipeline_length_km)
        ax1.set_ylim(0, 1.1)
        ax1.set_xlabel('Pipeline Distance (km)')
        ax1.set_ylabel('Detection Confidence')
        ax1.set_title('Current Leak Detections')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.alert_threshold, color='red', linestyle='--', alpha=0.7, label=f'Alert Threshold ({self.alert_threshold})')
        ax1.legend()
        
        # 2. Alert history over time
        ax2 = axes[0, 1]
        if len(analysis_results) > 1:
            timestamps = [result['timestamp'] for result in analysis_results]
            detection_counts = [result['total_detections'] for result in analysis_results]
            
            ax2.plot(timestamps, detection_counts, 'b-o', markersize=4)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Number of Detections')
            ax2.set_title('Detection History')
            ax2.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor trend analysis', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Detection History')
        
        # 3. Pipeline overview
        ax3 = axes[1, 0]
        pipeline_segments = np.linspace(0, self.pipeline_length_km, 100)
        
        # Color code pipeline based on recent detections
        segment_colors = ['green'] * len(pipeline_segments)  # Default: normal
        
        for leak in latest_result['clustered_leaks']:
            # Find nearest segment
            nearest_segment = np.argmin(np.abs(pipeline_segments - leak['leak_position_km']))
            if leak['severity'] == 'HIGH':
                segment_colors[nearest_segment] = 'red'
            elif leak['severity'] == 'MEDIUM':
                segment_colors[nearest_segment] = 'orange'
            else:
                segment_colors[nearest_segment] = 'yellow'
        
        for i in range(len(pipeline_segments)-1):
            ax3.barh(0, pipeline_segments[i+1] - pipeline_segments[i], 
                    left=pipeline_segments[i], color=segment_colors[i], height=0.5)
        
        ax3.set_xlim(0, self.pipeline_length_km)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_xlabel('Pipeline Distance (km)')
        ax3.set_title('Pipeline Status Overview')
        ax3.set_yticks([])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Normal'),
            Patch(facecolor='yellow', label='Low Risk'),
            Patch(facecolor='orange', label='Medium Risk'),
            Patch(facecolor='red', label='High Risk')
        ]
        ax3.legend(handles=legend_elements, loc='upper right')
        
        # 4. System status
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # System status text
        status_text = f"""
SYSTEM STATUS
─────────────────
Monitoring: {'ACTIVE' if self.monitoring_active else 'INACTIVE'}
Baseline: {'ESTABLISHED' if self.baseline_established else 'NOT SET'}
Pipeline Length: {self.pipeline_length_km:.1f} km
Alert Threshold: {self.alert_threshold:.2f}

CURRENT ANALYSIS
─────────────────
File: {latest_result['filename']}
Timestamp: {latest_result['timestamp'].strftime('%H:%M:%S')}
Total Detections: {latest_result['total_detections']}
High Severity: {sum(1 for leak in latest_result['clustered_leaks'] if leak['severity'] == 'HIGH')}

ALERT SUMMARY
─────────────────
Total Alerts Today: {len([a for a in self.alert_history if a['timestamp'].date() == datetime.now().date()])}
Active Locations: {len(self.last_alerts)}
Last Alert: {max([a['timestamp'] for a in self.alert_history]).strftime('%H:%M:%S') if self.alert_history else 'None'}
        """
        
        ax4.text(0.05, 0.95, status_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_filename = f"pipeline_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(dashboard_filename, dpi=200, bbox_inches='tight')
        plt.show()
        
        return dashboard_filename
    
    def monitor_directory(self, data_directory, check_interval=60):
        """Monitor directory for new data files and analyze them"""
        print(f"🔍 Starting real-time monitoring of {data_directory}")
        print(f"   Check interval: {check_interval} seconds")
        print(f"   Alert threshold: {self.alert_threshold}")
        print("   Press Ctrl+C to stop monitoring\n")
        
        self.monitoring_active = True
        analysis_results = []
        processed_files = set()
        
        try:
            while self.monitoring_active:
                # Check for new CSV files
                csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
                new_files = [f for f in csv_files if f not in processed_files]
                
                for filename in new_files:
                    file_path = os.path.join(data_directory, filename)
                    print(f"📁 Processing new file: {filename}")
                    
                    try:
                        # Analyze the file
                        analysis_result = self.analyze_current_data(file_path)
                        
                        if analysis_result:
                            analysis_results.append(analysis_result)
                            
                            # Process alerts
                            alerts = self.process_alerts(analysis_result)
                            
                            if alerts:
                                self.print_alerts(alerts)
                                self.save_alert_log(alerts)
                            else:
                                print(f"✅ {filename}: No leaks detected (Normal operation)")
                            
                            # Update dashboard every few files or if alerts generated
                            if len(analysis_results) % 3 == 0 or alerts:
                                dashboard_file = self.create_monitoring_dashboard(analysis_results)
                                print(f"📊 Dashboard updated: {dashboard_file}")
                            
                            processed_files.add(filename)
                            
                        else:
                            print(f"❌ Failed to analyze {filename}")
                            
                    except Exception as e:
                        print(f"❌ Error processing {filename}: {str(e)}")
                
                # Wait before next check
                if not new_files:
                    print(f"⏳ No new files. Next check in {check_interval} seconds...")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {str(e)}")
        finally:
            self.monitoring_active = False
            print("📊 Final dashboard...")
            if analysis_results:
                self.create_monitoring_dashboard(analysis_results)

def main():
    """Main function for real-time pipeline monitoring"""
    print("🛢️  REAL-TIME PIPELINE LEAK MONITORING SYSTEM")
    print("=" * 50)
    
    # Initialize monitor
    monitor = RealtimePipelineMonitor(pipeline_length_km=40.88, alert_threshold=0.6)
    
    # Check if we should establish baseline
    data_directory = "TestData"
    csv_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.csv')]
    
    if len(csv_files) >= 2:
        print("📊 Establishing baseline from existing files...")
        baseline_files = csv_files[:2]  # Use first 2 files as baseline
        monitor.establish_baseline(baseline_files)
        
        # Analyze remaining files as if they were new
        remaining_files = csv_files[2:]
        if remaining_files:
            print(f"\n🔍 Analyzing {len(remaining_files)} files for demonstration...")
            analysis_results = []
            
            for file_path in remaining_files:
                print(f"\n📁 Analyzing: {os.path.basename(file_path)}")
                analysis_result = monitor.analyze_current_data(file_path)
                
                if analysis_result:
                    analysis_results.append(analysis_result)
                    
                    # Process alerts
                    alerts = monitor.process_alerts(analysis_result)
                    
                    if alerts:
                        monitor.print_alerts(alerts)
                        monitor.save_alert_log(alerts)
                    else:
                        print("✅ No leaks detected - Normal operation")
            
            # Create final dashboard
            if analysis_results:
                dashboard_file = monitor.create_monitoring_dashboard(analysis_results)
                print(f"\n📊 Analysis complete! Dashboard: {dashboard_file}")
        
        print(f"\n🔄 To start continuous monitoring, uncomment the line below:")
        print(f"# monitor.monitor_directory('{data_directory}', check_interval=60)")
        
    else:
        print("❌ Need at least 2 files to establish baseline and demonstrate monitoring")

if __name__ == "__main__":
    main()
