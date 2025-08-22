# Advanced Data Explorer for DAS Anomaly Detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import os
import warnings
warnings.filterwarnings('ignore')

class DASDataExplorer:
    def __init__(self, folder_path="TestData"):
        self.folder_path = folder_path
        self.data_files = []
        self.combined_data = None
        self.metadata = {}
        
    def load_all_data(self):
        """Load all CSV files and combine them for analysis"""
        print("Loading all DAS data files...")
        
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        all_data = []
        
        for i, csv_file in enumerate(sorted(csv_files)):
            file_path = os.path.join(self.folder_path, csv_file)
            print(f"Loading {csv_file}...")
            
            df = pd.read_csv(file_path)
            # Add file identifier and timestamp info
            df['file_id'] = i
            df['filename'] = csv_file
            
            # Extract timestamp from filename if available
            timestamp_part = csv_file.split('_')[2:5]  # Extract time components
            df['timestamp'] = '_'.join(timestamp_part)
            
            all_data.append(df)
            self.data_files.append(csv_file)
        
        # Combine all data
        self.combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {self.combined_data.shape}")
        
        # Store metadata
        self.metadata = {
            'num_files': len(csv_files),
            'total_samples': self.combined_data.shape[0],
            'spatial_points': self.combined_data.shape[1] - 4,  # Excluding metadata columns
            'distance_columns': [col for col in self.combined_data.columns if col not in ['Time(ms)/Distance(m)', 'file_id', 'filename', 'timestamp']]
        }
        
        return self.combined_data
    
    def analyze_temporal_patterns(self):
        """Analyze patterns across time samples"""
        print("\n" + "="*60)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*60)
        
        # Get distance columns only
        distance_cols = self.metadata['distance_columns']
        
        # Calculate statistics across time for each spatial point
        temporal_stats = {}
        
        for file_id in self.combined_data['file_id'].unique():
            file_data = self.combined_data[self.combined_data['file_id'] == file_id]
            file_values = file_data[distance_cols].values
            
            temporal_stats[f'file_{file_id}'] = {
                'mean_across_space': np.mean(file_values, axis=1),
                'std_across_space': np.std(file_values, axis=1),
                'max_across_space': np.max(file_values, axis=1),
                'min_across_space': np.min(file_values, axis=1)
            }
        
        # Plot temporal patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Temporal Patterns Analysis', fontsize=16)
        
        for i, (file_key, stats) in enumerate(temporal_stats.items()):
            if i >= 4:  # Only plot first 4 files
                break
            
            ax = axes[i//2, i%2]
            time_points = range(len(stats['mean_across_space']))
            
            ax.plot(time_points, stats['mean_across_space'], label='Mean', alpha=0.8)
            ax.fill_between(time_points, 
                          stats['mean_across_space'] - stats['std_across_space'],
                          stats['mean_across_space'] + stats['std_across_space'],
                          alpha=0.3, label='±1 Std')
            ax.plot(time_points, stats['max_across_space'], '--', alpha=0.6, label='Max')
            ax.plot(time_points, stats['min_across_space'], '--', alpha=0.6, label='Min')
            
            ax.set_title(f'File {i}: Temporal Evolution')
            ax.set_xlabel('Time Sample')
            ax.set_ylabel('Signal Amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return temporal_stats
    
    def analyze_spatial_patterns(self):
        """Analyze patterns across spatial dimensions"""
        print("\n" + "="*60)
        print("SPATIAL PATTERN ANALYSIS")
        print("="*60)
        
        distance_cols = self.metadata['distance_columns']
        
        # Calculate spatial statistics
        spatial_stats = {}
        
        for file_id in self.combined_data['file_id'].unique():
            file_data = self.combined_data[self.combined_data['file_id'] == file_id]
            file_values = file_data[distance_cols].values
            
            spatial_stats[f'file_{file_id}'] = {
                'mean_across_time': np.mean(file_values, axis=0),
                'std_across_time': np.std(file_values, axis=0),
                'max_across_time': np.max(file_values, axis=0),
                'min_across_time': np.min(file_values, axis=0)
            }
        
        # Convert distance column names to actual distances
        distances = [float(col) for col in distance_cols]
        
        # Plot spatial patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Spatial Patterns Analysis', fontsize=16)
        
        for i, (file_key, stats) in enumerate(spatial_stats.items()):
            if i >= 4:  # Only plot first 4 files
                break
            
            ax = axes[i//2, i%2]
            
            # Sample every 100th point for visualization (too many points otherwise)
            sample_indices = range(0, len(distances), 100)
            sampled_distances = [distances[i] for i in sample_indices]
            sampled_means = [stats['mean_across_time'][i] for i in sample_indices]
            sampled_stds = [stats['std_across_time'][i] for i in sample_indices]
            
            ax.plot(sampled_distances, sampled_means, label='Mean', alpha=0.8)
            ax.fill_between(sampled_distances, 
                          np.array(sampled_means) - np.array(sampled_stds),
                          np.array(sampled_means) + np.array(sampled_stds),
                          alpha=0.3, label='±1 Std')
            
            ax.set_title(f'File {i}: Spatial Distribution')
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Signal Amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('spatial_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return spatial_stats
    
    def detect_statistical_anomalies(self, threshold_std=3):
        """Detect anomalies using statistical methods"""
        print("\n" + "="*60)
        print("STATISTICAL ANOMALY DETECTION")
        print("="*60)
        
        distance_cols = self.metadata['distance_columns']
        anomalies = {}
        
        for file_id in self.combined_data['file_id'].unique():
            file_data = self.combined_data[self.combined_data['file_id'] == file_id]
            filename = file_data['filename'].iloc[0]
            
            print(f"\nAnalyzing {filename}...")
            
            # Get the measurement data
            measurements = file_data[distance_cols].values
            
            # Method 1: Z-score based anomalies
            z_scores = np.abs(stats.zscore(measurements, axis=None, nan_policy='omit'))
            z_anomalies = np.where(z_scores > threshold_std)
            
            # Method 2: IQR based anomalies
            Q1 = np.percentile(measurements, 25)
            Q3 = np.percentile(measurements, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_anomalies = np.where((measurements < lower_bound) | (measurements > upper_bound))
            
            # Method 3: Moving average based anomalies
            # Calculate moving average for each spatial point
            window_size = 50
            ma_anomalies = []
            
            for col_idx in range(measurements.shape[1]):
                if col_idx % 1000 == 0:  # Sample every 1000th column
                    signal = measurements[:, col_idx]
                    if len(signal) >= window_size:
                        moving_avg = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
                        # Pad to match original length
                        moving_avg = np.concatenate([signal[:window_size//2], moving_avg, signal[-(window_size//2):]])
                        
                        # Find deviations
                        deviations = np.abs(signal - moving_avg)
                        threshold = np.std(deviations) * 2
                        anomaly_indices = np.where(deviations > threshold)[0]
                        
                        for idx in anomaly_indices:
                            ma_anomalies.append((idx, col_idx))
            
            anomalies[filename] = {
                'z_score_anomalies': len(z_anomalies[0]),
                'iqr_anomalies': len(iqr_anomalies[0]),
                'moving_avg_anomalies': len(ma_anomalies),
                'total_points': measurements.size,
                'anomaly_percentage_z': (len(z_anomalies[0]) / measurements.size) * 100,
                'anomaly_percentage_iqr': (len(iqr_anomalies[0]) / measurements.size) * 100,
                'anomaly_percentage_ma': (len(ma_anomalies) / (measurements.shape[0] * (measurements.shape[1]//1000))) * 100
            }
            
            print(f"  Z-score anomalies: {anomalies[filename]['z_score_anomalies']} ({anomalies[filename]['anomaly_percentage_z']:.2f}%)")
            print(f"  IQR anomalies: {anomalies[filename]['iqr_anomalies']} ({anomalies[filename]['anomaly_percentage_iqr']:.2f}%)")
            print(f"  Moving avg anomalies: {anomalies[filename]['moving_avg_anomalies']} ({anomalies[filename]['anomaly_percentage_ma']:.2f}%)")
        
        return anomalies
    
    def create_heatmap_visualization(self):
        """Create heatmap visualization of the data"""
        print("\n" + "="*60)
        print("CREATING HEATMAP VISUALIZATIONS")
        print("="*60)
        
        distance_cols = self.metadata['distance_columns']
        
        # Create heatmaps for first 2 files
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('DAS Data Heatmaps (Sample)', fontsize=16)
        
        for i in range(min(2, self.metadata['num_files'])):
            file_data = self.combined_data[self.combined_data['file_id'] == i]
            filename = file_data['filename'].iloc[0]
            
            # Sample data for visualization (every 50th column, all rows)
            sample_cols = distance_cols[::50]  # Every 50th spatial point
            sample_data = file_data[sample_cols].values
            
            # Create heatmap
            im = axes[i].imshow(sample_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i].set_title(f'{filename}\n(Sampled: every 50th spatial point)')
            axes[i].set_xlabel('Time Sample')
            axes[i].set_ylabel('Spatial Point (sampled)')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], label='Signal Amplitude')
        
        plt.tight_layout()
        plt.savefig('data_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive data analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE DATA ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"  • Number of files: {self.metadata['num_files']}")
        print(f"  • Total samples: {self.metadata['total_samples']}")
        print(f"  • Spatial measurement points: {self.metadata['spatial_points']}")
        print(f"  • Distance range: {self.metadata['distance_columns'][0]} to {self.metadata['distance_columns'][-1]} meters")
        
        # Calculate overall statistics
        distance_cols = self.metadata['distance_columns']
        all_measurements = self.combined_data[distance_cols].values
        
        print(f"\n📈 STATISTICAL SUMMARY:")
        print(f"  • Overall mean: {np.mean(all_measurements):.3f}")
        print(f"  • Overall std: {np.std(all_measurements):.3f}")
        print(f"  • Overall min: {np.min(all_measurements):.3f}")
        print(f"  • Overall max: {np.max(all_measurements):.3f}")
        print(f"  • Data range: {np.ptp(all_measurements):.3f}")
        
        # File-by-file comparison
        print(f"\n📋 FILE-BY-FILE COMPARISON:")
        for file_id in self.combined_data['file_id'].unique():
            file_data = self.combined_data[self.combined_data['file_id'] == file_id]
            filename = file_data['filename'].iloc[0]
            file_measurements = file_data[distance_cols].values
            
            print(f"  {filename}:")
            print(f"    - Mean: {np.mean(file_measurements):.3f}")
            print(f"    - Std:  {np.std(file_measurements):.3f}")
            print(f"    - Range: [{np.min(file_measurements):.1f}, {np.max(file_measurements):.1f}]")
        
        print(f"\n🎯 RECOMMENDATIONS FOR ANOMALY DETECTION:")
        print(f"  1. Use multiple detection methods (statistical + ML-based)")
        print(f"  2. Consider both temporal and spatial anomalies")
        print(f"  3. Apply preprocessing (normalization, filtering)")
        print(f"  4. Use sliding window approach for real-time detection")
        print(f"  5. Validate results with domain expertise")

def main():
    # Initialize the explorer
    explorer = DASDataExplorer()
    
    # Load and analyze data
    print("Starting comprehensive DAS data analysis...")
    explorer.load_all_data()
    
    # Perform various analyses
    temporal_stats = explorer.analyze_temporal_patterns()
    spatial_stats = explorer.analyze_spatial_patterns()
    anomalies = explorer.detect_statistical_anomalies()
    explorer.create_heatmap_visualization()
    explorer.generate_comprehensive_report()
    
    print("\n✅ Analysis complete! Check the generated plots and results above.")
    print("📁 Generated files: temporal_patterns.png, spatial_patterns.png, data_heatmaps.png")

if __name__ == "__main__":
    main()
