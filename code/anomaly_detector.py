# Advanced Anomaly Detection for DAS Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
import warnings
warnings.filterwarnings('ignore')

class DASAnomalyDetector:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_single_file(self, file_path):
        """Load a single DAS file for analysis"""
        df = pd.read_csv(file_path)
        # Get only the measurement columns (exclude time column)
        measurement_cols = [col for col in df.columns if col != 'Time(ms)/Distance(m)']
        return df[measurement_cols].values
    
    def preprocess_data(self, data, method='standard'):
        """Preprocess the data for anomaly detection"""
        print(f"Preprocessing data using {method} method...")
        
        if method == 'standard':
            # Standardize the data
            scaler = StandardScaler()
            processed_data = scaler.fit_transform(data)
            self.scalers['standard'] = scaler
            
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(data, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1  # Avoid division by zero
            processed_data = (data - median) / iqr
            
        elif method == 'minmax':
            # Min-max normalization
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            data_range = data_max - data_min
            data_range[data_range == 0] = 1  # Avoid division by zero
            processed_data = (data - data_min) / data_range
            
        elif method == 'smooth':
            # Apply smoothing filter
            processed_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                if data.shape[0] > 10:  # Need enough points for smoothing
                    processed_data[:, i] = savgol_filter(data[:, i], 
                                                       window_length=min(11, data.shape[0]//2*2-1), 
                                                       polyorder=3)
                else:
                    processed_data[:, i] = data[:, i]
        else:
            processed_data = data.copy()
            
        return processed_data
    
    def statistical_anomaly_detection(self, data, method='zscore', threshold=3):
        """Detect anomalies using statistical methods"""
        print(f"Running statistical anomaly detection ({method})...")
        
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))
            anomalies = np.any(z_scores > threshold, axis=1)
            scores = np.max(z_scores, axis=1)
            
        elif method == 'iqr':
            # IQR method
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = np.any((data < lower_bound) | (data > upper_bound), axis=1)
            # Calculate anomaly scores based on distance from bounds
            lower_dist = np.maximum(0, lower_bound - data)
            upper_dist = np.maximum(0, data - upper_bound)
            scores = np.max(lower_dist + upper_dist, axis=1)
            
        elif method == 'modified_zscore':
            # Modified Z-score using median
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            mad[mad == 0] = 1e-6  # Avoid division by zero
            modified_z_scores = 0.6745 * (data - median) / mad
            anomalies = np.any(np.abs(modified_z_scores) > threshold, axis=1)
            scores = np.max(np.abs(modified_z_scores), axis=1)
        
        return anomalies, scores
    
    def isolation_forest_detection(self, data, contamination=0.1):
        """Detect anomalies using Isolation Forest"""
        print("Running Isolation Forest anomaly detection...")
        
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(data)
        scores = model.decision_function(data)
        
        # Convert predictions (-1 for anomaly, 1 for normal) to boolean
        anomalies = predictions == -1
        
        self.models['isolation_forest'] = model
        return anomalies, -scores  # Negate scores so higher = more anomalous
    
    def one_class_svm_detection(self, data, nu=0.1):
        """Detect anomalies using One-Class SVM"""
        print("Running One-Class SVM anomaly detection...")
        
        model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        predictions = model.fit_predict(data)
        scores = model.decision_function(data)
        
        # Convert predictions (-1 for anomaly, 1 for normal) to boolean
        anomalies = predictions == -1
        
        self.models['one_class_svm'] = model
        return anomalies, -scores  # Negate scores so higher = more anomalous
    
    def pca_based_detection(self, data, n_components=0.95, threshold=3):
        """Detect anomalies using PCA reconstruction error"""
        print("Running PCA-based anomaly detection...")
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data)
        data_reconstructed = pca.inverse_transform(data_pca)
        
        # Calculate reconstruction error
        reconstruction_errors = np.sum((data - data_reconstructed) ** 2, axis=1)
        
        # Identify anomalies based on reconstruction error
        error_threshold = np.mean(reconstruction_errors) + threshold * np.std(reconstruction_errors)
        anomalies = reconstruction_errors > error_threshold
        
        self.models['pca'] = pca
        return anomalies, reconstruction_errors
    
    def dbscan_detection(self, data, eps=0.5, min_samples=5):
        """Detect anomalies using DBSCAN clustering"""
        print("Running DBSCAN-based anomaly detection...")
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data)
        
        # Points labeled as -1 are considered anomalies
        anomalies = cluster_labels == -1
        
        # Calculate anomaly scores based on distance to nearest cluster
        scores = np.zeros(len(data))
        for i, point in enumerate(data):
            if anomalies[i]:
                # For anomalies, calculate distance to nearest non-anomaly point
                normal_points = data[~anomalies]
                if len(normal_points) > 0:
                    distances = np.linalg.norm(normal_points - point, axis=1)
                    scores[i] = np.min(distances)
                else:
                    scores[i] = 1.0
            else:
                scores[i] = 0.0
        
        self.models['dbscan'] = dbscan
        return anomalies, scores
    
    def temporal_anomaly_detection(self, data, window_size=50, threshold=3):
        """Detect temporal anomalies using sliding window"""
        print("Running temporal anomaly detection...")
        
        anomalies = np.zeros(data.shape[0], dtype=bool)
        scores = np.zeros(data.shape[0])
        
        # For each spatial point, detect temporal anomalies
        for col in range(0, data.shape[1], 100):  # Sample every 100th column
            signal = data[:, col]
            
            # Calculate moving statistics
            for i in range(len(signal)):
                start_idx = max(0, i - window_size//2)
                end_idx = min(len(signal), i + window_size//2)
                window_data = signal[start_idx:end_idx]
                
                if len(window_data) > 1:
                    window_mean = np.mean(window_data)
                    window_std = np.std(window_data)
                    
                    if window_std > 0:
                        z_score = abs(signal[i] - window_mean) / window_std
                        if z_score > threshold:
                            anomalies[i] = True
                            scores[i] = max(scores[i], z_score)
        
        return anomalies, scores
    
    def ensemble_detection(self, data, methods=['isolation_forest', 'statistical', 'pca'], 
                         voting='majority'):
        """Combine multiple detection methods"""
        print("Running ensemble anomaly detection...")
        
        all_anomalies = {}
        all_scores = {}
        
        # Run each method
        if 'statistical' in methods:
            anom, scores = self.statistical_anomaly_detection(data, method='zscore')
            all_anomalies['statistical'] = anom
            all_scores['statistical'] = scores
        
        if 'isolation_forest' in methods:
            anom, scores = self.isolation_forest_detection(data)
            all_anomalies['isolation_forest'] = anom
            all_scores['isolation_forest'] = scores
        
        if 'one_class_svm' in methods:
            anom, scores = self.one_class_svm_detection(data)
            all_anomalies['one_class_svm'] = anom
            all_scores['one_class_svm'] = scores
        
        if 'pca' in methods:
            anom, scores = self.pca_based_detection(data)
            all_anomalies['pca'] = anom
            all_scores['pca'] = scores
        
        if 'dbscan' in methods:
            anom, scores = self.dbscan_detection(data)
            all_anomalies['dbscan'] = anom
            all_scores['dbscan'] = scores
        
        if 'temporal' in methods:
            anom, scores = self.temporal_anomaly_detection(data)
            all_anomalies['temporal'] = anom
            all_scores['temporal'] = scores
        
        # Combine results
        if voting == 'majority':
            # Majority voting
            votes = np.array(list(all_anomalies.values())).astype(int)
            ensemble_anomalies = np.sum(votes, axis=0) > len(methods) // 2
        elif voting == 'unanimous':
            # All methods must agree
            votes = np.array(list(all_anomalies.values())).astype(int)
            ensemble_anomalies = np.sum(votes, axis=0) == len(methods)
        elif voting == 'any':
            # Any method detecting anomaly
            votes = np.array(list(all_anomalies.values())).astype(int)
            ensemble_anomalies = np.sum(votes, axis=0) > 0
        
        # Combine scores (average)
        all_scores_normalized = {}
        for method, scores in all_scores.items():
            # Normalize scores to [0, 1]
            if np.max(scores) > np.min(scores):
                all_scores_normalized[method] = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            else:
                all_scores_normalized[method] = scores
        
        ensemble_scores = np.mean(list(all_scores_normalized.values()), axis=0)
        
        return ensemble_anomalies, ensemble_scores, all_anomalies, all_scores
    
    def visualize_results(self, data, anomalies, scores, method_name="Anomaly Detection"):
        """Visualize anomaly detection results"""
        print(f"Creating visualizations for {method_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{method_name} Results', fontsize=16)
        
        # 1. Anomaly distribution over time
        ax1 = axes[0, 0]
        time_points = range(len(anomalies))
        ax1.scatter(time_points, anomalies.astype(int), c=scores, cmap='Reds', alpha=0.6)
        ax1.set_title('Anomalies Over Time')
        ax1.set_xlabel('Time Sample')
        ax1.set_ylabel('Anomaly (1=Yes, 0=No)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Score distribution
        ax2 = axes[0, 1]
        ax2.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        ax2.set_title('Anomaly Score Distribution')
        ax2.set_xlabel('Anomaly Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample data with anomalies highlighted
        ax3 = axes[1, 0]
        # Plot first spatial dimension
        sample_data = data[:, 0]  # First spatial point
        normal_indices = ~anomalies
        anomaly_indices = anomalies
        
        ax3.plot(time_points, sample_data, 'b-', alpha=0.6, label='Normal')
        ax3.scatter(np.array(time_points)[anomaly_indices], 
                   sample_data[anomaly_indices], 
                   c='red', s=50, label='Anomalies', zorder=5)
        ax3.set_title('Sample Signal with Anomalies')
        ax3.set_xlabel('Time Sample')
        ax3.set_ylabel('Signal Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Detection Statistics:
        
        Total Samples: {len(anomalies)}
        Anomalies Detected: {np.sum(anomalies)}
        Anomaly Rate: {(np.sum(anomalies)/len(anomalies)*100):.2f}%
        
        Score Statistics:
        Mean Score: {np.mean(scores):.3f}
        Std Score: {np.std(scores):.3f}
        Max Score: {np.max(scores):.3f}
        Min Score: {np.min(scores):.3f}
        
        Threshold Used: Auto-determined
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{method_name.lower().replace(" ", "_")}_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_file(self, file_path, methods=['ensemble']):
        """Complete anomaly analysis for a single file"""
        print(f"\n{'='*60}")
        print(f"ANALYZING FILE: {file_path}")
        print(f"{'='*60}")
        
        # Load data
        data = self.load_single_file(file_path)
        print(f"Data shape: {data.shape}")
        
        # Preprocess data
        processed_data = self.preprocess_data(data, method='standard')
        
        results = {}
        
        for method in methods:
            if method == 'ensemble':
                anomalies, scores, individual_results, individual_scores = self.ensemble_detection(
                    processed_data, 
                    methods=['isolation_forest', 'statistical', 'pca'],
                    voting='majority'
                )
                results['ensemble'] = {
                    'anomalies': anomalies,
                    'scores': scores,
                    'individual_results': individual_results,
                    'individual_scores': individual_scores
                }
                self.visualize_results(processed_data, anomalies, scores, "Ensemble Detection")
                
            elif method == 'statistical':
                anomalies, scores = self.statistical_anomaly_detection(processed_data)
                results['statistical'] = {'anomalies': anomalies, 'scores': scores}
                self.visualize_results(processed_data, anomalies, scores, "Statistical Detection")
                
            elif method == 'isolation_forest':
                anomalies, scores = self.isolation_forest_detection(processed_data)
                results['isolation_forest'] = {'anomalies': anomalies, 'scores': scores}
                self.visualize_results(processed_data, anomalies, scores, "Isolation Forest Detection")
        
        self.results[file_path] = results
        return results

def main():
    """Main function to demonstrate anomaly detection"""
    import os
    
    detector = DASAnomalyDetector()
    
    # Analyze first file as example
    test_data_folder = "TestData"
    csv_files = [f for f in os.listdir(test_data_folder) if f.endswith('.csv')]
    
    if csv_files:
        first_file = os.path.join(test_data_folder, csv_files[0])
        print(f"Analyzing {first_file} as example...")
        
        # Run different detection methods
        results = detector.analyze_file(first_file, methods=['ensemble', 'statistical', 'isolation_forest'])
        
        print("\n✅ Anomaly detection complete!")
        print("📁 Check the generated visualization files.")
        
    else:
        print("❌ No CSV files found in TestData folder!")

if __name__ == "__main__":
    main()
