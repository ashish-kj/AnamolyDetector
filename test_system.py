#!/usr/bin/env python3
"""
Quick System Test for Pipeline Leak Detection
===========================================

This script tests all components of our pipeline leak detection system
to ensure everything is working properly.
"""

import os
import sys
import importlib.util

def test_imports():
    """Test all required imports"""
    print("🧪 Testing System Components...")
    print("-" * 40)
    
    tests = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical computations"),
        ("matplotlib", "Basic plotting"),
        ("scipy", "Scientific computing"),
        ("sklearn", "Machine learning"),
        ("flask", "Web framework"),
        ("flask_socketio", "Real-time communication"),
        ("plotly", "Interactive charts")
    ]
    
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {module:<15} - {description}")
        except ImportError:
            print(f"❌ {module:<15} - {description} (MISSING)")
    
    print()

def test_data_files():
    """Test data file availability"""
    print("📁 Testing Data Files...")
    print("-" * 40)
    
    data_dir = "TestData"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found")
        return
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"✅ Found {len(csv_files)} CSV files:")
    for i, file in enumerate(csv_files, 1):
        file_path = os.path.join(data_dir, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"   {i}. {file} ({size_mb:.1f} MB)")
    
    print()

def test_code_files():
    """Test code file availability"""
    print("🐍 Testing Code Files...")
    print("-" * 40)
    
    code_files = [
        ("code/analyze.py", "Basic data analysis"),
        ("code/pipeline_leak_detector.py", "Specialized leak detection"),
        ("code/live_detection.py", "Live web dashboard"),
        ("code/realtime_pipeline_monitor.py", "Real-time monitoring"),
        ("PIPELINE_LEAK_DETECTION_GUIDE.md", "User guide")
    ]
    
    for file_path, description in code_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path:<35} - {description}")
        else:
            print(f"❌ {file_path:<35} - {description} (MISSING)")
    
    print()

def test_pipeline_detector():
    """Test pipeline detector functionality"""
    print("🔍 Testing Pipeline Detector...")
    print("-" * 40)
    
    try:
        # Import pipeline detector
        sys.path.append("code")
        spec = importlib.util.spec_from_file_location("pipeline_leak_detector", "code/pipeline_leak_detector.py")
        pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_module)
        
        # Create detector instance
        detector = pipeline_module.PipelineLeakDetector()
        print("✅ Pipeline detector initialized successfully")
        
        # Test data loading
        data_dir = "TestData"
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if csv_files:
            test_file = os.path.join(data_dir, csv_files[0])
            data, distances = detector.load_pipeline_data(test_file)
            print(f"✅ Data loading test passed: {data.shape[0]} samples, {len(distances)} spatial points")
        else:
            print("⚠️  No CSV files found for testing")
            
    except Exception as e:
        print(f"❌ Pipeline detector test failed: {e}")
    
    print()

def run_quick_analysis():
    """Run a quick analysis on the first data file"""
    print("📊 Running Quick Analysis...")
    print("-" * 40)
    
    try:
        import pandas as pd
        import numpy as np
        
        data_dir = "TestData"
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            print("❌ No CSV files found")
            return
        
        # Load first file
        file_path = os.path.join(data_dir, csv_files[0])
        df = pd.read_csv(file_path)
        
        # Basic analysis
        print(f"✅ Loaded: {csv_files[0]}")
        print(f"   Shape: {df.shape[0]} time samples × {df.shape[1]-1} spatial points")
        print(f"   Pipeline length: ~{(df.shape[1]-1)*4/1000:.1f} km")
        print(f"   Data range: [{df.iloc[:,1:].min().min():.1f}, {df.iloc[:,1:].max().max():.1f}]")
        
        # Quick anomaly check
        data_values = df.iloc[:,1:].values
        mean_val = np.mean(data_values)
        std_val = np.std(data_values)
        z_scores = np.abs((data_values - mean_val) / std_val)
        anomalies = np.sum(z_scores > 3)
        
        print(f"   Mean signal: {mean_val:.2f}")
        print(f"   Std deviation: {std_val:.2f}")
        print(f"   Potential anomalies (Z>3): {anomalies} ({anomalies/data_values.size*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Quick analysis failed: {e}")
    
    print()

def main():
    """Run all tests"""
    print("🛢️ PIPELINE LEAK DETECTION SYSTEM TEST")
    print("=" * 50)
    print()
    
    test_imports()
    test_data_files()
    test_code_files()
    test_pipeline_detector()
    run_quick_analysis()
    
    print("🎯 NEXT STEPS:")
    print("-" * 40)
    print("1. Run basic analysis:     python code/analyze.py")
    print("2. Run leak detection:     python code/pipeline_leak_detector.py")
    print("3. Start live dashboard:   python code/live_detection.py")
    print("                          (then open http://localhost:5000)")
    print()
    print("📖 Read the guide:         PIPELINE_LEAK_DETECTION_GUIDE.md")
    print()
    print("✅ System test complete!")

if __name__ == "__main__":
    main()
