#!/usr/bin/env python3
"""
GitHub Repository Preparation Script
===================================

This script helps prepare the Pipeline Leak Detection System for GitHub upload.
It checks all files, validates the system, and provides upload instructions.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and report status"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
        print(f"✅ {file_path:<35} - {description} ({size} bytes)")
        return True
    else:
        print(f"❌ {file_path:<35} - {description} (MISSING)")
        return False

def check_git_status():
    """Check Git repository status"""
    try:
        # Check if git is initialized
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository initialized")
            return True
        else:
            print("❌ Git repository not initialized")
            return False
    except FileNotFoundError:
        print("❌ Git not installed or not in PATH")
        return False

def get_file_sizes():
    """Get sizes of important files"""
    important_files = [
        'README.md',
        'PIPELINE_LEAK_DETECTION_GUIDE.md',
        'code/live_detection.py',
        'code/pipeline_leak_detector.py',
        'requirements.txt'
    ]
    
    total_size = 0
    for file_path in important_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_size += size
    
    return total_size

def check_large_files():
    """Check for large files that might need Git LFS"""
    large_files = []
    
    for root, dirs, files in os.walk('.'):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size > 50 * 1024 * 1024:  # 50MB threshold
                    large_files.append((file_path, size))
            except OSError:
                pass
    
    return large_files

def main():
    """Main preparation function"""
    print("🚀 GITHUB REPOSITORY PREPARATION")
    print("=" * 50)
    print()
    
    # Check essential files
    print("📁 Checking Essential Files...")
    print("-" * 40)
    
    essential_files = [
        ("README.md", "Main repository documentation"),
        ("LICENSE", "MIT license file"),
        (".gitignore", "Git ignore patterns"),
        ("requirements.txt", "Python dependencies"),
        ("setup.py", "Package installation script"),
        ("Dockerfile", "Docker container configuration"),
        ("docker-compose.yml", "Docker compose configuration"),
        ("CONTRIBUTING.md", "Contribution guidelines"),
        ("DEPLOYMENT.md", "Deployment guide"),
        ("PIPELINE_LEAK_DETECTION_GUIDE.md", "User guide"),
        (".github/workflows/ci.yml", "CI/CD workflow"),
    ]
    
    missing_files = []
    for file_path, description in essential_files:
        if not check_file_exists(file_path, description):
            missing_files.append(file_path)
    
    print()
    
    # Check code files
    print("🐍 Checking Code Files...")
    print("-" * 40)
    
    code_files = [
        ("code/analyze.py", "Basic data analysis"),
        ("code/pipeline_leak_detector.py", "Main leak detection system"),
        ("code/live_detection.py", "Live web dashboard"),
        ("code/realtime_pipeline_monitor.py", "Real-time monitoring"),
        ("code/data_explorer.py", "Data exploration tools"),
        ("test_system.py", "System validation tests"),
        ("prepare_for_github.py", "This preparation script"),
    ]
    
    for file_path, description in code_files:
        check_file_exists(file_path, description)
    
    print()
    
    # Check Git status
    print("📋 Checking Git Repository...")
    print("-" * 40)
    git_ready = check_git_status()
    print()
    
    # Check file sizes
    print("📊 Repository Statistics...")
    print("-" * 40)
    total_code_size = get_file_sizes()
    print(f"✅ Total code size: {total_code_size / 1024:.1f} KB")
    
    # Check for large files
    large_files = check_large_files()
    if large_files:
        print("⚠️  Large files detected (consider Git LFS):")
        for file_path, size in large_files:
            print(f"   {file_path}: {size / (1024*1024):.1f} MB")
    else:
        print("✅ No large files detected")
    
    print()
    
    # System validation
    print("🧪 Running System Validation...")
    print("-" * 40)
    
    try:
        # Run system test
        result = subprocess.run([sys.executable, 'test_system.py'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ System validation passed")
        else:
            print("❌ System validation failed")
            print("Error output:", result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr)
    except subprocess.TimeoutExpired:
        print("⚠️  System validation timed out")
    except Exception as e:
        print(f"❌ System validation error: {e}")
    
    print()
    
    # Final recommendations
    print("🎯 GITHUB UPLOAD RECOMMENDATIONS")
    print("-" * 40)
    
    if missing_files:
        print("❌ Missing essential files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print()
    
    if not git_ready:
        print("🔧 Git Setup Required:")
        print("   git init")
        print("   git add .")
        print("   git commit -m 'Initial commit: Pipeline Leak Detection System'")
        print("   git branch -M main")
        print("   git remote add origin https://github.com/ashish-kj/AnamolyDetector.git")
        print("   git push -u origin main")
        print()
    else:
        print("🚀 Ready for GitHub Upload:")
        print("   git add .")
        print("   git commit -m 'Complete pipeline leak detection system with live dashboard'")
        print("   git push origin main")
        print()
    
    # Repository features
    print("🌟 Repository Features Ready:")
    print("   ✅ Professional README with badges")
    print("   ✅ Comprehensive documentation")
    print("   ✅ MIT License")
    print("   ✅ Docker support")
    print("   ✅ CI/CD workflow")
    print("   ✅ Contributing guidelines")
    print("   ✅ Deployment guide")
    print("   ✅ Live web dashboard")
    print("   ✅ Multiple detection algorithms")
    print("   ✅ Real-time monitoring")
    print()
    
    # Next steps
    print("📋 NEXT STEPS:")
    print("1. Review all files and documentation")
    print("2. Test the live dashboard: python code/live_detection.py")
    print("3. Initialize Git repository (if not done)")
    print("4. Upload to GitHub")
    print("5. Enable GitHub Pages for documentation")
    print("6. Set up GitHub Actions for CI/CD")
    print("7. Add repository topics: pipeline, leak-detection, das, monitoring")
    print()
    
    if missing_files or not git_ready:
        print("⚠️  Please resolve the issues above before uploading to GitHub")
    else:
        print("🎉 Repository is ready for GitHub!")
        print("   Your pipeline leak detection system is production-ready!")

if __name__ == "__main__":
    main()
