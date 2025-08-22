"""
Setup configuration for Pipeline Leak Detection System
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="pipeline-leak-detector",
    version="1.0.0",
    author="Ashish Kumar Jha",
    author_email="your.email@example.com",
    description="Advanced pipeline leak detection system using DAS technology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashish-kj/AnamolyDetector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.8.0",
            "black>=21.0.0",
            "bandit>=1.7.0",
            "safety>=1.10.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pipeline-detector=code.pipeline_leak_detector:main",
            "live-detection=code.live_detection:main",
            "pipeline-monitor=code.realtime_pipeline_monitor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="pipeline leak detection DAS fiber optic monitoring anomaly detection",
    project_urls={
        "Bug Reports": "https://github.com/ashish-kj/AnamolyDetector/issues",
        "Source": "https://github.com/ashish-kj/AnamolyDetector",
        "Documentation": "https://github.com/ashish-kj/AnamolyDetector/blob/main/PIPELINE_LEAK_DETECTION_GUIDE.md",
    },
)
