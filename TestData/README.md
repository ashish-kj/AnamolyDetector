# TestData Directory

## 📊 **Test Data Location**

This directory is where you should place your **DAS (Distributed Acoustic Sensing) CSV data files** for pipeline leak detection analysis.

## 📋 **Required Data Format**

Your CSV files should contain:
- **Time column**: `Time(ms)/Distance(m)` (first column)
- **Distance columns**: Each column represents measurements at different distances along the pipeline
- **File naming**: Preferably with timestamps (e.g., `DAS_Test_YYYYMMDD_HHMMSS_*.csv`)

### Example File Structure:
```
TestData/
├── DAS_Test_20250805_100000_325(36991)_Raw.csv
├── DAS_Test_20250805_100001_326(37991)_Raw.csv
├── DAS_Test_20250805_100002_325(38991)_Raw.csv
└── ... (more CSV files)
```

## 🚀 **Getting Started**

1. **Add your CSV files** to this directory
2. **Run the analysis**: `python code/analyze.py`
3. **Start live monitoring**: `python code/live_detection.py`
4. **Access dashboard**: Open http://localhost:5000

## 📁 **Supported File Types**

- ✅ **CSV files** (.csv) - Primary data format
- ✅ **Multiple files** - System processes all CSV files in sequence
- ✅ **Large files** - Optimized for files up to 100MB+

## ⚠️ **Important Notes**

- **CSV files are not committed to Git** (excluded in .gitignore)
- **File size**: Each file can be 30MB+ (typical for DAS data)
- **Processing order**: Files are processed in chronological order based on filename timestamps
- **Real-time simulation**: The system uses time intervals between files to simulate live streaming

## 🔧 **Data Requirements**

Your DAS data should represent:
- **Optical fiber measurements** from oil pipeline monitoring
- **Temporal samples** (rows) at regular intervals
- **Spatial points** (columns) along the pipeline length
- **Acoustic intensity values** for leak detection

## 📖 **Need Help?**

- Check the main [README.md](../README.md) for full system documentation
- Review [PIPELINE_LEAK_DETECTION_GUIDE.md](../PIPELINE_LEAK_DETECTION_GUIDE.md) for detailed usage
- Run `python code/analyze.py` to validate your data format

---

**🛢️ Ready to detect pipeline leaks with your DAS data!**
