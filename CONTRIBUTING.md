# Contributing to Pipeline Leak Detection System

Thank you for your interest in contributing to this pipeline leak detection system! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Basic understanding of pipeline monitoring and DAS technology
- Familiarity with anomaly detection concepts

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/AnamolyDetector.git
   cd AnamolyDetector
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test the setup**
   ```bash
   python test_system.py
   ```

## 🎯 How to Contribute

### Types of Contributions Welcome

1. **Bug Fixes**
   - Fix detection algorithm issues
   - Resolve web dashboard problems
   - Improve error handling

2. **Feature Enhancements**
   - New detection algorithms
   - Additional visualization options
   - Performance improvements
   - Mobile-responsive dashboard features

3. **Documentation**
   - Improve user guides
   - Add code comments
   - Create tutorials or examples

4. **Testing**
   - Add unit tests
   - Improve system validation
   - Test with different data formats

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards below
   - Add appropriate tests
   - Update documentation if needed

3. **Test your changes**
   ```bash
   python test_system.py
   python code/live_detection.py  # Test web dashboard
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide clear description of changes
   - Include any relevant issue numbers
   - Add screenshots for UI changes

## 📝 Coding Standards

### Python Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions and classes
- Keep functions focused and modular

### Example Code Structure
```python
def detect_anomalies(data: np.ndarray, threshold: float = 3.0) -> List[Dict]:
    """
    Detect anomalies in pipeline data using statistical methods.
    
    Args:
        data: Pipeline sensor data array
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        List of detected anomalies with metadata
    """
    # Implementation here
    pass
```

### Web Development
- Use semantic HTML
- Follow responsive design principles
- Ensure accessibility standards
- Test across different browsers

## 🧪 Testing Guidelines

### Running Tests
```bash
# System validation
python test_system.py

# Manual testing
python code/analyze.py
python code/pipeline_leak_detector.py
python code/live_detection.py
```

### Adding New Tests
- Add tests for new detection algorithms
- Test edge cases and error conditions
- Validate web dashboard functionality
- Test with different data formats

## 📋 Issue Guidelines

### Reporting Bugs
Please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)
- Sample data if relevant (anonymized)

### Feature Requests
Please include:
- Clear description of the feature
- Use case and benefits
- Proposed implementation approach
- Any relevant examples or references

## 🔍 Code Review Process

### What We Look For
- **Functionality**: Does the code work as intended?
- **Code Quality**: Is it readable, maintainable, and efficient?
- **Testing**: Are there appropriate tests?
- **Documentation**: Is it well-documented?
- **Compatibility**: Does it work with existing system?

### Review Timeline
- Initial review: Within 1-2 weeks
- Follow-up reviews: Within 3-5 days
- Merge: After approval and all checks pass

## 🏗️ Architecture Guidelines

### Core Components
- **Detection Algorithms**: Located in `code/` directory
- **Web Dashboard**: Flask-based with Socket.IO for real-time updates
- **Data Processing**: NumPy/Pandas for efficient computation
- **Visualization**: Matplotlib/Plotly for charts

### Adding New Detection Methods
1. Create method in `PipelineLeakDetector` class
2. Add to ensemble detection workflow
3. Update web dashboard to display results
4. Add configuration parameters
5. Update documentation

### Database/Storage
- Currently file-based (CSV input, JSON alerts)
- Consider database integration for production use
- Maintain backward compatibility

## 🔒 Security Considerations

- Validate all user inputs
- Sanitize file paths and names
- Use secure communication for web dashboard
- Follow security best practices for web applications
- Be mindful of sensitive pipeline data

## 📚 Documentation Standards

### Code Documentation
- Docstrings for all public functions/classes
- Inline comments for complex logic
- Type hints for function parameters and returns

### User Documentation
- Update README.md for new features
- Update PIPELINE_LEAK_DETECTION_GUIDE.md for usage changes
- Include examples and use cases

## 🌟 Recognition

Contributors will be:
- Listed in the repository contributors
- Mentioned in release notes for significant contributions
- Credited in documentation where appropriate

## 📞 Getting Help

- Create an issue for questions
- Check existing documentation
- Review similar implementations in the codebase

## 📋 Checklist for Pull Requests

- [ ] Code follows project style guidelines
- [ ] Tests pass (`python test_system.py`)
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] No sensitive data included
- [ ] Web dashboard tested if UI changes made
- [ ] Backward compatibility maintained

Thank you for contributing to making pipeline monitoring safer and more effective! 🛢️🔍
