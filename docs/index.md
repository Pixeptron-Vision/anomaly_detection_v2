# Anomaly Detection

An anomaly-based defect detection system for industrial quality control, built as part of the **Building AI-Powered Defect Detection Systems for Industrial Quality Control** course.

## What This Project Does

This application inspects metal nut images from the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and classifies them as **Good** or **Defective** using the PatchCore anomaly detection algorithm. It produces heatmaps showing exactly where the model identifies anomalies.

## Project Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Interactive inspection UI with image display and metrics |
| Backend | PatchCore (Anomalib) | Anomaly detection model with heatmap generation |
| Camera Simulation | Custom Python module | Mimics industrial camera acquisition from dataset images |
| Documentation | MkDocs + Material | This site — project docs and guides |
| CI/CD | GitHub Actions | Automated testing and linting on every push |
| Logging | Python logging | Structured logs for debugging and monitoring |

## Quick Start

```bash
# Clone and set up
git clone <your-repo-url>
cd anomaly_detection

# Install dependencies (see Installation page for full details)
pip install -r requirements.txt

# Train the model (one-time, ~2 minutes)
python -m anomaly_detection.train

# Launch the app
streamlit run anomaly_detection/app.py
```
