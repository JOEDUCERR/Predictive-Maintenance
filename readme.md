# 🔧 IoT Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Predictive Maintenance System for IoT Devices Using Edge Sensor Fusion and Lightweight Deep Learning**

A complete implementation of edge-enabled predictive maintenance framework that combines multi-sensor fusion with lightweight deep learning models for real-time fault prediction without cloud infrastructure dependency.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Research Paper](#research-paper)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This system demonstrates a lightweight predictive maintenance framework that:

- Integrates **temperature, vibration, and acoustic** sensor data
- Extracts 20 statistical and frequency-domain features
- Achieves **97.1% prediction accuracy** on IoT fault detection
- Operates with **<10ms inference latency**
- Optimized for edge devices (ESP32, Raspberry Pi)
- Eliminates hardware dependencies during development

## ✨ Features

### 🔬 Core Capabilities

- **Multi-Sensor Fusion**: Combines 3 sensor modalities for robust fault detection
- **Lightweight DNN**: 4-layer neural network optimized for edge deployment
- **Real-Time Monitoring**: Live fault prediction with probability-based alerts
- **Edge Computing**: No cloud dependency, fully autonomous operation
- **Software Simulation**: Complete development without physical hardware

### 📊 Technical Specifications

- **Input Features**: 20 (statistical + frequency-domain + fusion)
- **Model Size**: ~4 MB (after quantization)
- **Inference Time**: 8-9 ms per prediction
- **Accuracy**: 97.1%
- **Memory Footprint**: ~42 MB
- **Power Consumption**: <2W on edge devices

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│           Multi-Sensor Data Acquisition             │
│  (Vibration | Temperature | Acoustic)               │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│         Feature Extraction & Fusion                 │
│  • Statistical Features (mean, std, etc.)           │
│  • Frequency Features (FFT-based)                   │
│  • Cross-Correlation Features                       │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│       Lightweight Deep Neural Network               │
│  Input(20) → Dense(128) → Dense(64) → Dense(32) → 1│
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│         Real-Time Alert Generation                  │
│  Fault Probability > Threshold → Maintenance Alert  │
└─────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/iot-predictive-maintenance.git
cd iot-predictive-maintenance
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Streamlit App

```bash
streamlit run predictive_maintenance.py
```

The app will open in your browser at `http://localhost:8501`

### App Navigation

#### 1️⃣ **Train Model Tab**
- Configure dataset size (1,000 - 10,000 samples)
- Set training epochs and batch size
- Start training and view real-time progress
- Visualize training history

#### 2️⃣ **Evaluate Tab**
- View comprehensive performance metrics
- Analyze dataset distribution
- Examine model architecture

#### 3️⃣ **Live Monitor Tab**
- Start real-time fault monitoring
- Configure alert threshold (0.0 - 1.0)
- Set reading interval (1-5 seconds)
- View live probability charts and alerts

#### 4️⃣ **About Tab**
- System overview and specifications
- Research paper information
- Technology stack details

## 📈 Model Performance

### Metrics on Test Set

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.10% |
| **Precision** | 96.85% |
| **Recall** | 97.42% |
| **F1-Score** | 97.13% |
| **Specificity** | 96.77% |

### Computational Efficiency

| Parameter | Value |
|-----------|-------|
| **Inference Latency** | 8.7 ms |
| **Memory Footprint** | 42 MB |
| **CPU Utilization** | <6% |
| **Power Consumption** | 1.8W |
| **Model Size** | 4 MB (quantized) |

## 📄 Research Paper

**Title:** Predictive Maintenance System and Method for IoT Devices Using Edge Sensor Fusion and Lightweight Deep Learning

**Author:** Jonah Sudhir  
**Institution:** Lovely Professional University, Punjab, India  
**Conference:** International Conference on Sustainable Computing and Artificial Intelligence (ICSCAIconf2025)  
**Paper ID:** 370  
**Submission Date:** November 13, 2025

### Abstract

This paper presents a lightweight predictive maintenance framework combining multi-sensor fusion with edge-deployed deep learning models to enable real-time fault prediction without reliance on cloud infrastructure. The proposed system integrates temperature, vibration, and acoustic sensor data into a unified feature set, achieving 97.1% prediction accuracy with 9ms inference latency on constrained devices.

## 🖼️ Screenshots

### Training Dashboard
![Training](screenshots/training.png)

### Real-Time Monitoring
![Monitoring](screenshots/monitoring.png)

### Performance Metrics
![Metrics](screenshots/metrics.png)

## 🔧 Hardware Deployment

The system is designed for seamless migration to physical IoT hardware:

### Supported Platforms
- **ESP32** - Low-cost WiFi/Bluetooth microcontroller
- **Raspberry Pi** - Single-board computer
- **Arduino Nano 33 BLE Sense** - Embedded ML platform
- **NVIDIA Jetson Nano** - AI edge computing

### Deployment Steps
1. Convert model to TensorFlow Lite format
2. Quantize to 8-bit for memory efficiency
3. Flash firmware to target device
4. Connect physical sensors (MPU6050, DHT22, etc.)
5. Configure MQTT/HTTP communication

## 🛠️ Technology Stack

- **Framework**: TensorFlow 2.15 / Keras
- **UI**: Streamlit 1.29
- **Language**: Python 3.9+
- **Data Processing**: NumPy, Pandas, SciPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Jonah Sudhir**  
Email: joeg123456789014@gmail.com  
Institution: Lovely Professional University, Punjab, India

**Project Link**: [https://github.com/yourusername/iot-predictive-maintenance](https://github.com/yourusername/iot-predictive-maintenance)

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- Streamlit for the interactive web framework
- Open-source IoT and TinyML communities
- Lovely Professional University for research support

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{sudhir2025predictive,
  title={Predictive Maintenance System and Method for IoT Devices Using Edge Sensor Fusion and Lightweight Deep Learning},
  author={Sudhir, Jonah},
  booktitle={International Conference on Sustainable Computing and Artificial Intelligence},
  year={2025},
  organization={ICSCAIconf2025}
}
```

---

⭐ **Star this repository if you find it helpful!** ⭐
