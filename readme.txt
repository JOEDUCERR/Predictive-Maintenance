AI4I 2020 Predictive Maintenance Dataset

6 features

"Walk me through your entire system"
Answer: "My system has 5 stages:
Stage 1 - Data Acquisition: Three sensors (vibration, temperature, acoustic) collect 256 samples each at 200 Hz, giving us 1.28 seconds of data per reading.
Stage 2 - Feature Extraction: We extract 20 features: 12 statistical features like mean and standard deviation, 6 frequency features using FFT like dominant frequency and spectral entropy, and 2 cross-sensor correlation features.
Stage 3 - Model Inference: A lightweight DNN with 4 layers (128-64-32-1 neurons) processes the features. It uses ReLU activation and dropout for regularization, outputting a fault probability.
Stage 4 - Decision Making: If probability exceeds 0.6 threshold, we trigger a fault alert. The entire pipeline takes less than 10ms.
Stage 5 - Monitoring: The Streamlit interface displays real-time results, historical trends, and allows parameter adjustment."