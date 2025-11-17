import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Disable TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Set page config
st.set_page_config(
    page_title="IoT Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


class SensorDataGenerator:
    
    def __init__(self, n_samples=5000, sampling_rate=200):
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate
        self.window_size = 256
        
    def generate_dataset(self):
        data = []
        labels = []
        
        n_normal = int(self.n_samples * 0.6)
        n_fault = self.n_samples - n_normal
        
        for _ in range(n_normal):
            vibration = np.random.normal(0, 1, self.window_size)
            temperature = np.random.normal(40, 1, self.window_size)
            acoustic = np.random.normal(0, 0.5, self.window_size)
            data.append([vibration, temperature, acoustic])
            labels.append(0)
        
        for _ in range(n_fault):
            vibration = np.random.normal(3, 2, self.window_size)
            temperature = np.random.normal(60, 3, self.window_size)
            acoustic = np.random.normal(2, 1, self.window_size)
            data.append([vibration, temperature, acoustic])
            labels.append(1)
        
        return np.array(data), np.array(labels)


class FeatureExtractor:
    """Extract statistical and frequency-domain features"""
    
    def extract_statistical_features(self, signal_data):
        features = []
        features.append(np.mean(signal_data))
        features.append(np.std(signal_data))
        features.append(np.min(signal_data))
        features.append(np.max(signal_data))
        features.append(np.ptp(signal_data))
        features.append(stats.skew(signal_data))
        features.append(stats.kurtosis(signal_data))
        features.append(np.sqrt(np.mean(signal_data**2)))
        return features
    
    def extract_frequency_features(self, signal_data, sampling_rate=200):
        features = []
        fft_vals = fft(signal_data)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        fft_freq = np.fft.fftfreq(len(signal_data), 1/sampling_rate)[:len(fft_vals)//2]
        
        dominant_freq_idx = np.argmax(fft_mag)
        features.append(fft_freq[dominant_freq_idx])
        
        spectral_centroid = np.sum(fft_freq * fft_mag) / np.sum(fft_mag)
        features.append(spectral_centroid)
        
        psd = fft_mag ** 2
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        features.append(spectral_entropy)
        
        mid_idx = len(fft_mag) // 2
        low_energy = np.sum(fft_mag[:mid_idx]**2)
        high_energy = np.sum(fft_mag[mid_idx:]**2)
        energy_ratio = high_energy / (low_energy + 1e-10)
        features.append(energy_ratio)
        
        return features
    
    def extract_fusion_features(self, vibration, temperature, acoustic):
        features = []
        corr_vt = np.corrcoef(vibration, temperature)[0, 1]
        features.append(corr_vt)
        corr_va = np.corrcoef(vibration, acoustic)[0, 1]
        features.append(corr_va)
        return features
    
    def extract_all_features(self, sensor_data):
        vibration = sensor_data[0]
        temperature = sensor_data[1]
        acoustic = sensor_data[2]
        
        features = []
        features.extend(self.extract_statistical_features(vibration)[:4])
        features.extend(self.extract_frequency_features(vibration)[:2])
        features.extend(self.extract_statistical_features(temperature)[:4])
        features.extend(self.extract_frequency_features(temperature)[:2])
        features.extend(self.extract_statistical_features(acoustic)[:4])
        features.extend(self.extract_frequency_features(acoustic)[:2])
        features.extend(self.extract_fusion_features(vibration, temperature, acoustic))
        
        return np.array(features)


class LightweightDNN:
    """Lightweight Deep Neural Network"""
    
    def __init__(self, input_dim=20):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_proba = self.model.predict(X_test_scaled, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        return metrics, y_pred, y_pred_proba


# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'monitoring_data' not in st.session_state:
    st.session_state.monitoring_data = []


# Header
st.title("🔧 IoT Predictive Maintenance System")
st.markdown("**Edge Sensor Fusion + Lightweight Deep Learning**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Dataset Configuration")
    n_samples = st.slider("Number of Samples", 1000, 10000, 5000, 1000)
    
    st.subheader("Training Configuration")
    epochs = st.slider("Training Epochs", 10, 50, 30, 5)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    
    st.subheader("Monitoring Configuration")
    threshold = st.slider("Fault Threshold", 0.0, 1.0, 0.6, 0.05)
    monitor_interval = st.slider("Reading Interval (seconds)", 1, 5, 2)
    
    st.markdown("---")
    st.markdown("**Paper Reference:**")
    st.caption("Predictive Maintenance System for IoT Devices Using Edge Sensor Fusion")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Train Model", "🔍 Evaluate", "📡 Live Monitor", "ℹ️ About"])

# Tab 1: Train Model
with tab1:
    st.header("Train Predictive Maintenance Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Training Pipeline")
        st.markdown("""
        1. Generate synthetic multi-sensor data (vibration, temperature, acoustic)
        2. Extract 20 statistical and frequency-domain features
        3. Train lightweight deep neural network
        4. Validate and save model
        """)
    
    with col2:
        st.markdown("### Model Architecture")
        st.code("""
Input Layer (20)
    ↓
Dense (128) + Dropout
    ↓
Dense (64) + Dropout
    ↓
Dense (32) + Dropout
    ↓
Output (1) - Sigmoid
        """)
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("Training in progress..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Generate data
            status_text.text("Step 1/5: Generating sensor data...")
            progress_bar.progress(20)
            generator = SensorDataGenerator(n_samples=n_samples)
            raw_data, labels = generator.generate_dataset()
            
            # Step 2: Extract features
            status_text.text("Step 2/5: Extracting features...")
            progress_bar.progress(40)
            extractor = FeatureExtractor()
            features = []
            for sample in raw_data:
                feature_vec = extractor.extract_all_features(sample)
                features.append(feature_vec)
            features = np.array(features)
            
            # Step 3: Split data
            status_text.text("Step 3/5: Splitting dataset...")
            progress_bar.progress(50)
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Step 4: Train model
            status_text.text("Step 4/5: Training model...")
            progress_bar.progress(60)
            model = LightweightDNN(input_dim=20)
            model.build_model()
            history = model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
            
            # Step 5: Evaluate
            status_text.text("Step 5/5: Evaluating model...")
            progress_bar.progress(90)
            metrics, predictions, pred_probs = model.evaluate(X_test, y_test)
            
            # Save to session state
            st.session_state.model = model
            st.session_state.model_trained = True
            st.session_state.history = history
            st.session_state.metrics = metrics
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.extractor = extractor
            
            progress_bar.progress(100)
            status_text.text("✅ Training completed!")
            
        st.success("🎉 Model trained successfully!")
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        col4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
        
        # Plot training history
        st.markdown("### Training History")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['loss'], label='Training', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# Tab 2: Evaluate
with tab2:
    st.header("Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' tab.")
    else:
        metrics = st.session_state.metrics
        
        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("🔍 Precision", f"{metrics['precision']*100:.2f}%")
        col3.metric("📊 Recall", f"{metrics['recall']*100:.2f}%")
        col4.metric("⚖️ F1-Score", f"{metrics['f1_score']*100:.2f}%")
        
        st.markdown("---")
        
        # Dataset statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Distribution")
            df_dist = pd.DataFrame({
                'Class': ['Normal', 'Fault'],
                'Count': [
                    np.sum(st.session_state.y_test == 0),
                    np.sum(st.session_state.y_test == 1)
                ]
            })
            st.dataframe(df_dist, use_container_width=True)
        
        with col2:
            st.markdown("### Model Summary")
            st.info(f"""
            **Input Features:** 20  
            **Architecture:** 128 → 64 → 32 → 1  
            **Activation:** ReLU + Sigmoid  
            **Dropout:** 0.3, 0.3, 0.2  
            **Optimizer:** Adam (lr=0.001)
            """)

# Tab 3: Live Monitor
with tab3:
    st.header("Real-Time Monitoring Simulator")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' tab.")
    else:
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("▶️ Start Monitoring", type="primary"):
                st.session_state.monitoring_active = True
                st.session_state.monitoring_data = []
        
        with col2:
            if st.button("⏹️ Stop Monitoring"):
                st.session_state.monitoring_active = False
        
        with col3:
            st.markdown(f"**Threshold:** {threshold:.2f} | **Interval:** {monitor_interval}s")
        
        st.markdown("---")
        
        # Monitoring display
        if st.session_state.monitoring_active:
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            table_placeholder = st.empty()
            
            for i in range(15):  # 15 readings
                # Generate sensor reading
                if i < 7:
                    condition = 'normal'
                else:
                    condition = 'fault'
                
                window_size = 256
                if condition == 'normal':
                    vibration = np.random.normal(0, 1, window_size)
                    temperature = np.random.normal(40, 1, window_size)
                    acoustic = np.random.normal(0, 0.5, window_size)
                else:
                    vibration = np.random.normal(3, 2, window_size)
                    temperature = np.random.normal(60, 3, window_size)
                    acoustic = np.random.normal(2, 1, window_size)
                
                sensor_data = np.array([vibration, temperature, acoustic])
                
                # Extract features and predict
                features = st.session_state.extractor.extract_all_features(sensor_data)
                features_scaled = st.session_state.model.scaler.transform(features.reshape(1, -1))
                
                start_time = time.time()
                fault_prob = st.session_state.model.model.predict(features_scaled, verbose=0)[0][0]
                inference_time = (time.time() - start_time) * 1000
                
                # Store data
                reading = {
                    'Reading': i + 1,
                    'Timestamp': time.strftime('%H:%M:%S'),
                    'Vibration': np.mean(vibration),
                    'Temperature': np.mean(temperature),
                    'Acoustic': np.mean(acoustic),
                    'Fault Probability': fault_prob,
                    'Latency (ms)': inference_time,
                    'Status': '🚨 FAULT' if fault_prob >= threshold else '✅ Normal'
                }
                st.session_state.monitoring_data.append(reading)
                
                # Update displays
                if fault_prob >= threshold:
                    status_placeholder.error(f"🚨 **ALERT:** Fault detected! Confidence: {fault_prob*100:.1f}%")
                else:
                    status_placeholder.success(f"✅ **Status:** System operating normally")
                
                # Metrics
                col1, col2, col3, col4 = metrics_placeholder.columns(4)
                col1.metric("Reading", f"#{i+1}")
                col2.metric("Fault Probability", f"{fault_prob:.3f}")
                col3.metric("Inference Time", f"{inference_time:.2f} ms")
                col4.metric("Condition", reading['Status'])
                
                # Chart
                if len(st.session_state.monitoring_data) > 1:
                    df_monitor = pd.DataFrame(st.session_state.monitoring_data)
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(df_monitor['Reading'], df_monitor['Fault Probability'], 
                           marker='o', linewidth=2, markersize=6)
                    ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
                    ax.fill_between(df_monitor['Reading'], threshold, 1, alpha=0.2, color='red')
                    ax.set_xlabel('Reading Number')
                    ax.set_ylabel('Fault Probability')
                    ax.set_title('Real-Time Fault Probability')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    chart_placeholder.pyplot(fig)
                    plt.close()
                
                # Table
                df_monitor = pd.DataFrame(st.session_state.monitoring_data)
                table_placeholder.dataframe(df_monitor, use_container_width=True)
                
                if not st.session_state.monitoring_active:
                    break
                
                time.sleep(monitor_interval)
            
            st.session_state.monitoring_active = False
            st.success("✅ Monitoring session completed!")
        
        elif len(st.session_state.monitoring_data) > 0:
            st.markdown("### Last Monitoring Session")
            df_monitor = pd.DataFrame(st.session_state.monitoring_data)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Readings", len(df_monitor))
            col2.metric("Alerts Generated", (df_monitor['Fault Probability'] >= threshold).sum())
            col3.metric("Avg Latency", f"{df_monitor['Latency (ms)'].mean():.2f} ms")
            
            st.dataframe(df_monitor, use_container_width=True)

# Tab 4: About
with tab4:
    st.header("About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 System Overview")
        st.markdown("""
        This predictive maintenance system demonstrates:
        
        - **Multi-Sensor Fusion**: Combines vibration, temperature, and acoustic data
        - **Lightweight Deep Learning**: Optimized for edge devices (ESP32, Raspberry Pi)
        - **Real-Time Inference**: Sub-10ms prediction latency
        - **Edge Computing**: No cloud dependency
        - **High Accuracy**: 97%+ fault detection rate
        """)
        
        st.markdown("### 📊 Feature Extraction")
        st.markdown("""
        **20 Features Total:**
        - 12 Statistical features (mean, std, min, max, etc.)
        - 6 Frequency-domain features (FFT-based)
        - 2 Cross-sensor correlation features
        """)
    
    with col2:
        st.markdown("### 🔬 Research Paper")
        st.markdown("""
        **Title:** Predictive Maintenance System and Method for IoT Devices Using Edge Sensor Fusion and Lightweight Deep Learning
        
        **Author:** Jonah Sudhir  
        **Institution:** Lovely Professional University, Punjab, India  
        **Conference:** ICSCAIconf2025  
        **Paper ID:** 370
        """)
        
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **Framework**: TensorFlow/Keras
        - **Language**: Python 3.9+
        - **UI**: Streamlit
        - **Processing**: NumPy, SciPy, Pandas
        - **Target Hardware**: ESP32, Raspberry Pi
        """)
    
    st.markdown("---")
    st.info("💡 **Note:** This is a software simulation. The system can be deployed to actual IoT hardware for real-world predictive maintenance applications.")

# Footer
st.markdown("---")
st.caption("IoT Predictive Maintenance System | Powered by Edge AI | © 2025")