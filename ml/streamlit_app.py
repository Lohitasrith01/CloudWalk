import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort
import json
import io
import time
from typing import Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal
from audio_recorder_streamlit import audio_recorder

# Load model and preprocessing config
@st.cache_resource
def load_model_and_config():
    """Load ONNX model and preprocessing configuration"""
    try:
        # Load ONNX model
        import os
        
        # Debug: Show current working directory and file location
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        model_path = os.path.join(current_dir, "assets", "digit_cnn.onnx")
        config_path = os.path.join(current_dir, "assets", "preprocess.json")
        
        # Verify files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return session, config
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.exception(e)
        return None, None

def simple_vad(y: np.ndarray, sr: int, energy_threshold: float = 0.001, spectral_threshold: float = 0.15) -> tuple[bool, dict]:
    """
    Improved Voice Activity Detection with debug info
    
    Args:
        y: Input audio signal
        sr: Sample rate
        energy_threshold: Minimum RMS energy for speech (lowered default)
        spectral_threshold: Minimum spectral centroid ratio for speech (lowered default)
    
    Returns:
        Tuple of (is_speech_detected, debug_info)
    """
    # Energy-based detection
    rms_energy = np.sqrt(np.mean(y**2))
    energy_pass = rms_energy >= energy_threshold
    
    # Spectral centroid-based detection (speech has energy in mid frequencies)
    try:
        stft = librosa.stft(y, n_fft=min(512, len(y)//2), hop_length=128)
        spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft), sr=sr)[0]
        avg_centroid = np.mean(spectral_centroids)
        
        # For 8kHz audio, Nyquist is 4kHz. Speech centroids are typically 800-2500Hz
        # So we expect centroid_ratio between 0.2-0.6
        centroid_ratio = avg_centroid / (sr / 2)
        spectral_pass = centroid_ratio >= spectral_threshold
    except:
        # If spectral analysis fails, just use energy
        avg_centroid = 0
        centroid_ratio = 0
        spectral_pass = True
    
    # More lenient: pass if EITHER energy OR spectral condition is met (not both)
    is_speech = energy_pass or spectral_pass
    
    # Debug info
    debug_info = {
        'rms_energy': rms_energy,
        'energy_threshold': energy_threshold,
        'energy_pass': energy_pass,
        'avg_centroid': avg_centroid,
        'centroid_ratio': centroid_ratio,
        'spectral_threshold': spectral_threshold,
        'spectral_pass': spectral_pass,
        'final_decision': is_speech
    }
    
    return is_speech, debug_info

def spectral_subtraction_denoise(y: np.ndarray, sr: int, alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
    """
    Basic spectral subtraction noise reduction
    
    Args:
        y: Input audio signal
        sr: Sample rate
        alpha: Over-subtraction factor
        beta: Spectral floor factor
    """
    # STFT parameters
    n_fft = 512
    hop_length = 128
    
    # Compute STFT
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise from first 0.1 seconds (assume silence)
    noise_frames = int(0.1 * sr / hop_length)
    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Spectral subtraction
    enhanced_magnitude = magnitude - alpha * noise_spectrum
    
    # Apply spectral floor
    enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
    
    # Reconstruct signal
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
    
    return enhanced_audio.astype(np.float32)

def extract_mel_features(y: np.ndarray, sr: int, config: dict) -> np.ndarray:
    """Extract mel spectrogram features matching training preprocessing exactly"""
    preproc = config["preprocessing"]
    
    # CRITICAL: Ensure we're using the exact same sample rate as training
    target_sr = preproc["sample_rate"]  # Should be 8000 Hz
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # CRITICAL: Pad or trim to exact target duration (1.0 second)
    target_len = int(sr * preproc["target_duration_seconds"])
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    
    # CRITICAL: Extract mel spectrogram with EXACT same parameters as training
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=preproc["n_fft"],
        hop_length=preproc["hop_length"],
        n_mels=preproc["n_mels"],
        fmin=preproc["fmin"],
        fmax=preproc["fmax"],
        power=2.0,  # Same as training
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # Same as training
    
    # CRITICAL: Normalize using training statistics
    S_db = (S_db - config["mean"]) / config["std"]
    
    # CRITICAL: Return in exact same format as training expects: (1, 1, n_mels, time)
    return S_db[None, None, :, :].astype(np.float32)

def predict_digit(session: ort.InferenceSession, features: np.ndarray) -> Tuple[int, np.ndarray]:
    """Run inference and return predicted digit and confidence scores"""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features})
    logits = outputs[0][0]  # Remove batch dimension
    
    # Apply softmax to get probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits))
    predicted_digit = int(np.argmax(probs))
    
    return predicted_digit, probs

def sliding_window_predict(session: ort.InferenceSession, audio_data: np.ndarray, sr: int, config: dict, 
                          window_size: float = 1.0, step_size: float = 0.3, confidence_threshold: float = 0.6) -> list:
    """
    Apply sliding window approach to predict digit sequence
    
    Args:
        session: ONNX inference session
        audio_data: Input audio signal
        sr: Sample rate
        config: Preprocessing config
        window_size: Window size in seconds (default 1.0s to match training)
        step_size: Step size in seconds (default 0.3s for overlap)
        confidence_threshold: Minimum confidence to keep prediction
    
    Returns:
        List of predictions with metadata
    """
    window_samples = int(window_size * sr)
    step_samples = int(step_size * sr)
    predictions = []
    
    # Apply sliding window
    for start in range(0, len(audio_data) - window_samples + 1, step_samples):
        end = start + window_samples
        window_audio = audio_data[start:end]
        
        # Extract features for this window
        features = extract_mel_features(window_audio, sr, config)
        predicted_digit, probs = predict_digit(session, features)
        max_confidence = float(probs[predicted_digit])
        
        # Only keep high-confidence predictions
        if max_confidence >= confidence_threshold:
            predictions.append({
                'start_time': start / sr,
                'end_time': end / sr,
                'digit': predicted_digit,
                'confidence': max_confidence,
                'probabilities': probs.copy()
            })
    
    return predictions

def post_process_sequence(predictions: list, min_gap: float = 0.2) -> list:
    """
    Post-process sliding window predictions to create clean sequence
    
    Args:
        predictions: List of prediction dictionaries
        min_gap: Minimum gap between different digits (seconds)
    
    Returns:
        Cleaned sequence of digits
    """
    if not predictions:
        return []
    
    # Sort by start time
    predictions.sort(key=lambda x: x['start_time'])
    
    # Group overlapping predictions of the same digit
    grouped = []
    current_group = [predictions[0]]
    
    for pred in predictions[1:]:
        last_pred = current_group[-1]
        
        # If same digit and overlapping/close in time, add to current group
        if (pred['digit'] == last_pred['digit'] and 
            pred['start_time'] - last_pred['end_time'] < min_gap):
            current_group.append(pred)
        else:
            # Finalize current group and start new one
            grouped.append(current_group)
            current_group = [pred]
    
    # Don't forget the last group
    grouped.append(current_group)
    
    # For each group, take the prediction with highest confidence
    final_sequence = []
    for group in grouped:
        best_pred = max(group, key=lambda x: x['confidence'])
        final_sequence.append(best_pred)
    
    return final_sequence

def create_confidence_plot(probs: np.ndarray, predicted_digit: int):
    """Create a bar plot showing confidence for each digit with dark theme"""
    colors = ['#FFD700' if i == predicted_digit else 'rgba(255, 215, 0, 0.3)' for i in range(10)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(10)),
            y=probs,
            marker_color=colors,
            marker_line=dict(color='#FFA500', width=2),
            text=[f'{p:.1%}' for p in probs],
            textposition='auto',
            textfont=dict(color='black', size=12, family='Orbitron'),
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="🎯 Digit Confidence Scores",
            font=dict(color='#FFD700', size=18, family='Orbitron')
        ),
        xaxis_title="Digit",
        yaxis_title="Confidence",
        xaxis=dict(
            color='#FFD700',
            gridcolor='rgba(255, 215, 0, 0.2)',
            title_font=dict(color='#FFD700', family='Orbitron')
        ),
        yaxis=dict(
            range=[0, 1],
            color='#FFD700',
            gridcolor='rgba(255, 215, 0, 0.2)',
            title_font=dict(color='#FFD700', family='Orbitron')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        showlegend=False,
        font=dict(color='#FFD700', family='Orbitron')
    )
    
    return fig

def create_spectrogram_plot(features: np.ndarray):
    """Create a spectrogram visualization with dark theme"""
    # Remove batch and channel dimensions
    spec = features[0, 0, :, :]
    
    # Create custom colorscale (black to gold)
    colorscale = [
        [0.0, '#000000'],
        [0.2, '#1a1a1a'], 
        [0.4, '#2d1810'],
        [0.6, '#8B4513'],
        [0.8, '#FFA500'],
        [1.0, '#FFD700']
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=spec,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(text="dB", font=dict(color='#FFD700', family='Orbitron')),
            tickfont=dict(color='#FFD700', family='Orbitron'),
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='#FFD700',
            borderwidth=1
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="🌊 Mel Spectrogram Analysis",
            font=dict(color='#FFD700', size=18, family='Orbitron')
        ),
        xaxis_title="Time Frames",
        yaxis_title="Mel Frequency Bins",
        xaxis=dict(
            color='#FFD700',
            gridcolor='rgba(255, 215, 0, 0.2)',
            title_font=dict(color='#FFD700', family='Orbitron')
        ),
        yaxis=dict(
            color='#FFD700',
            gridcolor='rgba(255, 215, 0, 0.2)',
            title_font=dict(color='#FFD700', family='Orbitron')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        font=dict(color='#FFD700', family='Orbitron')
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="🎤 AI Digit Recognition", 
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for ash + green theme
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #111827 0%, #1f2937 25%, #0f172a 50%, #1f2937 75%, #111827 100%);
        background-attachment: fixed;
    }
    
    /* Add subtle pattern overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(16, 185, 129, 0.10) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(52, 211, 153, 0.06) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Main title styling */
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #34D399, #10B981, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(16, 185, 129, 0.5);
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
        to { text-shadow: 0 0 40px rgba(16, 185, 129, 0.8); }
    }
    
    .subtitle {
        font-family: 'Exo 2', sans-serif;
        font-size: 1.2rem;
        text-align: center;
        color: #C0C0C0;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Mode selection styling */
    .stRadio > div {
        background: rgba(16, 185, 129, 0.12);
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid rgba(16, 185, 129, 0.35);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
        border-right: 2px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(0, 0, 0, 0.8) 100%);
        border: 2px solid rgba(16, 185, 129, 0.35);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #34D399, #10B981);
        color: #0b1220;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-family: 'Exo 2', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #10B981, #34D399);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
        transform: translateY(-2px);
    }
    
    /* Audio recorder styling */
    .stAudio {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(0, 0, 0, 0.9) 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px solid rgba(16, 185, 129, 0.4);
        margin: 1rem 0;
    }
    
    /* Enhanced audio controls */
    .stAudio > div {
        background: transparent !important;
    }
    
    /* Audio player controls visibility */
    audio {
        width: 100%;
        height: 50px;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    audio::-webkit-media-controls-panel {
        background-color: rgba(16, 185, 129, 0.2);
        border-radius: 8px;
    }
    
    audio::-webkit-media-controls-play-button,
    audio::-webkit-media-controls-pause-button {
        background-color: #34D399;
        border-radius: 50%;
        margin: 0 8px;
    }
    
    audio::-webkit-media-controls-timeline {
        background-color: rgba(16, 185, 129, 0.3);
        border-radius: 5px;
        margin: 0 8px;
    }
    
    /* Success/Warning/Info boxes */
    .stSuccess {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.12));
        border-left: 4px solid #10B981;
    }
    
    .stWarning {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.1), rgba(52, 211, 153, 0.08));
        border-left: 4px solid #F59E0B;
    }
    
    .stInfo {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.08), rgba(16, 185, 129, 0.06));
        border-left: 4px solid #10B981;
    }
    
    /* Headers */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #A7B1BB;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.25);
    }
    
    /* Sequence display */
    .sequence-display {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #34D399, #10B981, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 2rem;
        border: 3px solid rgba(16, 185, 129, 0.5);
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
        animation: pulse 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        from { box-shadow: 0 0 20px rgba(255, 215, 0, 0.3); }
        to { box-shadow: 0 0 40px rgba(255, 215, 0, 0.6); }
    }
    
    /* Plotly charts dark theme */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1f2937;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #34D399, #10B981);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #10B981, #34D399);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Animated title with particles effect
    st.markdown("""
    <div class="main-title">
        🎤 AI DIGIT RECOGNITION
    </div>
    <div class="subtitle">
        ✨ Advanced Neural Network • Real-time Audio Processing • Sliding Window Analysis ✨
    </div>
    """, unsafe_allow_html=True)
    
    # Mode selection
    mode = st.radio(
        "**Recognition Mode:**",
        ["Single Digit", "Sequence (Sliding Window)", "🔮 Real-time Orb"],
        horizontal=True,
        help="Single Digit: Record one digit at a time. Sequence: Record multiple digits in sequence. Real-time Orb: Interactive orb that changes colors as you speak!"
    )
    
    # Load model
    session, config = load_model_and_config()
    if session is None or config is None:
        st.error("Failed to load model. Please ensure the model files exist.")
        return
    
    # Real-time Orb Mode
    if mode == "🔮 Real-time Orb":
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="color: #FFD700; font-family: 'Orbitron', monospace;">
                🔮 Real-time Interactive Orb
            </h2>
            <p style="color: #C0C0C0; font-family: 'Exo 2', sans-serif; font-size: 1.1rem;">
                Click the orb to start recording. It will change colors in real-time as you speak digits!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions for orb mode
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("🚀 **Quick Setup:** Run `python realtime_server.py` in terminal, then click the orb below!")
            
            # Color legend
            st.markdown("""
            <div style="background: rgba(255, 215, 0, 0.1); border-radius: 15px; padding: 1.5rem; margin: 1rem 0;">
                <h4 style="color: #FFD700; text-align: center; margin-bottom: 1rem;">🎨 Digit Color Map</h4>
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; text-align: center;">
                    <div><span style="color: #FF0000;">⚫ 0</span></div>
                    <div><span style="color: #FF7F00;">⚫ 1</span></div>
                    <div><span style="color: #FFFF00;">⚫ 2</span></div>
                    <div><span style="color: #7FFF00;">⚫ 3</span></div>
                    <div><span style="color: #00FF00;">⚫ 4</span></div>
                    <div><span style="color: #00FF7F;">⚫ 5</span></div>
                    <div><span style="color: #00FFFF;">⚫ 6</span></div>
                    <div><span style="color: #007FFF;">⚫ 7</span></div>
                    <div><span style="color: #0000FF;">⚫ 8</span></div>
                    <div><span style="color: #7F00FF;">⚫ 9</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Embed the orb
        st.components.v1.iframe("http://localhost:8001/orb", height=600, scrolling=False)
        
        return  # Skip the rest of the app
    
    # Sidebar controls
    st.sidebar.header("Settings")
    enable_noise_reduction = st.sidebar.checkbox("Enable Noise Reduction", value=True)
    noise_alpha = st.sidebar.slider("Noise Reduction Strength", 1.0, 5.0, 2.0, 0.5)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.7, 0.05)  # Higher default
    
    # Sequence mode settings
    if mode == "Sequence (Sliding Window)":
        st.sidebar.markdown("**Sequence Settings**")
        window_step = st.sidebar.slider("Window Step Size (seconds)", 0.1, 0.8, 0.3, 0.1)
        seq_confidence_threshold = st.sidebar.slider("Sequence Confidence Threshold", 0.3, 0.9, 0.6, 0.05)
        min_gap = st.sidebar.slider("Minimum Gap Between Digits (seconds)", 0.1, 0.5, 0.2, 0.05)
    
    # VAD settings
    st.sidebar.markdown("**Voice Activity Detection**")
    enable_vad = st.sidebar.checkbox("Enable VAD (Filter Non-Speech)", value=False)  # Disabled by default
    energy_threshold = st.sidebar.slider("Energy Threshold", 0.0001, 0.01, 0.001, 0.0001)  # Much more sensitive
    spectral_threshold = st.sidebar.slider("Spectral Threshold", 0.05, 0.5, 0.15, 0.05)  # Lower default
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Instructions:**")
    if mode == "Single Digit":
        st.sidebar.markdown("1. Click 'Start Recording'")
        st.sidebar.markdown("2. Speak a single digit clearly")
        st.sidebar.markdown("3. Click 'Stop Recording'")
        st.sidebar.markdown("4. View the prediction results")
    else:
        st.sidebar.markdown("1. Click 'Start Recording'")
        st.sidebar.markdown("2. Speak multiple digits with pauses")
        st.sidebar.markdown("3. Example: 'one... two... three'")
        st.sidebar.markdown("4. Click 'Stop Recording'")
        st.sidebar.markdown("5. View the sequence results")
    
    st.header("🎙️ Record Audio")
    
    # Initialize session state for recording control
    if 'recorder_key' not in st.session_state:
        st.session_state.recorder_key = 0
    
    # Clear instructions for click-to-record
    if mode == "Single Digit":
        st.info("🎯 **Single Digit Mode** - Record one digit at a time")
        st.markdown("**Instructions:** Click the microphone to **start recording**, speak a digit clearly, then click again to **stop recording**.")
    else:
        st.info("🔄 **Sequence Mode** - Record multiple digits with sliding window analysis")
        st.markdown("**Instructions:** Click the microphone to **start recording**, speak multiple digits with short pauses between them (e.g., 'one... two... three'), then click to **stop recording**.")
    
    # Audio recorder
    audio_bytes = audio_recorder(
        text="Click to Record",
        recording_color="#22c55e",
        neutral_color="#065f46",
        icon_name="microphone",
        icon_size="2x",
        sample_rate=16000,
        auto_start=False,
        key=f"audio_recorder_{st.session_state.recorder_key}"
    )
    
    if st.button("🔄 Reset Recording"):
        st.session_state.recorder_key += 1
        st.rerun()
    
    if audio_bytes:
        # Process the recorded audio
        try:
            # Convert bytes to audio array
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # CRITICAL: Always resample to model's expected sample rate (8000 Hz)
            target_sr = config["preprocessing"]["sample_rate"]
            if sr != target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                st.info(f"Resampled audio from {sr} Hz to {target_sr} Hz to match training data")
            
            # Display audio info with better formatting
            duration = len(audio_data)/sr
            st.success(f"✅ Audio recorded successfully!")
            
            # Enhanced audio player section
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(0, 0, 0, 0.8) 100%);
                        border: 2px solid rgba(16, 185, 129, 0.4);
                        border-radius: 15px;
                        padding: 1.5rem;
                        margin: 1rem 0;">
                <h4 style="color: #34D399; font-family: 'Orbitron', monospace; margin: 0 0 1rem 0;">
                    🎵 Recorded Audio Playback
                </h4>
                <div style="color: #A7B1BB; font-family: 'Exo 2', sans-serif; margin-bottom: 1rem;">
                    📊 Duration: <span style="color: #34D399; font-weight: bold;">{:.2f} seconds</span> | 
                    🔊 Sample Rate: <span style="color: #34D399; font-weight: bold;">{} Hz</span> | 
                    📈 Quality: <span style="color: #34D399; font-weight: bold;">16-bit</span>
                </div>
            </div>
            """.format(duration, sr), unsafe_allow_html=True)
            
            # Audio player with better visibility
            st.audio(audio_bytes, format="audio/wav", start_time=0)
            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
            
            # Voice Activity Detection
            if enable_vad:
                is_speech, vad_debug = simple_vad(audio_data, sr, energy_threshold, spectral_threshold)
                
                # Show VAD debug info in sidebar
                st.sidebar.markdown("**VAD Debug Info:**")
                st.sidebar.write(f"RMS Energy: {vad_debug['rms_energy']:.4f} (threshold: {vad_debug['energy_threshold']:.4f})")
                st.sidebar.write(f"Energy Pass: {'✅' if vad_debug['energy_pass'] else '❌'}")
                st.sidebar.write(f"Spectral Centroid: {vad_debug['avg_centroid']:.0f} Hz")
                st.sidebar.write(f"Centroid Ratio: {vad_debug['centroid_ratio']:.3f} (threshold: {vad_debug['spectral_threshold']:.3f})")
                st.sidebar.write(f"Spectral Pass: {'✅' if vad_debug['spectral_pass'] else '❌'}")
                st.sidebar.write(f"**Final Decision: {'✅ Speech' if is_speech else '❌ No Speech'}**")
                
                if not is_speech:
                    st.warning("⚠️ No speech detected! Try adjusting VAD settings in the sidebar or speaking louder/clearer.")
                    st.info("💡 **Tip**: Lower the Energy Threshold or Spectral Threshold in the sidebar if VAD is too strict.")
                    st.stop()
                else:
                    st.success("✅ Speech detected")
            
            # Apply noise reduction if enabled
            if enable_noise_reduction:
                with st.spinner("Applying noise reduction..."):
                    audio_data = spectral_subtraction_denoise(audio_data, sr, alpha=noise_alpha)
                st.info("✅ Noise reduction applied")
            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
            
            # Extract features and predict based on mode
            if mode == "Single Digit":
                with st.spinner("Processing audio and making prediction..."):
                    features = extract_mel_features(audio_data, sr, config)
                    predicted_digit, probs = predict_digit(session, features)
                    max_confidence = float(probs[predicted_digit])
                
                    # Display single digit results heading once later
            else:
                with st.spinner("Analyzing sequence with sliding window..."):
                    # Apply sliding window
                    raw_predictions = sliding_window_predict(
                        session, audio_data, sr, config,
                        step_size=window_step,
                        confidence_threshold=seq_confidence_threshold
                    )
                    
                    # Post-process to get clean sequence
                    sequence = post_process_sequence(raw_predictions, min_gap=min_gap)
                
                # Display sequence results
                st.header("🔍 Sequence Results")
                
                if sequence:
                    # Show the detected sequence with custom styling
                    sequence_digits = [str(pred['digit']) for pred in sequence]
                    sequence_str = " → ".join(sequence_digits)
                    
                    st.markdown(f"""
                    <div class="sequence-display">
                        {sequence_str}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show timing and confidence for each digit with custom styling
                    st.subheader("📊 Sequence Details")
                    
                    cols = st.columns(min(len(sequence), 5))  # Max 5 columns
                    for i, pred in enumerate(sequence):
                        with cols[i % 5]:
                            st.markdown(f"""
                            <div class="metric-container">
                                <h3 style="color: #FFD700; font-family: 'Orbitron', monospace; margin: 0;">
                                    Digit {i+1}
                                </h3>
                                <div style="font-size: 2.5rem; font-weight: bold; color: #FFD700; font-family: 'Orbitron', monospace;">
                                    {pred['digit']}
                                </div>
                                <div style="color: #FFA500; font-size: 1.2rem; font-family: 'Exo 2', sans-serif;">
                                    {pred['confidence']:.1%} confidence
                                </div>
                                <div style="color: #C0C0C0; font-size: 0.9rem; margin-top: 0.5rem;">
                                    ⏱️ {pred['start_time']:.1f}s - {pred['end_time']:.1f}s
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Show all raw detections for debugging
                    expander = st.expander("🔧 Debug: All Raw Detections")
                    with expander:
                        for i, pred in enumerate(raw_predictions):
                            st.write(f"Window {i+1}: Digit {pred['digit']} ({pred['confidence']:.1%}) at {pred['start_time']:.1f}s-{pred['end_time']:.1f}s")
                else:
                    st.warning("⚠️ No digits detected in sequence. Try speaking louder or adjusting the confidence threshold.")
                    
                return  # Skip single digit display for sequence mode
            
            # Single digit results display
            st.header("🔍 Prediction Results")
            
            # Create columns for results
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                # Predicted digit with custom styling
                confidence_color = "#00FF00" if max_confidence >= confidence_threshold else "#FFA500"
                st.markdown(f"""
                <div class="metric-container" style="border-color: {confidence_color};">
                    <h3 style="color: #FFD700; font-family: 'Orbitron', monospace; margin: 0;">
                        🎯 Predicted Digit
                    </h3>
                    <div style="font-size: 4rem; font-weight: bold; color: #FFD700; font-family: 'Orbitron', monospace; text-shadow: 0 0 20px rgba(255, 215, 0, 0.8);">
                        {predicted_digit}
                    </div>
                    <div style="color: {confidence_color}; font-size: 1.5rem; font-family: 'Exo 2', sans-serif;">
                        {max_confidence:.1%} confidence
                    </div>
                    {"<div style='color: #FFA500; font-size: 0.9rem; margin-top: 0.5rem;'>⚠️ Low confidence prediction</div>" if max_confidence < confidence_threshold else ""}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Audio statistics with custom styling
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #FFD700; font-family: 'Orbitron', monospace; margin: 0;">
                        📊 Audio Stats
                    </h3>
                    <div style="margin: 1rem 0; font-family: 'Exo 2', sans-serif;">
                        <div style="color: #C0C0C0; margin: 0.5rem 0;">
                            🕐 Duration: <span style="color: #FFD700;">{len(audio_data)/sr:.2f}s</span>
                        </div>
                        <div style="color: #C0C0C0; margin: 0.5rem 0;">
                            🔊 Sample Rate: <span style="color: #FFD700;">{sr} Hz</span>
                        </div>
                        <div style="color: #C0C0C0; margin: 0.5rem 0;">
                            ⚡ RMS Energy: <span style="color: #FFD700;">{np.sqrt(np.mean(audio_data**2)):.3f}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Confidence plot
                fig_conf = create_confidence_plot(probs, predicted_digit)
                st.plotly_chart(fig_conf, use_container_width=True)
            
            # Spectrogram visualization
            st.header("📊 Audio Analysis")
            fig_spec = create_spectrogram_plot(features)
            st.plotly_chart(fig_spec, use_container_width=True)
            
            # Detailed confidence breakdown
            st.header("📈 Detailed Confidence Scores")
            confidence_df = {
                "Digit": list(range(10)),
                "Confidence": [f"{p:.3f}" for p in probs],
                "Percentage": [f"{p:.1%}" for p in probs]
            }
            st.table(confidence_df)
            
            # Model insights
            st.header("🧠 Model Insights")
            
            # Find top 3 predictions
            top_indices = np.argsort(probs)[-3:][::-1]
            
            for i, idx in enumerate(top_indices):
                if i == 0:
                    st.write(f"🥇 **Most likely:** {idx} ({probs[idx]:.1%})")
                elif i == 1:
                    st.write(f"🥈 **Second choice:** {idx} ({probs[idx]:.1%})")
                else:
                    st.write(f"🥉 **Third choice:** {idx} ({probs[idx]:.1%})")
            
            # Uncertainty analysis
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            st.write(f"**Prediction uncertainty:** {entropy:.3f}")
            if entropy > 1.5:
                st.warning("High uncertainty - the model is not very confident about this prediction")
            elif entropy < 0.5:
                st.success("Low uncertainty - the model is very confident about this prediction")
            
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            st.exception(e)
    
    # Model information
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Info:**")
    if config:
        st.sidebar.write(f"Sample Rate: {config['preprocessing']['sample_rate']} Hz")
        st.sidebar.write(f"Mel Bins: {config['preprocessing']['n_mels']}")
        st.sidebar.write(f"FFT Size: {config['preprocessing']['n_fft']}")

if __name__ == "__main__":
    main()
