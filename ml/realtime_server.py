import asyncio
import json
import time
import numpy as np
import onnxruntime as ort
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os

# Load your trained model
model_path = os.path.join("assets", "digit_cnn.onnx")
config_path = os.path.join("assets", "preprocess.json")

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
with open(config_path, 'r') as f:
    config = json.load(f)

app = FastAPI()

def extract_mel_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract mel features exactly like training"""
    preproc = config["preprocessing"]
    
    # Pad or trim to exact target duration (1.0 second)
    target_len = int(sr * preproc["target_duration_seconds"])
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    
    # Extract mel spectrogram with EXACT same parameters as training
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=preproc["n_fft"],
        hop_length=preproc["hop_length"],
        n_mels=preproc["n_mels"],
        fmin=preproc["fmin"],
        fmax=preproc["fmax"],
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalize using training statistics
    S_db = (S_db - config["mean"]) / config["std"]
    
    # Return in exact same format as training expects: (1, 1, n_mels, time)
    return S_db[None, None, :, :].astype(np.float32)

def predict_digit(audio_window: np.ndarray) -> tuple:
    """Predict digit from audio window"""
    try:
        features = extract_mel_features(audio_window, 8000)  # 8kHz sample rate
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: features})
        logits = outputs[0][0]  # Remove batch dimension
        
        # Apply softmax to get probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits))
        predicted_digit = int(np.argmax(probs))
        max_confidence = float(probs[predicted_digit])
        
        return predicted_digit, max_confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0, 0.0

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    audio_buffer = []
    window_size = 8000  # 1 second at 8kHz
    
    try:
        while True:
            # Receive audio data (base64 encoded float32 array)
            data = await websocket.receive_text()
            audio_chunk = json.loads(data)
            
            # Convert to numpy array
            audio_array = np.array(audio_chunk, dtype=np.float32)
            audio_buffer.extend(audio_array)
            
            # Process when we have enough data
            if len(audio_buffer) >= window_size:
                # Take the latest 1-second window
                window = np.array(audio_buffer[-window_size:])
                
                # Predict digit
                digit, confidence = predict_digit(window)
                
                # Send result back to frontend
                result = {
                    "digit": digit,
                    "confidence": confidence,
                    "timestamp": time.time()
                }
                
                await websocket.send_text(json.dumps(result))
                
                # Keep buffer manageable (keep last 2 seconds)
                if len(audio_buffer) > window_size * 2:
                    audio_buffer = audio_buffer[-window_size:]
                    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/orb")
async def get_orb_page():
    """Serve the orb HTML page"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Digit Orb</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 25%, #2d1810 50%, #1a1a1a 75%, #0a0a0a 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: 'Arial', sans-serif;
            overflow: hidden;
        }
        
        .orb-container {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2rem;
        }
        
        .orb {
            width: 300px;
            height: 300px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #FFD700, #FFA500);
            box-shadow: 
                0 0 60px rgba(255, 215, 0, 0.8),
                inset 0 0 60px rgba(255, 255, 255, 0.2);
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            animation: breathe 3s ease-in-out infinite;
        }
        
        .orb:hover {
            transform: scale(1.05);
            box-shadow: 
                0 0 80px rgba(255, 215, 0, 1),
                inset 0 0 80px rgba(255, 255, 255, 0.3);
        }
        
        .orb.recording {
            animation: pulse 0.8s ease-in-out infinite;
        }
        
        .orb-text {
            font-size: 1.2rem;
            font-weight: bold;
            color: #000;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
        }
        
        @keyframes breathe {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        @keyframes pulse {
            0%, 100% { 
                transform: scale(1);
                box-shadow: 0 0 60px rgba(255, 215, 0, 0.8);
            }
            50% { 
                transform: scale(1.08);
                box-shadow: 0 0 100px rgba(255, 215, 0, 1);
            }
        }
        
        .sequence-display {
            color: #FFD700;
            font-size: 2rem;
            font-weight: bold;
            text-shadow: 0 0 20px rgba(255, 215, 0, 0.6);
            min-height: 3rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .status {
            color: #FFA500;
            font-size: 1rem;
            text-align: center;
            min-height: 2rem;
        }
    </style>
</head>
<body>
    <div class="orb-container">
        <div class="sequence-display" id="sequence">Ready to detect digits...</div>
        <div class="orb" id="orb" onclick="toggleRecording()">
            <div class="orb-text" id="orbText">TAP TO START</div>
        </div>
        <div class="status" id="status">Click the orb to begin real-time digit recognition</div>
    </div>

    <script>
        const DIGIT_COLORS = {
            0: "#FF0000", 1: "#FF7F00", 2: "#FFFF00", 3: "#7FFF00", 4: "#00FF00",
            5: "#00FF7F", 6: "#00FFFF", 7: "#007FFF", 8: "#0000FF", 9: "#7F00FF"
        };
        
        let isRecording = false;
        let websocket = null;
        let audioContext = null;
        let mediaStream = null;
        let processor = null;
        let sequence = [];
        
        const orb = document.getElementById('orb');
        const orbText = document.getElementById('orbText');
        const sequenceDisplay = document.getElementById('sequence');
        const status = document.getElementById('status');
        
        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }
        
        async function startRecording() {
            try {
                // Connect WebSocket
                websocket = new WebSocket('ws://localhost:8001/ws');
                
                websocket.onopen = () => {
                    status.textContent = 'Connected! Speak digits now...';
                };
                
                websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.confidence > 0.6) {  // Only high confidence predictions
                        updateOrb(data.digit, data.confidence);
                        addToSequence(data.digit);
                    }
                };
                
                websocket.onerror = (error) => {
                    status.textContent = 'WebSocket error: ' + error;
                };
                
                // Start audio capture
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                
                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(mediaStream);
                
                // Create processor for real-time audio processing
                processor = audioContext.createScriptProcessor(2048, 1, 1);
                
                processor.onaudioprocess = (event) => {
                    const audioData = event.inputBuffer.getChannelData(0);
                    
                    // Downsample to 8kHz (take every other sample)
                    const downsampled = [];
                    for (let i = 0; i < audioData.length; i += 2) {
                        downsampled.push(audioData[i]);
                    }
                    
                    // Send to WebSocket
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify(downsampled));
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                // Update UI
                isRecording = true;
                orb.classList.add('recording');
                orbText.textContent = 'LISTENING...';
                sequence = [];
                updateSequenceDisplay();
                
            } catch (error) {
                status.textContent = 'Error starting recording: ' + error.message;
            }
        }
        
        function stopRecording() {
            if (websocket) {
                websocket.close();
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
            }
            if (audioContext) {
                audioContext.close();
            }
            
            isRecording = false;
            orb.classList.remove('recording');
            orb.style.background = 'radial-gradient(circle at 30% 30%, #FFD700, #FFA500)';
            orbText.textContent = 'TAP TO START';
            status.textContent = 'Recording stopped. Sequence: ' + sequence.join(' → ');
        }
        
        function updateOrb(digit, confidence) {
            const color = DIGIT_COLORS[digit];
            orb.style.background = `radial-gradient(circle at 30% 30%, ${color}, ${color}AA)`;
            orbText.textContent = `DIGIT ${digit}`;
            
            // Flash effect for high confidence
            if (confidence > 0.8) {
                orb.style.transform = 'scale(1.15)';
                setTimeout(() => {
                    orb.style.transform = 'scale(1)';
                }, 200);
            }
        }
        
        function addToSequence(digit) {
            // Avoid duplicates (same digit detected multiple times)
            if (sequence.length === 0 || sequence[sequence.length - 1] !== digit) {
                sequence.push(digit);
                updateSequenceDisplay();
            }
        }
        
        function updateSequenceDisplay() {
            if (sequence.length === 0) {
                sequenceDisplay.textContent = 'Sequence will appear here...';
            } else {
                sequenceDisplay.textContent = sequence.join(' → ');
            }
        }
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
