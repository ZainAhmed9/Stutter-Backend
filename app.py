# app.py
import os
import librosa
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# ================= CONFIG =================
SR = 16000
TARGET_SEC = 3
TARGET_LEN = SR * TARGET_SEC

N_MELS = 128
N_MFCC = 20
N_FFT = 1024
HOP_LENGTH = 512
TOP_DB_TRIM = 25

CLASSES = ["Block", "Interjection", "NaturalPause", "Prolongation", "WordRep"]

# ================= LOAD TFLITE MODEL =================
# Make sure "cnn_model.tflite" is in the exact same folder as this app.py
try:
    interpreter = tf.lite.Interpreter(model_path="cnn_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")

# ================= FEATURE EXTRACTION LOGIC =================
def safe_fix_frames(feat, target_frames):
    if feat.shape[1] < target_frames:
        pad = target_frames - feat.shape[1]
        feat = np.pad(feat, ((0, 0), (0, pad)), mode="constant")
    elif feat.shape[1] > target_frames:
        feat = feat[:, :target_frames]
    return feat

def preprocess_audio(y):
    # 1) trim silence
    y_trim, _ = librosa.effects.trim(y, top_db=TOP_DB_TRIM)

    # fallback: if trim removed too much, use original
    if len(y_trim) > int(0.2 * SR):
        y = y_trim

    # 2) normalize peak safely
    peak = np.max(np.abs(y)) + 1e-8
    y = 0.95 * (y / peak)

    # 3) fix final length
    y = librosa.util.fix_length(y, size=TARGET_LEN)

    return y.astype(np.float32)

def extract_all_features(y):
    y = preprocess_audio(y)

    # Mel spectrogram reference frames
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    target_frames = mel_db.shape[1]

    # MFCC family
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Temporal / energy behavior
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_LENGTH).reshape(1, -1)

    # Silence mask from RMS
    rms_norm = rms / (np.max(rms) + 1e-8)
    silence_mask = (rms_norm < 0.08).astype(np.float32)

    # Align frame lengths
    feature_list = [
        mel_db,         # 128
        mfcc,           # 20
        mfcc_delta,     # 20
        mfcc_delta2,    # 20
        chroma,         # 12
        centroid,       # 1
        bandwidth,      # 1
        rolloff,        # 1
        contrast,       # 7
        flatness,       # 1
        rms,            # 1
        zcr,            # 1
        onset_env,      # 1
        silence_mask    # 1
    ]

    feature_list = [safe_fix_frames(f, target_frames) for f in feature_list]

    combined = np.vstack(feature_list).astype(np.float32)   # total 215 channels
    return combined

# ================= API ROUTE =================
@app.route('/')
def home():
    return jsonify({"status": "Stutter Detection API is Running!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file found in the request'})
    
    audio_file = request.files['audio']
    
    try:
        # Load audio directly from the uploaded file
        y, _ = librosa.load(audio_file, sr=SR, mono=True)
        
        # Extract the exact 215x94 features your model needs
        features = extract_all_features(y)
        
        # Reshape to (1, 215, 94, 1) to match your CNN input layer
        input_data = np.expand_dims(np.expand_dims(features, axis=0), axis=-1)
        
        # Run TFLite Prediction
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Get results
        max_idx = int(np.argmax(output_data))
        max_conf = float(output_data[max_idx])
        
        all_preds = {CLASSES[i]: f"{(output_data[i]*100):.2f}%" for i in range(len(CLASSES))}
        
        return jsonify({
            'success': True,
            'predicted_class': CLASSES[max_idx],
            'confidence': max_conf,
            'all_predictions': all_preds
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # host='0.0.0.0' allows external devices (like your phone/emulator) to access it
    app.run(host='0.0.0.0', port=5000, debug=True)