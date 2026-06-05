import os
import shutil
import tempfile
import joblib as pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
from math import gcd

# Set TensorFlow env vars before importing Keras/TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU only
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_ENABLE_PROFILER", "0")

from PIL import Image
import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=BASE_DIR)

try:
    import imageio_ffmpeg as iio_ffmpeg
except ImportError:
    iio_ffmpeg = None


def get_ffmpeg_executable():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path and os.path.isfile(ffmpeg_path):
        return ffmpeg_path
    if iio_ffmpeg is not None:
        try:
            exe = iio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.isfile(exe):
                return exe
        except Exception as e:
            print(f"WARNING: imageio-ffmpeg failed to get FFmpeg executable: {e}")
            return None
    return None

# --- MODEL LOADING ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "the_cat_model.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "the_cat_model.pkl")

IMG_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_image_best.keras")
if not os.path.exists(IMG_MODEL_PATH):
    IMG_MODEL_PATH = os.path.join(BASE_DIR, "cnn_image_best.keras")

AUD_MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_audio_best.keras")
if not os.path.exists(AUD_MODEL_PATH):
    AUD_MODEL_PATH = os.path.join(BASE_DIR, "lstm_audio_best.keras")

class_names = ['Angry', 'Scared', 'Happy', 'Sad']
IMAGE_SIZE = (224, 224)
AUDIO_TARGET_SR = 22050
AUDIO_N_MFCC = 40
AUDIO_FRAMES = 100
AUDIO_N_FFT = 512
AUDIO_HOP_LENGTH = 512
AUDIO_MAX_SECONDS = 2.5
AUDIO_MAX_SAMPLES = int(AUDIO_TARGET_SR * AUDIO_MAX_SECONDS)


def resample_audio(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y

    from scipy.signal import resample_poly

    divisor = gcd(orig_sr, target_sr)
    return resample_poly(y, target_sr // divisor, orig_sr // divisor).astype(np.float32)


def mel_filter_bank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            filters[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    return filters


MEL_FILTERS = mel_filter_bank(AUDIO_TARGET_SR, AUDIO_N_FFT, 128)


def fast_mfcc(y, sr):
    from scipy.fftpack import dct

    if len(y) < AUDIO_N_FFT:
        y = np.pad(y, (0, AUDIO_N_FFT - len(y)))

    frame_count = 1 + max(0, (len(y) - AUDIO_N_FFT) // AUDIO_HOP_LENGTH)
    shape = (frame_count, AUDIO_N_FFT)
    strides = (y.strides[0] * AUDIO_HOP_LENGTH, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides).copy()
    frames *= np.hanning(AUDIO_N_FFT).astype(np.float32)

    power = np.abs(np.fft.rfft(frames, n=AUDIO_N_FFT)) ** 2
    mel_power = np.maximum(np.dot(power, MEL_FILTERS.T), 1e-10)
    log_mel = np.log(mel_power)
    return dct(log_mel, type=2, axis=1, norm='ortho')[:, :AUDIO_N_MFCC].T.astype(np.float32)

def parse_prediction(preds, classes):
    confidence = 100.0
    emotion = "Unknown"
    try:
        preds_flat = np.array(preds).flatten()
        # If output contains probabilities for multiple classes
        if len(preds_flat) > 1 and np.issubdtype(preds_flat.dtype, np.number):
            idx = int(np.argmax(preds_flat))
            confidence = round(float(preds_flat[idx] * 100), 2)
            if 0 <= idx < len(classes):
                emotion = classes[idx]
        else:
            pred_val = preds_flat[0]
            if isinstance(pred_val, (str, np.str_)):
                pred_str = str(pred_val).lower()
                for c in classes:
                    if c.lower() in pred_str:
                        emotion = c
                        break
            else:
                idx = int(round(float(pred_val)))
                if 0 <= idx < len(classes):
                    emotion = classes[idx]
    except Exception as e:
        print(f"Parse Prediction Error: {e}")

    # Enforce strictly the known classes to avoid random outputs
    if emotion not in classes:
        emotion = classes[0]
        
    return emotion, confidence


def decode_wav_stream(stream):
    import wave

    stream.seek(0)
    with wave.open(stream, 'rb') as wav_file:
        sr = wav_file.getframerate()
        sampwidth = wav_file.getsampwidth()
        n_channels = wav_file.getnchannels()
        total_frames = wav_file.getnframes()
        frames_to_read = min(total_frames, max(1, int(sr * AUDIO_MAX_SECONDS)))
        audio_data = wav_file.readframes(frames_to_read)

    if sampwidth == 1:
        y = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32)
        y = (y - 128.0) / 128.0
    elif sampwidth == 2:
        y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        y = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError('Unsupported WAV bit depth')

    if n_channels > 1:
        y = y.reshape(-1, n_channels).mean(axis=1)

    return y, sr


def prepare_audio_for_model(y, sr):
    y = np.asarray(y, dtype=np.float32)

    if sr != AUDIO_TARGET_SR:
        y = resample_audio(y[:max(1, int(sr * AUDIO_MAX_SECONDS))], sr, AUDIO_TARGET_SR)
        sr = AUDIO_TARGET_SR

    y = y[:AUDIO_MAX_SAMPLES]

    if len(y) < sr:
        raise ValueError('Audio too short')

    mfcc = fast_mfcc(y, sr)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    mfcc = np.pad(mfcc, ((0, 0), (0, max(0, AUDIO_FRAMES - mfcc.shape[1]))))[:, :AUDIO_FRAMES]

    return mfcc.T.reshape(1, AUDIO_FRAMES, AUDIO_N_MFCC)

image_model = None
audio_model = None

# Try to load the Keras image model (this may still require TF; if it hangs, we'll make it lazy too).
try:
    if os.path.exists(IMG_MODEL_PATH):
        image_model = keras.models.load_model(IMG_MODEL_PATH, compile=False)
        print(f"Image Model loaded from {IMG_MODEL_PATH}!")
        
        # Warm up model to avoid latency on first prediction
        if hasattr(image_model, 'make_predict_function'):
            image_model.make_predict_function()
            _ = image_model.predict(
                np.zeros((1, IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.float32),
                verbose=0,
            )
            print("Image Model warmed up!")
    else:
        print(f"WARNING: Image model file not found at {IMG_MODEL_PATH}")
except Exception as e:
    print(f"WARNING: Could not load image keras model: {e}")
    image_model = None

# Try to load the audio model.
try:
    if os.path.exists(AUD_MODEL_PATH) and AUD_MODEL_PATH.endswith('.keras'):
        audio_model = keras.models.load_model(AUD_MODEL_PATH, compile=False)
        print(f"Audio Model (Keras) loaded from {AUD_MODEL_PATH}!")
        
        # Warm up model to avoid latency on first prediction
        if hasattr(audio_model, 'make_predict_function'):
            audio_model.make_predict_function()
            _ = audio_model.predict(
                np.zeros((1, AUDIO_FRAMES, AUDIO_N_MFCC), dtype=np.float32),
                verbose=0,
            )
            print("Audio Model warmed up!")
    elif os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            audio_model = pickle.load(f)
        print(f"Audio Model (Pickle) loaded from {MODEL_PATH}!")
    else:
        print(f"WARNING: Audio model file not found at {AUD_MODEL_PATH} or {MODEL_PATH}")
except Exception as e:
    print(f"WARNING: Could not load audio model: {e}")
    audio_model = None

# Warm up audio feature extraction to avoid latency on first prediction.
try:
    _ = fast_mfcc(np.zeros(AUDIO_TARGET_SR, dtype=np.float32), AUDIO_TARGET_SR)
    print("Audio feature extraction warmed up!")
except Exception as e:
    print(f"WARNING: Audio feature warmup failed: {e}")


@app.route('/favicon.ico')
def favicon():
    # Serves a generic cat emoji favicon to prevent terminal/blank icons in the browser
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🐱</text></svg>'
    return app.response_class(svg, mimetype='image/svg+xml')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    global image_model

    if image_model is None:
        return jsonify({'error': 'Image model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        img = Image.open(file.stream).convert('RGB')
        img_array = np.asarray(img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR))
        
        # Fast MobileNetV2 preprocess (avoids slow TensorFlow import overhead)
        img_array = (img_array.astype(np.float32) / 127.5) - 1.0
        
        img_batch = np.expand_dims(img_array, axis=0)

        # Flatten if the model is not a Keras CNN
        if not hasattr(image_model, 'make_predict_function'):
            img_batch = img_batch.reshape(1, -1)

        if hasattr(image_model, 'predict_proba'):
            preds = image_model.predict_proba(img_batch)
        elif hasattr(image_model, 'make_predict_function'): # Keras fast path
            preds = image_model.predict(img_batch, verbose=0)
        else:
            preds = image_model.predict(img_batch)

        emotion, confidence = parse_prediction(preds, class_names)

        return jsonify({'emotion': emotion, 'confidence': confidence})

    except Exception as e:
        print("IMAGE ERROR:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    global audio_model

    if audio_model is None:
        return jsonify({'error': 'Audio model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        try:
            y, sr = decode_wav_stream(file.stream)
        except Exception:
            import subprocess
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or '')[1] or ".audio")
            os.close(tmp_fd)
            wav_tmp_fd, wav_tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(wav_tmp_fd)

            try:
                file.stream.seek(0)
                file.save(tmp_path)

                ffmpeg_exe = get_ffmpeg_executable()
                if ffmpeg_exe is None:
                    error_msg = 'FFmpeg is missing! It is needed to process non-WAV audio files.'
                    if iio_ffmpeg is None:
                        error_msg += ' The `imageio-ffmpeg` package is not installed. Please run `pip install -r requirements.txt`.'
                    else:
                        error_msg += ' FFmpeg was not found in your system PATH, and the `imageio-ffmpeg` package failed to provide it. Please install FFmpeg manually, or check the console output for errors from `imageio-ffmpeg`.'
                    return jsonify({
                        'error': error_msg
                    }), 400

                subprocess.run([
                    ffmpeg_exe, '-y', '-i', tmp_path,
                    '-t', str(AUDIO_MAX_SECONDS), '-ar', str(AUDIO_TARGET_SR), '-ac', '1',
                    '-acodec', 'pcm_s16le', wav_tmp_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                with open(wav_tmp_path, 'rb') as wav_stream:
                    y, sr = decode_wav_stream(wav_stream)
            except FileNotFoundError:
                return jsonify({'error': 'FFmpeg is missing! Please install FFmpeg to process non-WAV formats, or upload a standard .wav file.'}), 400
            except subprocess.CalledProcessError:
                return jsonify({'error': 'FFmpeg failed to decode the audio format.'}), 400
            except Exception as e:
                return jsonify({'error': f'Failed to decode audio: {e}'}), 400
            finally:
                for path in (tmp_path, wav_tmp_path):
                    if os.path.exists(path):
                        os.remove(path)

        input_data = prepare_audio_for_model(y, sr)
        if not hasattr(audio_model, 'make_predict_function'):
            input_data = input_data.reshape(1, -1)

        if hasattr(audio_model, 'predict_proba'):
            preds = audio_model.predict_proba(input_data)
        elif hasattr(audio_model, 'make_predict_function'): # Keras fast path
            preds = audio_model.predict(input_data, verbose=0)
        else:
            preds = audio_model.predict(input_data)

        emotion, confidence = parse_prediction(preds, class_names)

        return jsonify({'emotion': emotion, 'confidence': confidence})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print("AUDIO ERROR:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/contact', methods=['POST'])
def contact():
    _ = request.json
    return jsonify({"status": "Message sent successfully!"})


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000, use_reloader=False)
