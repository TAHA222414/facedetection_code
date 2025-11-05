import os
import cv2
import time
import warnings
import pickle
import sounddevice as sd
import wavio
import torch
import whisper
import numpy as np
from collections import deque
from deepface import DeepFace
import mediapipe as mp
import firebase_admin
from firebase_admin import credentials, firestore
import threading
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ================== CONFIG ==================
SAMPLERATE = 16000
WHISPER_MODEL_SIZE = "tiny"
FACE_MODEL_NAME = "Facenet"
SAFE_THRESHOLD = 0.28
MIN_VOTES = 3
TEMPORAL_WINDOW = 5
FRAME_W, FRAME_H = 320, 240
AUTO_START_RECORD_ON_RECOGNIZE = False
# ============================================

recent_predictions = deque(maxlen=TEMPORAL_WINDOW)

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
VOICE_FILE = os.path.abspath("voice_note.wav")
EMB_FILE = "embeddings.pkl"
DATASET_DIR = "dataset"

warnings.filterwarnings("ignore")

# ---------- Firebase init ----------
cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ====== Load Whisper ======
print(f"[⏳] Loading Whisper model ({WHISPER_MODEL_SIZE})...")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
print("[✅] Whisper model loaded.")

# ====== Load DistilBERT priority classifier ======
MODEL_DIR = r"D:\fyp\distilbert\distilbert_3label_priority_20250915_224504"
print("[⏳] Loading priority classifier...")
try:
    tok = DistilBertTokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
    priority_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    priority_model.eval()
    print("[✅] Priority classifier loaded.")
except Exception as e:
    print(f"[⚠] Failed to load priority classifier: {e}")
    tok = None
    priority_model = None

# ====== Cosine Distance ======
def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return 1.0 - float(np.dot(a, b) / denom)

# ====== Priority prediction ======
def predict_priority(sentence):
    if tok is None or priority_model is None:
        return "Unknown"
    try:
        print(f"Text input for priority prediction: {sentence}")  # Debug print
        inputs = tok(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = priority_model(**inputs)
            print(f"Model output logits: {outputs.logits}")  # Debug print
            prediction = torch.argmax(outputs.logits, dim=1).item()
        label_map = {2: "High", 1: "Medium", 0: "Low"}
        return label_map.get(prediction, "Unknown")
    except Exception as e:
        print(f"[⚠] Priority prediction failed: {e}")
        return "Unknown"

# ====== Firestore Upload ======
def upload_notice(person, text):
    if not text:
        print("[⚠] Empty notice, skipping upload.")
        return False
    priority_label = predict_priority(text)
    notice_data = {
        "person": person or "unknown",
        "notice": text,
        "text": text,
        "priority": priority_label,
        "status": "Displayed",
        "timestamp": firestore.SERVER_TIMESTAMP,
    }
    try:
        db.collection("notices_fr").add(notice_data)
        print(f"[✅] Notice uploaded to Firestore (notices_fr). Priority: {priority_label}")
        return True
    except Exception as e:
        print(f"[⚠] Firestore upload failed: {e}")
        return False

# ====== MediaPipe Setup ======
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.75)

# ====== Manual Start/Stop Recording ======
recording = False
audio_frames = []

def _callback(indata, frames, ttime, status):
    if recording:
        audio_frames.append(indata.copy())

def start_recording():
    global recording, audio_frames
    audio_frames = []
    recording = True
    with sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=_callback):
        print("[🎙] Recording... Press 2 to stop.")
        while recording:
            sd.sleep(100)

def stop_recording():
    global recording
    recording = False
    if audio_frames:
        audio_data = np.concatenate(audio_frames, axis=0)
        if audio_data.dtype != np.int16:
            audio_data_to_write = (audio_data * 32767).astype(np.int16)
        else:
            audio_data_to_write = audio_data
        wavio.write(VOICE_FILE, audio_data_to_write, SAMPLERATE, sampwidth=2)
        print(f"[✅] Saved recording: {VOICE_FILE}")

# ====== Transcribe and Confirm ======
def transcribe_voice(person):
    if not os.path.exists(VOICE_FILE):
        print(f"[⚠] {VOICE_FILE} not found")
        return False
    try:
        result = whisper_model.transcribe(VOICE_FILE, language='en')
        result_text = result.get("text", "").strip()
        print(f"[📝] Whisper Transcription: {result_text}")  # Debug print
        if not result_text:
            print("[⚠] Could not detect any speech. Please try again.")
            return False

        print(f"\n[📝] You said: \"{result_text}\"\n")
        print("Press: ")
        print("1: CONFIRM YOUR NOTICE.")
        print("2: RE-RECORD NOTICE.")
        print("3: CANCEL NOTICE.")
        while True:
            choice = input("Choice: ").strip()
            if choice == '1':
                upload_notice(person, result_text)
                print("Press: ")
                print("1: GIVE ANOTHER NOTICE.")
                print("2: To STOP")
                while True:
                    nxt = input("Choice: ").strip()
                    if nxt == '1':
                        return True  # Give another notice
                    elif nxt == '2':
                        print("Exiting... Goodbye!")
                        exit(0)
                    else:
                        print("Invalid input. Press 1 or 2.")
            elif choice == '2':
                print("Press: ")
                print("1: START RECORDING.")
                print("2: STOP RECORDING.")
                while True:
                    ch = input("Choice: ").strip()
                    if ch == '1':
                        t = threading.Thread(target=start_recording)
                        t.start()
                    elif ch == '2':
                        stop_recording()
                        return transcribe_voice(person)
                    else:
                        print("Invalid input")
            elif choice == '3':
                print("Cancelled.")
                return False
            else:
                print("Invalid input. Press 1, 2, or 3.")
    except Exception as e:
        print(f"[⚠] Error in transcription/prompt flow: {e}")
        return False

# ====== Embeddings ======
def save_embeddings(data):
    with open(EMB_FILE, 'wb') as f:
        pickle.dump(data, f)

def load_embeddings():
    if os.path.exists(EMB_FILE):
        with open(EMB_FILE, 'rb') as f:
            return pickle.load(f)
    return None

# ====== Camera and Face Model ======
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

face_model = DeepFace.build_model(FACE_MODEL_NAME)

# ====== Dataset ======
if os.path.exists(EMB_FILE):
    print("Press: ")
    print("1: UPDATE DATASET.")
    print("2: LOAD GIVEN DATASET.")
    c = input("Choice: ").strip()
else:
    c = '1'

if c == '1':
    reps = []
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.jpg','.jpeg','.png')):
                path = os.path.join(root,file)
                try:
                    r = DeepFace.represent(img_path=path, model_name=FACE_MODEL_NAME, enforce_detection=True)[0]
                    r['identity'] = path
                    reps.append(r)
                except Exception as e:
                    print(f"Skipped {path}: {e}")
    save_embeddings(reps)
    representations = reps
else:
    representations = load_embeddings()

if not representations:
    print("No embeddings available")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

# ====== Main Loop ======
print("\n[✅] Ready. Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (w//2-100, h//2-120), (w//2+100, h//2+120), (0, 255, 0), 2)
    cv2.putText(frame, "Align Face in Box", (w//2-110, h//2-130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    if not results.detections:
        cv2.imshow("Smart Notice Board", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
        continue

    try:
        emb = DeepFace.represent(img_path=frame, model_name=FACE_MODEL_NAME, enforce_detection=True)[0]['embedding']
        votes = {}
        for rep in representations:
            d = cosine_distance(emb, rep['embedding'])
            if d < SAFE_THRESHOLD:
                person = os.path.basename(os.path.dirname(rep['identity']))
                votes[person] = votes.get(person, 0) + 1

        identity = 'unknown'
        if votes:
            best = max(votes, key=votes.get)
            if votes[best] >= MIN_VOTES:
                identity = best

        recent_predictions.append(identity)
        if len(recent_predictions) == recent_predictions.maxlen:
            top = max(set(recent_predictions), key=recent_predictions.count)
            if recent_predictions.count(top) >= 3 and top != 'unknown':
                print(f"\n[✅] Face recognized as {top}.")
                give_another = True
                while give_another:
                    if AUTO_START_RECORD_ON_RECOGNIZE:
                        t = threading.Thread(target=start_recording)
                        t.start()
                    else:
                        print("Press: ")
                        print("1: START RECORDING.")
                        print("2: STOP RECORDING.")
                    while True:
                        ch = input("Choice: ").strip()
                        if ch == '1':
                            t = threading.Thread(target=start_recording)
                            t.start()
                        elif ch == '2':
                            stop_recording()
                            give_another = transcribe_voice(top)
                            break
                        else:
                            print("Invalid input")
    except Exception as e:
        print(f"[⚠] {e}")

    cv2.imshow("Smart Notice Board", frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
