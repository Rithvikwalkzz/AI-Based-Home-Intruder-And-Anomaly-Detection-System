# server.py
# Robust Home Intruder Detection System backend
# - OpenCV captures server webcam
# - MediaPipe for face detection
# - keras-facenet (FaceNet) for embeddings
# - SQLite stores normalized embeddings
# - MJPEG stream, register (multi-frame average), recognition, unknown recording

import os
import time
import threading
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# try to import FaceNet
try:
    from keras_facenet import FaceNet
    FACENET_AVAILABLE = True
except Exception as e:
    FaceNet = None
    FACENET_AVAILABLE = False
    print("⚠️ keras-facenet not available:", e)

# -------------------------
# Config
# -------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATABASE_PATH = DATA_DIR / "faces.db"

RECOG_THRESHOLD = 0.75      # cosine similarity threshold (0..1). increase to be stricter.
REGISTRATION_FRAMES = 5     # number of frames to average when registering a new face
FPS = 15
UNKNOWN_CLIP_SECONDS = 8
UNKNOWN_COOLDOWN = 8        # seconds between recordings start

CAMERA_FALLBACKS = [1, 0]

# -------------------------
# App init
# -------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount data folder to allow direct streaming of saved mp4s
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# -------------------------
# DB
# -------------------------
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

init_db()

# -------------------------
# FaceNet model (single instance)
# -------------------------
facenet = None
if FACENET_AVAILABLE:
    try:
        facenet = FaceNet()
    except Exception as e:
        print("⚠️ Failed to initialize FaceNet:", e)
        facenet = None
else:
    print("⚠️ keras-facenet unavailable — registration / recognition will fail until installed.")

# -------------------------
# Utilities: embeddings & DB helpers
# -------------------------
def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-10)

def embedding_from_image(face_bgr: np.ndarray) -> np.ndarray:
    """
    Given a face crop in BGR (OpenCV), produce a normalized embedding using FaceNet.
    Raises RuntimeError if Facenet not loaded.
    """
    if facenet is None:
        raise RuntimeError("FaceNet not available")
    # convert to RGB and resize to 160x160 expected by FaceNet wrapper
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (160, 160))
    emb = facenet.embeddings([face_rgb])[0]
    emb = np.array(emb, dtype=np.float32)
    return normalize_vector(emb)

def add_known_face_db(name: str, emb: np.ndarray):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO known_faces (name, embedding, created_at) VALUES (?, ?, ?)",
              (name, sqlite3.Binary(emb.tobytes()), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def load_known_faces_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, embedding FROM known_faces")
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        _id, name, emb_blob = r
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        # ensure normalized (defensive)
        emb = normalize_vector(emb)
        out.append({"id": _id, "name": name, "emb": emb})
    return out

def clear_known_faces_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM known_faces")
    conn.commit()
    conn.close()

# cosine similarity (dot product for normalized vectors)
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# -------------------------
# Video Processor (captures, detects, recognizes, records unknowns)
# -------------------------
def open_camera_with_fallback(src_list):
    """
    Try camera indices in order and return an opened cv2.VideoCapture.
    Raises RuntimeError if none are available.
    """
    for src in src_list:
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            print(f"✅ Camera opened successfully at index {src}")
            return cap, src
        cap.release()

    raise RuntimeError("❌ No available camera found (USB or PC camera).")

class VideoProcessor:
    def __init__(self, src_list):
        try:
            self.cap, self.active_src = open_camera_with_fallback(src_list)
        except RuntimeError as e:
            print(e)
            self.cap = None
            self.running = False
            return

        # try to set resolution for more stable crops (optional)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.frame_count = 0
            self.DETECT_EVERY = 3
        except Exception:
            pass

        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        # mediapipe face detection
        self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.last_unknown_time = 0
        self.thread.start()
        # keep a cached known list to reduce DB hits in tight loops, but reload often
        self.known_cache = []
        self._reload_known_cache()

    def _reload_known_cache(self):
        try:
            self.known_cache = load_known_faces_db()
        except Exception:
            self.known_cache = []

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        try:
            self.mp_face.close()
        except Exception:
            pass

    def _run(self):
        # main capture loop
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()

            


            # make a copy we can annotate
            annotated = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = self.mp_face.process(rgb)
            except Exception:
                results = None

            names_found = []

            if results and results.detections:
                # reload known cache occasionally (cheap)
                self._reload_known_cache()
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x1 = max(0, int(bbox.xmin * w))
                    y1 = max(0, int(bbox.ymin * h))
                    x2 = min(w - 1, int((bbox.xmin + bbox.width) * w))
                    y2 = min(h - 1, int((bbox.ymin + bbox.height) * h))
                    # avoid degenerate boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_crop = frame[y1:y2, x1:x2]
                    label = "Unknown"
                    try:
                        emb = embedding_from_image(face_crop)  # normalized
                        # compare against known cache
                        if self.known_cache:
                            # vectorized dot product for speed
                            known_embs = np.stack([k["emb"] for k in self.known_cache], axis=0)  # (N, D)
                            sims = known_embs.dot(emb)  # since normalized -> cosine similarity
                            best_idx = int(np.argmax(sims))
                            best_sim = float(sims[best_idx])
                            if best_sim >= RECOG_THRESHOLD:
                                label = f'{self.known_cache[best_idx]["name"]} ({best_sim:.2f})'
                            else:
                                label = "Unknown"
                        else:
                            label = "Unknown"
                    except Exception:
                        label = "Unknown"

                    color = (0, 220, 180) if label != "Unknown" else (0, 180, 200)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1, max(16, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                    names_found.append(label)

        
            with self.lock:
                self.frame = annotated

            time.sleep(1.0 / FPS)

    def get_frame_jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            ret, buf = cv2.imencode(".jpg", self.frame)
            if not ret:
                return None
            return buf.tobytes()



# initialize processor
vp = VideoProcessor(src_list=CAMERA_FALLBACKS)

# -------------------------
# MJPEG stream generator (proper headers)
# -------------------------
def mjpeg_generator():
    while True:
        frame = vp.get_frame_jpeg()
        if frame is None:
            time.sleep(0.03)
            continue
        # multipart frame
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.002)

# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/register")
async def register(name: str = Form(...), file: UploadFile = File(None)):
    """
    Register a new face:
    - If `file` provided (image upload), extract single face and register.
    - Otherwise, capture REGISTRATION_FRAMES frames from live camera, detect face in each,
      compute embeddings and average them to create a robust embedding.
    """
    if facenet is None:
        raise HTTPException(status_code=500, detail="FaceNet model not available on server.")

    embeddings = []
    # helper to process an image (BGR) and return embedding or None
    def try_embedding_from_img(img_bgr):
        mp_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
        res = mp_detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mp_detector.close()
        if not res or not res.detections:
            return None
        det = res.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w, _ = img_bgr.shape
        x1 = max(0, int(bbox.xmin * w)); y1 = max(0, int(bbox.ymin * h))
        x2 = min(w - 1, int((bbox.xmin + bbox.width) * w)); y2 = min(h - 1, int((bbox.ymin + bbox.height) * h))
        if x2 <= x1 or y2 <= y1:
            return None
        face = img_bgr[y1:y2, x1:x2]
        try:
            emb = embedding_from_image(face)
            return emb
        except Exception:
            return None

    if file is not None:
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        emb = try_embedding_from_img(img)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected in uploaded image")
        embeddings.append(emb)
    else:
        # capture REGISTRATION_FRAMES from live camera with small delay, accumulate embeddings
        count = 0
        attempts = 0
        max_attempts = REGISTRATION_FRAMES * 4
        while count < REGISTRATION_FRAMES and attempts < max_attempts:
            attempts += 1
            frame_jpg = vp.get_frame_jpeg()
            if frame_jpg is None:
                time.sleep(0.05)
                continue
            arr = np.frombuffer(frame_jpg, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            emb = try_embedding_from_img(img)
            if emb is not None:
                embeddings.append(emb)
                count += 1
            time.sleep(0.25)  # short delay to get slightly different angles/expressions

        if len(embeddings) == 0:
            raise HTTPException(status_code=400, detail="No face detected during registration - try again with better lighting")

    # average embeddings (if multiple) and re-normalize
    avg = np.mean(np.stack(embeddings, axis=0), axis=0)
    avg = normalize_vector(avg.astype(np.float32))

    add_known_face_db(name, avg)
    return JSONResponse({"status": "ok", "name": name, "frames_used": len(embeddings)})

@app.get("/known_faces")
async def get_known_faces():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, created_at FROM known_faces ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    faces = [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]
    return JSONResponse(faces)

@app.delete("/known_faces")
async def delete_all_known_faces():
    clear_known_faces_db()
    return JSONResponse({"status": "ok", "message": "All known faces cleared."})

@app.delete("/known_faces/{face_id}")
async def delete_known_face(face_id: int):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM known_faces WHERE id = ?", (face_id,))
    conn.commit()
    conn.close()
    return JSONResponse({"status": "ok", "deleted_id": face_id})


# -------------------------
# Serve the files via /data already mounted above
# -------------------------
# (no separate download route required; /data/<filename> streams)

# -------------------------
# Lifespan / shutdown
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    try:
        vp.stop()
    except Exception:
        pass

app.router.lifespan_context = lifespan

# -------------------------
# Auto-open browser and run
# -------------------------
def get_free_port(default=8000):
    import socket
    port = default
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                s.close()
                return port
            except OSError:
                port += 1

import recorder 

if __name__ == "__main__":
    import uvicorn, webbrowser, requests
    port = get_free_port(8000)
    url = f"http://127.0.0.1:{port}"

    def open_browser_when_ready():
        for _ in range(20):
            try:
                requests.get(url)
                webbrowser.open(url)
                return
            except Exception:
                time.sleep(0.5)
        print("⚠️ Auto-open failed - open", url)

    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    print(f"\n🚀 Home Intruder Detection System running at {url}\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
