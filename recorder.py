# recorder.py
"""
Recorder module for server.py:
- Listens to server.vp frames
- When an 'Unknown' face is detected, records a short mp4 clip to DATA_DIR
- Sends the clip to Telegram via bot token + chat id

Place next to server.py and run alongside server (or import recorder from server.py).
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime
import requests
import traceback

import cv2
import numpy as np
from continuous_beep import start_beeping, stop_beeping


# Import helpers/constants from server.py (must be in same directory)
try:
    from server import (
        vp,
        embedding_from_image,
        load_known_faces_db,
        RECOG_THRESHOLD,
        FPS,
        UNKNOWN_CLIP_SECONDS,
        UNKNOWN_COOLDOWN,
        DATA_DIR,
    )
except Exception as e:
    raise RuntimeError(
        "recorder.py must be placed next to server.py and server.py must be importable. Import error: "
        + str(e)
    )

# Telegram config: set via env vars or edit below
TELEGRAM_BOT_TOKEN = "8304229700:AAF382L7DCr8DW-p-ZFL70p91DH_xQZqEpc"
TELEGRAM_CHAT_ID = "1515568039"


# Ensure DATA_DIR exists
DATA_DIR = Path(DATA_DIR) if not isinstance(DATA_DIR, Path) else DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_FPS = int(FPS) if FPS else 15
OUT_CODEC = "mp4v"  # FourCC for .mp4
OUT_EXT = ".mp4"
RECORD_SECONDS = float(UNKNOWN_CLIP_SECONDS) if UNKNOWN_CLIP_SECONDS else 8.0
COOLDOWN = float(UNKNOWN_COOLDOWN) if UNKNOWN_COOLDOWN else 8.0

class Recorder:
    def __init__(self):
        self.known_cache = []
        self.last_record_ts = 0.0
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _reload_known(self):
        try:
            self.known_cache = load_known_faces_db()
        except Exception:
            self.known_cache = []

    def _is_known(self, face_bgr):
        """Return (is_known, best_sim, name_or_empty). Uses embedding_from_image from server.py."""
        try:
            emb = embedding_from_image(face_bgr)
        except Exception:
            return False, 0.0, ""
        if not self.known_cache:
            return False, 0.0, ""
        known_embs = np.stack([k["emb"] for k in self.known_cache], axis=0)
        sims = known_embs.dot(emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= RECOG_THRESHOLD:
            return True, best_sim, self.known_cache[best_idx]["name"]
        return False, best_sim, ""

    def _get_frame_image(self, timeout=2.0):
        """Obtain a BGR image from vp.get_frame_jpeg() within timeout seconds (or None)."""
        start = time.time()
        while time.time() - start < timeout:
            jpg = vp.get_frame_jpeg()
            if jpg is None:
                time.sleep(0.02)
                continue
            arr = np.frombuffer(jpg, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                time.sleep(0.01)
                continue
            return img
        return None

    def _record_clip(self, out_path: Path, duration: float) -> bool:
        # Try to grab first frame for size
        first = self._get_frame_image(timeout=3.0)
        if first is None:
            print("[recorder] couldn't get first frame for recording")
            return False
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*OUT_CODEC)
        writer = cv2.VideoWriter(str(out_path), fourcc, OUT_FPS, (w, h))
        if not writer.isOpened():
            print("[recorder] VideoWriter open failed for", out_path)
            return False
        frame_target = max(1, int(round(duration * OUT_FPS)))
        written = 0
        start = time.time()
        while written < frame_target and time.time() - start < duration + 3.0:
            frm = self._get_frame_image(timeout=1.0)
            if frm is None:
                time.sleep(0.02)
                continue
            writer.write(frm)
            written += 1
            # pace slightly
            time.sleep(max(0, (1.0 / OUT_FPS) - 0.001))
        writer.release()
        return written > 0

    def _send_telegram(self, file_path: Path, caption: str) -> bool:
        if TELEGRAM_BOT_TOKEN.startswith("PUT_") or TELEGRAM_CHAT_ID.startswith("PUT_"):
            print("[recorder] Telegram credentials not set — skipping send. File:", file_path)
            return False
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
        try:
            with open(file_path, "rb") as fh:
                files = {"video": (file_path.name, fh)}
                data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
                resp = requests.post(url, data=data, files=files, timeout=60)
            if resp.status_code == 200:
                return True
            else:
                print("[recorder] Telegram API error:", resp.status_code, resp.text)
                return False
        except Exception as e:
            print("[recorder] Exception sending to Telegram:", e)
            return False

    def _loop(self):
        print("[recorder] started")
        while self.running:
            try:
                self._reload_known()
                jpg = vp.get_frame_jpeg()
                if jpg is None:
                    time.sleep(0.05)
                    continue
                arr = np.frombuffer(jpg, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    time.sleep(0.03)
                    continue

                # Lightweight face detection using MediaPipe (same style as server.VideoProcessor)
                try:
                    import mediapipe as mp
                    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.45)
                    res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    mp_face.close()
                except Exception as e:
                    # If mediapipe not available, skip this iteration
                    print("[recorder] mediapipe error:", e)
                    time.sleep(0.5)
                    continue

                unknown_seen = False
                faces_detected = 0
                if res and res.detections:
                    faces_detected = len(res.detections)
                    h, w = frame.shape[:2]
                    for det in res.detections:
                        bbox = det.location_data.relative_bounding_box
                        x1 = max(0, int(bbox.xmin * w))
                        y1 = max(0, int(bbox.ymin * h))
                        x2 = min(w - 1, int((bbox.xmin + bbox.width) * w))
                        y2 = min(h - 1, int((bbox.ymin + bbox.height) * h))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        face_crop = frame[y1:y2, x1:x2]
                        is_known, sim, name = self._is_known(face_crop)
                        if not is_known:
                            unknown_seen = True
                            break
                # -------------------------
                # Continuous beep control
                # -------------------------
                if unknown_seen:
                    start_beeping()   # intruder present → start or continue beeping
                else:
                    stop_beeping()    # intruder gone → stop beeping


                # Trigger recording if unknown found and cooldown passed
                if unknown_seen and (time.time() - self.last_record_ts) >= COOLDOWN:
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    filename = f"intruder_{ts}{OUT_EXT}"
                    out_path = DATA_DIR / filename
                    print(f"[recorder] Unknown face detected ({faces_detected} faces). Recording {out_path} ...")
                    ok = self._record_clip(out_path, RECORD_SECONDS)
                    if ok:
                        self.last_record_ts = time.time()
                        caption = (
                            f"Intruder alert — {datetime.utcnow().isoformat()} UTC\n"
                            f"Detected faces: {faces_detected}\n"
                            f"Approx duration: {RECORD_SECONDS}s\n"
                            f"File: {filename}"
                        )
                        sent = self._send_telegram(out_path, caption)
                        if sent:
                            print("[recorder] Sent to Telegram:", filename)
                        else:
                            print("[recorder] Saved locally (send failed or disabled):", out_path)
                    else:
                        print("[recorder] Recording failed for", out_path)

                time.sleep(0.12)
            except Exception:
                print("[recorder] Exception in loop:")
                traceback.print_exc()
                time.sleep(1.0)

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass

# instantiate on import
recorder = Recorder()

if __name__ == "__main__":
    print("recorder.py running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        recorder.stop()
        print("recorder stopped.")
