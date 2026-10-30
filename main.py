import asyncio
from collections import deque
from datetime import datetime
import json
import os
import time
from typing import Set
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import supervision as sv
from ultralytics import YOLO
import uvicorn

from classifier import classify_vehicle_crop, encode_crop_to_base64
from config import (
    CAMERA_SOURCE,
    CONFIDENCE_THRESHOLD,
    COUNTING_LINE_Y_RATIO,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIN_CROP_SIZE,
    STATS_FILE,
    TARGET_FPS,
    TRACKED_CLASSES,
    YOLO_MODEL,
)

app = FastAPI(title="Colombia 24/7 TikTok Street AI")
app.mount("/static", StaticFiles(directory="static"), name="static")

# COCO class ID to name
CLASS_NAMES = {2: "car", 3: "moto", 5: "bus", 7: "truck"}


class StreamState:
    def __init__(self):
        self.date_str = datetime.now().strftime("%Y-%m-%d")
        self.total_vehicles = 0
        self.counts_by_type = {"car": 0, "moto": 0, "bus": 0, "truck": 0}
        self.taxis_count = 0
        self.prado_count = 0
        self.twingo_count = 0
        self.brand_counts = {
            "Renault": 0,
            "Chevrolet": 0,
            "Mazda": 0,
            "Kia": 0,
            "Toyota": 0,
            "Suzuki": 0,
            "Nissan": 0,
            "Bajaj": 0,
            "Yamaha": 0,
        }
        self.recent_spottings = deque(maxlen=6)
        self.latest_alert = None
        self.fps = 0.0
        self.load()

    def check_daily_reset(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.date_str:
            self.date_str = today
            self.total_vehicles = 0
            self.counts_by_type = {"car": 0, "moto": 0, "bus": 0, "truck": 0}
            self.taxis_count = 0
            self.prado_count = 0
            self.twingo_count = 0
            self.brand_counts = {k: 0 for k in self.brand_counts}
            self.save()

    def load(self):
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date_str") == self.date_str:
                    self.total_vehicles = data.get("total_vehicles", 0)
                    self.counts_by_type.update(data.get("counts_by_type", {}))
                    self.taxis_count = data.get("taxis_count", 0)
                    self.prado_count = data.get("prado_count", 0)
                    self.twingo_count = data.get("twingo_count", 0)
                    self.brand_counts.update(data.get("brand_counts", {}))
            except Exception as e:
                print(f"[Stats] Could not load previous stats: {e}")

    def save(self):
        try:
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "date_str": self.date_str,
                        "total_vehicles": self.total_vehicles,
                        "counts_by_type": self.counts_by_type,
                        "taxis_count": self.taxis_count,
                        "prado_count": self.prado_count,
                        "twingo_count": self.twingo_count,
                        "brand_counts": self.brand_counts,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"[Stats] Could not save stats: {e}")

    def to_payload(self):
        sorted_brands = sorted(
            self.brand_counts.items(), key=lambda x: x[1], reverse=True
        )[:6]
        return {
            "total_vehicles": self.total_vehicles,
            "counts_by_type": self.counts_by_type,
            "taxis_count": self.taxis_count,
            "prado_count": self.prado_count,
            "twingo_count": self.twingo_count,
            "top_brands": [{"brand": b, "count": c} for b, c in sorted_brands],
            "recent_spottings": list(self.recent_spottings),
            "latest_alert": self.latest_alert,
            "fps": round(self.fps, 1),
        }


state = StreamState()
connected_clients: Set[WebSocket] = set()
latest_jpeg_frame: bytes = b""
vlm_queue: asyncio.Queue = asyncio.Queue(maxsize=15)
# Store labels for tracked vehicles so bounding boxes update once VLM finishes
track_labels: dict[int, str] = {}


async def broadcast_state():
    if not connected_clients:
        return
    payload = json.dumps(state.to_payload())
    dead_clients = set()
    for ws in connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead_clients.add(ws)
    connected_clients.difference_update(dead_clients)


async def vlm_worker():
    """Background worker that pulls cropped cars from the queue and runs local VLM."""
    while True:
        item = await vlm_queue.get()
        try:
            track_id, crop_bgr, v_type = item
            result = await classify_vehicle_crop(crop_bgr, v_type)
            brand = result["brand"]
            model_name = result["model"]
            is_taxi = result["is_taxi"]
            special_tag = result["special_tag"]

            if brand != "Other":
                state.brand_counts[brand] = state.brand_counts.get(brand, 0) + 1

            if is_taxi:
                state.taxis_count += 1
            if special_tag == "PRADO" or "prado" in model_name.lower():
                state.prado_count += 1
                special_tag = "PRADO"
            if special_tag == "TWINGO" or "twingo" in model_name.lower():
                state.twingo_count += 1
                special_tag = "TWINGO"

            badge_label = f"#{track_id} {brand} {model_name}".strip()
            if is_taxi:
                badge_label = f"🚕 TAXI ({brand})"
            track_labels[track_id] = badge_label

            thumb_b64 = encode_crop_to_base64(crop_bgr, max_dim=180)
            spotting = {
                "track_id": int(track_id),
                "type": v_type,
                "brand": brand,
                "model": model_name,
                "color": result["color"],
                "is_taxi": is_taxi,
                "special_tag": special_tag,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "thumb": f"data:image/jpeg;base64,{thumb_b64}",
            }
            state.recent_spottings.appendleft(spotting)

            if special_tag in ("TAXI", "PRADO", "TWINGO"):
                state.latest_alert = {
                    "tag": special_tag,
                    "text": f"{brand} {model_name}",
                    "ts": time.time(),
                }

            state.save()
            await broadcast_state()
        except Exception as e:
            print(f"[VLM Worker Error]: {e}")
        finally:
            vlm_queue.task_done()


async def vision_pipeline_loop():
    """Main 30-60 FPS loop: reads Logitech Brio 4K, runs YOLOv11 + ByteTrack on RTX 4060."""
    global latest_jpeg_frame

    print(f"[AI Engine] Loading {YOLO_MODEL} on GPU...")
    model = YOLO(YOLO_MODEL)

    # Open Logitech Brio 4K with DirectShow on Windows for fast startup
    cap = cv2.VideoCapture(
        CAMERA_SOURCE, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    )
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    tracker = sv.ByteTrack(
        track_activation_threshold=CONFIDENCE_THRESHOLD,
        lost_track_buffer=45,
        minimum_matching_threshold=0.8,
        frame_rate=TARGET_FPS,
    )

    box_annotator = sv.BoxCornerAnnotator(thickness=3, corner_length=18)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.65, text_thickness=2, text_padding=6
    )
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=35)

    counted_ids: set[int] = set()
    previous_centers_y: dict[int, float] = {}
    prev_time = time.time()

    print("[AI Engine] Vision loop running! Open http://localhost:8000 in browser/OBS.")

    while True:
        success, frame = cap.read()
        if not success:
            # If testing with a video file, loop it back to the start
            if isinstance(CAMERA_SOURCE, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                await asyncio.sleep(0.05)
                continue
            print("[Camera Warning] Frame drop, retrying in 1s...")
            await asyncio.sleep(1.0)
            continue

        state.check_daily_reset()
        h, w = frame.shape[:2]
        line_y = int(h * COUNTING_LINE_Y_RATIO)

        # Run YOLOv11 inference
        results = model.predict(
            frame,
            classes=TRACKED_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            half=True,  # Uses FP16 Tensor Cores on RTX 4060
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        annotated = frame.copy()

        # Draw glowing counting line
        cv2.line(annotated, (0, line_y), (w, line_y), (0, 230, 255), 3)
        cv2.putText(
            annotated,
            "SCAN LINE // COLOMBIA AI",
            (20, line_y - 12),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (0, 230, 255),
            2,
        )

        labels = []
        if detections.tracker_id is not None:
            for xyxy, class_id, tracker_id in zip(
                detections.xyxy, detections.class_id, detections.tracker_id
            ):
                tid = int(tracker_id)
                cid = int(class_id)
                v_type = CLASS_NAMES.get(cid, "car")

                x1, y1, x2, y2 = map(int, xyxy)
                cy = (y1 + y2) / 2.0

                # Check if vehicle just crossed the counting line
                prev_cy = previous_centers_y.get(tid)
                previous_centers_y[tid] = cy

                crossed = False
                if prev_cy is not None and tid not in counted_ids:
                    if (prev_cy < line_y <= cy) or (prev_cy > line_y >= cy):
                        crossed = True

                if crossed:
                    counted_ids.add(tid)
                    state.total_vehicles += 1
                    state.counts_by_type[v_type] = (
                        state.counts_by_type.get(v_type, 0) + 1
                    )

                    # Flash the scan line green on crossing
                    cv2.line(annotated, (0, line_y), (w, line_y), (0, 255, 100), 6)

                    # Crop vehicle with a tiny padding for the VLM
                    pad = 12
                    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
                    crop = frame[cy1:cy2, cx1:cx2].copy()

                    if (cx2 - cx1) >= MIN_CROP_SIZE and (cy2 - cy1) >= MIN_CROP_SIZE:
                        if not vlm_queue.full():
                            vlm_queue.put_nowait((tid, crop, v_type))
                    else:
                        await broadcast_state()

                label_str = track_labels.get(tid, f"#{tid} {v_type.upper()}")
                labels.append(label_str)

            # Prevent memory leak on long 24/7 runs
            if len(counted_ids) > 5000:
                counted_ids.clear()
                track_labels.clear()
                previous_centers_y.clear()

            annotated = trace_annotator.annotate(
                scene=annotated, detections=detections
            )
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )

        # Calculate FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            state.fps = 0.9 * state.fps + 0.1 * (1.0 / dt)

        # Encode annotated video for the MJPEG feed inside the TikTok layout
        ok, buf = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
        )
        if ok:
            latest_jpeg_frame = buf.tobytes()

        await asyncio.sleep(0.001)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(vlm_worker())
    asyncio.create_task(vision_pipeline_loop())


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/video_feed")
async def video_feed():
    async def generate():
        while True:
            if latest_jpeg_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + latest_jpeg_frame + b"\r\n"
                )
            await asyncio.sleep(1.0 / TARGET_FPS)

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    await ws.send_text(json.dumps(state.to_payload()))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(ws)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
