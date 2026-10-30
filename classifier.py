import base64
import json
import re
import cv2
import httpx
import numpy as np
from config import (
    CLASSIFIER_MODE,
    KNOWN_BRANDS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)

SYSTEM_PROMPT = """You are an expert Colombian street vehicle recognizer.
Look at this cropped vehicle from a street camera in Colombia and return ONLY a valid JSON object with no markdown formatting:
{
  "brand": "One of: Renault, Chevrolet, Mazda, Kia, Toyota, Suzuki, Nissan, Hyundai, Volkswagen, Ford, BYD, Chery, JAC, Bajaj, Yamaha, Honda, AKT, Other",
  "model": "Short model name if recognizable (e.g. Duster, Logan, Sandero, Twingo, Picanto, Spark, Onix, Tracker, CX-30, Mazda 3, Hilux, Prado, Corolla, Boxer, NMAX, Unknown)",
  "color": "Dominant color (e.g. Yellow, White, Gray, Silver, Black, Red, Blue)",
  "is_taxi": true or false (true if it is a yellow Colombian taxi or public service taxi),
  "special_tag": "One of: TWINGO, PRADO, TAXI, CHIVA, SUPERCAR, NONE",
  "confidence": 0.0 to 1.0
}

Rules for Colombia:
1. If it is a yellow car (especially Kia Picanto/Ekotax, Hyundai Grand i10, Chevrolet Spark/Chronos, Renault Logan), set "is_taxi": true and "special_tag": "TAXI".
2. If it looks like a Toyota Land Cruiser Prado / Sumo / TXL / VX, set "brand": "Toyota", "model": "Prado", and "special_tag": "PRADO".
3. If it is a classic Renault Twingo, set "brand": "Renault", "model": "Twingo", and "special_tag": "TWINGO".
4. Always reply with raw JSON only."""


def encode_crop_to_base64(crop_bgr: np.ndarray, max_dim: int = 384) -> str:
    """Resize crop so VLM inference takes <250ms on RTX 4060 and encode as JPEG base64."""
    h, w = crop_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        crop_bgr = cv2.resize(
            crop_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    success, buffer = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def detect_yellow_taxi_heuristic(crop_bgr: np.ndarray) -> bool:
    """Fast OpenCV HSV color check to assist with Colombian Yellow Taxi detection."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Focus on the middle section of the car body
    h, w = hsv.shape[:2]
    center = hsv[int(h * 0.25) : int(h * 0.80), int(w * 0.15) : int(w * 0.85)]
    if center.size == 0:
        return False
    # Colombian taxi yellow hue range
    lower_yellow = np.array([18, 110, 110])
    upper_yellow = np.array([34, 255, 255])
    mask = cv2.inRange(center, lower_yellow, upper_yellow)
    yellow_ratio = np.count_nonzero(mask) / float(mask.size)
    return yellow_ratio > 0.28


def _parse_json_response(raw_text: str) -> dict:
    """Safely extract JSON from model output."""
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data
    except Exception:
        pass
    return {}


async def classify_vehicle_crop(crop_bgr: np.ndarray, yolo_class_name: str) -> dict:
    """
    Classifies a cropped vehicle using local Ollama (qwen2.5vl:3b) on RTX 4060,
    or falls back to OpenRouter API if configured.
    """
    is_yellow_heuristic = (
        detect_yellow_taxi_heuristic(crop_bgr) if yolo_class_name == "car" else False
    )

    # Default fallback result if VLM is busy or fails
    default_result = {
        "brand": "Other",
        "model": yolo_class_name.capitalize(),
        "color": "Yellow" if is_yellow_heuristic else "Unknown",
        "is_taxi": is_yellow_heuristic,
        "special_tag": "TAXI" if is_yellow_heuristic else "NONE",
        "confidence": 0.5,
    }

    b64_img = encode_crop_to_base64(crop_bgr)
    if not b64_img:
        return default_result

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            if CLASSIFIER_MODE == "openrouter" and OPENROUTER_API_KEY:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                    json={
                        "model": OPENROUTER_MODEL,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": SYSTEM_PROMPT},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{b64_img}"
                                        },
                                    },
                                ],
                            }
                        ],
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                raw_text = resp.json()["choices"][0]["message"]["content"]
            else:
                # 100% Local Ollama inference on RTX 4060
                resp = await client.post(
                    OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": SYSTEM_PROMPT,
                        "images": [b64_img],
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 120,
                        },
                    },
                )
                resp.raise_for_status()
                raw_text = resp.json().get("response", "{}")

        parsed = _parse_json_response(raw_text)
        if not parsed:
            return default_result

        brand = parsed.get("brand", "Other").strip()
        # Normalize brand name
        matched_brand = "Other"
        for kb in KNOWN_BRANDS:
            if kb.lower() in brand.lower():
                matched_brand = kb
                break

        is_taxi = bool(parsed.get("is_taxi", False)) or is_yellow_heuristic
        special_tag = parsed.get("special_tag", "NONE").upper()
        if is_taxi and special_tag == "NONE":
            special_tag = "TAXI"

        return {
            "brand": matched_brand,
            "model": parsed.get("model", "Unknown")[:18],
            "color": parsed.get("color", "Unknown"),
            "is_taxi": is_taxi,
            "special_tag": special_tag,
            "confidence": float(parsed.get("confidence", 0.8)),
        }

    except Exception as exc:
        print(f"[Classifier Warning] VLM call skipped/failed: {exc}")
        return default_result
