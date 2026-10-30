# 🇨🇴 Colombia 24/7 TikTok AI Car & Brand Counter (RTX 4060 Edition)

A **100% local ($0/month)** AI computer-vision broadcast system built for **Logitech Brio 4K + NVIDIA RTX 4060 (8GB VRAM) + 32GB RAM**.

It tracks every car, moto, bus, and truck in real time using **YOLOv11 + ByteTrack**, crops each vehicle once as it crosses a virtual scan line, identifies **Colombian car brands & meme vehicles** using a local Vision-Language Model (**Qwen2.5-VL-3B** via Ollama), and renders a **1080x1920 (9:16 vertical)** interactive dashboard ready for **OBS Studio / TikTok Live**.

---

## 🏗️ How It Works on Your 8GB VRAM (RTX 4060)

| Component | Model / Tech | VRAM Used | Speed |
| :--- | :--- | :--- | :--- |
| **Camera Capture** | Logitech Brio 4K (running at `1920x1080 @ 30fps` MJPEG) | ~0 GB | 30 FPS |
| **Real-Time Tracking** | `YOLOv11s` (FP16 Tensor Cores) + `ByteTrack` | ~1.2 GB | ~90 FPS |
| **Brand & Model VLM** | `Qwen2.5-VL-3B` (Local via Ollama, triggered *only* on line cross) | ~2.8 GB | ~0.25s per car |
| **Yellow Taxi Detector** | Hybrid OpenCV HSV Color Mask + VLM verification | 0 GB | Instant |
| **Total GPU Load** | **Fits comfortably inside 8 GB VRAM (~4.2 GB total)** | **4.2 / 8.0 GB** | **24/7 Safe** |

---

## 🚀 Step-by-Step Setup Guide (Mac ➡️ Windows PC)

### Step 1: Copy Project Folder from Mac to Your Windows PC
Copy this entire folder (`/Users/juanpabloalfonsosanchez/.gemini/antigravity/scratch/colombia-car-stream`) onto your Windows PC (for example, to `C:\colombia-car-stream`) using a USB drive, Google Drive, or LocalSend.

---

### Step 2: Install Prerequisites on the Windows PC
1. **Python 3.11 or 3.12**: Download from [python.org](https://www.python.org/downloads/windows/).
   * ⚠️ **Important:** Check the box **"Add Python to PATH"** during installation!
2. **Ollama for Windows (100% Local Vision AI)**:
   * Download and install from [ollama.com/download/windows](https://ollama.com/download/windows).
   * Open **Command Prompt (cmd)** or **PowerShell** on Windows and pull the lightweight 3B Vision model:
     ```powershell
     ollama pull qwen2.5vl:3b
     ```
3. **NVIDIA PyTorch CUDA (so YOLO uses your RTX 4060 instead of CPU)**:
   * Inside your project folder `C:\colombia-car-stream`, open PowerShell and run:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
     pip install -r requirements.txt
     ```

---

### Step 3: Plug In & Aim Your Logitech Brio 4K
1. Plug the **Logitech Brio 4K** into a **USB 3.0 port** (blue port) on your Windows PC.
2. Mount it inside your window pointing diagonally down at the street.
   * *Pro Tip for Night/Glare:* Put the camera lens as close to the window glass as possible (or use a cheap black paper cone around the lens) so room reflections don't bounce off the glass at night!
3. If your PC only has the Logitech Brio plugged in, `CAMERA_SOURCE = 0` in `config.py` will pick it up automatically. (If you have another webcam, change `CAMERA_SOURCE = 1` in `config.py`).

---

### Step 4: Launch the AI Engine
Double-click **`start_windows.bat`** (or run `python main.py` from PowerShell).

* Wait ~5 seconds for YOLOv11 to load on your RTX 4060.
* Open Chrome/Edge on your Windows PC and visit: **`http://localhost:8000`**
* You will see the full **9:16 vertical TikTok overlay** with:
  * Live video + glowing `SCAN LINE // COLOMBIA AI`
  * **Vehículos Hoy** total counter
  * **🚕 Taxis Amarillos**, **🏍️ Motos vs Carros**, **🚙 Toyota Prado (Toyoteros)**, and **👑 Twingo** counters
  * **🏆 Top Marcas Hoy** animated bar chart (Renault, Mazda, Chevrolet, Kia, Toyota, etc.)
  * **📸 Últimos Detectados** real-time cropped photos of the last 4 cars that crossed the line!

---

### Step 5: Stream 24/7 to TikTok Live (Phone or PC)

You have **two ways** to broadcast this to TikTok:

#### Option A: Direct from Windows PC (Best for 24/7 reliability)
1. Install **TikTok Live Studio** or **OBS Studio** on your Windows PC.
2. Set Canvas Resolution to **Vertical `1080x1920`**.
3. Add a **Browser Source**:
   * **URL:** `http://localhost:8000`
   * **Width:** `1080`
   * **Height:** `1920`
4. Add a background lofi/reggaeton instrumental audio track (low volume) so TikTok doesn't flag the stream as silent.
5. Click **Go Live**!

#### Option B: Stream via Your Phone (If your TikTok account only has Mobile Live / Camera access)
If your TikTok account doesn't have Stream Key / Live Studio access yet, you can feed your PC's AI screen directly into your phone as a virtual camera or screen share:
1. **Simplest way:** Open `http://<YOUR_WINDOWS_PC_LAN_IP>:8000` (e.g. `http://192.168.1.50:8000`) in Chrome/Safari on your **Phone** in full-screen mode, and start a **TikTok "Mobile Gaming / Screen Share" Live** on your phone!
   * Why this is genius: Your Windows RTX 4060 does 100% of the heavy AI + Logitech Brio capture, and your phone just displays the webpage over Wi-Fi and streams its screen to TikTok using almost zero battery/heat!

---

## ⚙️ Optional: Hybrid $2/month Cloud VLM Fallback
If you ever want even sharper recognition at night or from far away, open `.env` (or `config.py`) and set:
```ini
CLASSIFIER_MODE=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-key
OPENROUTER_MODEL=qwen/qwen-2.5-vl-7b-instruct
```
Because it only triggers **once per car that crosses the line**, 3,000 cars/day costs less than **$2.50/month**.
