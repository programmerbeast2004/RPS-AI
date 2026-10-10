# 🎮 Rock Paper Scissors • AI Hand-Drawn Arena

An interactive, high-precision Rock-Paper-Scissors battle arena powered by deep learning and real-time computer vision.

![Screenshot](/static/assets/image/win.jpeg)

---

## ✨ Features

- 🧠 **Dual-Engine Vision**: Real-time MediaPipe 21-landmark hand tracker with fallback to PyTorch MobileNetV2.
- 🎨 **Hand-Drawn Sketchbook Aesthetic**: Cozy watercolor theme matching paper textures and graphite lines.
- 📱 **100% Cross-Device Responsive**: Optimized layout for Desktops, Laptops, Tablets, iPads, and Mobile screens.
- 🔒 **Isolated Match Logs**: Per-user private stats and history tracking.
- 😂 **Meme Reactions**: Dynamic celebration and mocking polaroid memes for Wins, Losses, and Draws.
- 🔊 **Web Audio Synthesizer**: Low-latency browser sound effects without external audio files.

---

## 🚀 Quickstart (Local)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Server**:
   ```bash
   python server.py
   ```

3. **Play**:
   Open [http://localhost:5000](http://localhost:5000) in any modern browser with webcam access.

---

## ☁️ Deployment Guide

### Option 1: Render.com (Recommended Free Hosting)
1. Push your repository to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com/) -> **New Web Service**.
3. Connect your GitHub repository.
4. Set:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT server:app`
5. Click **Deploy Web Service**!

### Option 2: Railway.app / Hugging Face Spaces / Heroku
- Automatically detects `Procfile` and `requirements.txt`.
