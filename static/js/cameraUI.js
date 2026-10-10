/**
 * High-Precision Camera & Dual-Engine Neural Vision Service
 * Primary: MediaPipe Hands (21 3D Landmarks) with Geometric Finger State Solver
 * Secondary: Python Precision Backend (/api/predict) with PyTorch MobileNetV2
 * Temporal Stabilizer: Rolling Consensus Window with Exponential Moving Average (EMA)
 */

class CameraService {
  constructor(videoElement, onPredictionCallback) {
    this.video = videoElement;
    this.onPrediction = onPredictionCallback;
    this.stream = null;
    this.isInferencing = false;
    this.loopInterval = null;

    // Landmark Drawing Canvas
    this.canvas = document.getElementById('landmarkCanvas');
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

    // Off-screen canvas for backend fallback
    this.roiCanvas = document.createElement('canvas');
    this.roiCanvas.width = 240;
    this.roiCanvas.height = 240;
    this.roiCtx = this.roiCanvas.getContext('2d');

    // MediaPipe Hand Tracker
    this.handsTracker = null;
    this.isMediaPipeActive = false;

    // Temporal Smoothing & Debounce Buffer
    this.historySize = 5;
    this.history = []; // Array of recent gesture strings
    this.lockedGesture = 'none';
    this.lockedConfidence = 0.0;
    this.smoothedProbs = { rock: 0.0, paper: 0.0, scissors: 0.0 };
    this.noHandCounter = 0;

    // Landmark Connection Graph
    this.connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],       // Index
      [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
      [9, 13], [13, 14], [14, 15], [15, 16],// Ring
      [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
      [0, 17]                               // Palm Base
    ];
  }

  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      this.video.srcObject = this.stream;

      return new Promise((resolve) => {
        this.video.onloadedmetadata = async () => {
          this.video.play();
          if (this.canvas) {
            this.canvas.width = this.video.videoWidth || 640;
            this.canvas.height = this.video.videoHeight || 480;
          }

          // Initialize MediaPipe Hands if library is loaded
          await this.initMediaPipe();

          if (!this.isMediaPipeActive) {
            console.log('[*] MediaPipe unavailable, using backend inference engine');
            this.startBackendLoop();
          }

          resolve(true);
        };
      });
    } catch (err) {
      console.error('Camera initialization failure:', err);
      return false;
    }
  }

  async initMediaPipe() {
    if (typeof window.Hands !== 'undefined') {
      try {
        this.handsTracker = new window.Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        this.handsTracker.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.65,
          minTrackingConfidence: 0.60
        });

        this.handsTracker.onResults((results) => this.onMediaPipeResults(results));

        this.isMediaPipeActive = true;
        this.startMediaPipeLoop();
        console.log('[*] MediaPipe 21-Landmark Hand Engine Activated Successfully!');
      } catch (e) {
        console.warn('Failed to load MediaPipe Hands:', e);
        this.isMediaPipeActive = false;
      }
    }
  }

  startMediaPipeLoop() {
    const processFrame = async () => {
      if (!this.stream) return;
      if (this.video && this.video.readyState >= 2 && !this.isInferencing) {
        this.isInferencing = true;
        try {
          await this.handsTracker.send({ image: this.video });
        } catch (e) {
          // ignore transient frame send errors
        } finally {
          this.isInferencing = false;
        }
      }
      requestAnimationFrame(processFrame);
    };
    requestAnimationFrame(processFrame);
  }

  onMediaPipeResults(results) {
    if (this.ctx && this.canvas) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    // 1. Strict No-Hand Check:
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      this.noHandCounter++;
      // Debounce slightly to prevent single-frame flickering
      if (this.noHandCounter >= 2) {
        this.pushHistory('none', 0.0, { rock: 0, paper: 0, scissors: 0 });
      }
      return;
    }

    this.noHandCounter = 0;
    const landmarks = results.multiHandLandmarks[0];

    // Draw Sleek Cyber-Skeletal Hand
    if (this.ctx && this.canvas) {
      this.drawLandmarks(landmarks);
    }

    // 2. High-Precision 21-Landmark Mathematical Finger Solver
    const gestureData = this.solve21Landmarks(landmarks);
    this.pushHistory(gestureData.class, gestureData.confidence, gestureData.probabilities);
  }

  solve21Landmarks(l) {
    // Check vertical extension and Euclidean distance relative to wrist (0)
    const dist = (i, j) => Math.hypot(l[i].x - l[j].x, l[i].y - l[j].y, (l[i].z - l[j].z) * 0.5);

    // Index (8 vs 6 & 5)
    const isIndexExtended = (l[8].y < l[6].y - 0.02) || (dist(8, 0) > dist(6, 0) * 1.15);

    // Middle (12 vs 10 & 9)
    const isMiddleExtended = (l[12].y < l[10].y - 0.02) || (dist(12, 0) > dist(10, 0) * 1.15);

    // Ring (16 vs 14 & 13)
    const isRingExtended = (l[16].y < l[14].y - 0.02) || (dist(16, 0) > dist(14, 0) * 1.15);

    // Pinky (20 vs 18 & 17)
    const isPinkyExtended = (l[20].y < l[18].y - 0.02) || (dist(20, 0) > dist(18, 0) * 1.15);

    // Thumb (4 vs 3 & 17)
    const isThumbExtended = dist(4, 17) > dist(3, 17) * 1.22;

    const extendedCount = [isIndexExtended, isMiddleExtended, isRingExtended, isPinkyExtended].filter(Boolean).length;

    let pred = 'rock';
    let conf = 0.98;
    let prob = { rock: 0.98, paper: 0.01, scissors: 0.01 };

    // 1. ROCK: All 4 main fingers are folded into palm
    if (extendedCount === 0) {
      pred = 'rock';
      conf = 0.98;
      prob = { rock: 0.98, paper: 0.01, scissors: 0.01 };
    }
    // 2. SCISSORS: Exactly Index and Middle extended, Ring and Pinky folded
    else if (isIndexExtended && isMiddleExtended && !isRingExtended && !isPinkyExtended) {
      pred = 'scissors';
      conf = 0.98;
      prob = { scissors: 0.98, paper: 0.01, rock: 0.01 };
    }
    // 3. PAPER: 3 or 4 fingers extended wide
    else if (extendedCount >= 3) {
      pred = 'paper';
      conf = 0.98;
      prob = { paper: 0.98, rock: 0.01, scissors: 0.01 };
    }
    // 4. Disambiguation edge cases:
    else if (extendedCount === 2) {
      pred = 'scissors';
      conf = 0.92;
      prob = { scissors: 0.92, paper: 0.05, rock: 0.03 };
    } else {
      pred = 'rock';
      conf = 0.88;
      prob = { rock: 0.88, paper: 0.06, scissors: 0.06 };
    }

    return {
      class: pred,
      confidence: conf,
      probabilities: prob
    };
  }

  drawLandmarks(landmarks) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    // Draw skeletal bones
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = 'rgba(130, 182, 154, 0.85)';
    ctx.shadowColor = 'rgba(130, 182, 154, 0.5)';
    ctx.shadowBlur = 4;

    for (const [p1, p2] of this.connections) {
      ctx.beginPath();
      ctx.moveTo(landmarks[p1].x * w, landmarks[p1].y * h);
      ctx.lineTo(landmarks[p2].x * w, landmarks[p2].y * h);
      ctx.stroke();
    }

    // Draw landmark joint nodes
    ctx.shadowBlur = 2;
    for (let i = 0; i < landmarks.length; i++) {
      const x = landmarks[i].x * w;
      const y = landmarks[i].y * h;
      ctx.beginPath();
      ctx.arc(x, y, i === 8 || i === 12 || i === 4 ? 4.5 : 3, 0, 2 * Math.PI);
      ctx.fillStyle = (i === 4 || i === 8 || i === 12 || i === 16 || i === 20) ? '#E8C66A' : '#456FB5';
      ctx.fill();
    }
    ctx.shadowBlur = 0;
  }

  pushHistory(gesture, conf, rawProbs) {
    this.history.push(gesture);
    if (this.history.length > this.historySize) {
      this.history.shift();
    }

    // Temporal Consensus: Count frequencies
    const counts = {};
    for (const g of this.history) {
      counts[g] = (counts[g] || 0) + 1;
    }

    let candidate = 'none';
    let maxCount = 0;
    for (const [g, count] of Object.entries(counts)) {
      if (count > maxCount) {
        maxCount = count;
        candidate = g;
      }
    }

    // Require majority consistency before switching
    if (candidate === 'none') {
      this.lockedGesture = 'none';
      this.lockedConfidence = 0.0;
      this.smoothedProbs = { rock: 0.0, paper: 0.0, scissors: 0.0 };
    } else if (maxCount >= 3) {
      this.lockedGesture = candidate;
      this.lockedConfidence = conf;

      // Exponential Moving Average for smooth gauge transitions
      const alpha = 0.35;
      this.smoothedProbs.rock = alpha * rawProbs.rock + (1 - alpha) * this.smoothedProbs.rock;
      this.smoothedProbs.paper = alpha * rawProbs.paper + (1 - alpha) * this.smoothedProbs.paper;
      this.smoothedProbs.scissors = alpha * rawProbs.scissors + (1 - alpha) * this.smoothedProbs.scissors;
    }

    if (this.onPrediction) {
      this.onPrediction({
        has_hand: this.lockedGesture !== 'none',
        class: this.lockedGesture,
        confidence: this.lockedConfidence,
        probabilities: this.smoothedProbs
      });
    }
  }

  startBackendLoop() {
    if (this.loopInterval) return;

    this.loopInterval = setInterval(async () => {
      if (this.isInferencing || !this.video || !this.video.videoWidth) return;
      this.isInferencing = true;

      try {
        const vw = this.video.videoWidth;
        const vh = this.video.videoHeight;

        const rawCropX = Math.floor(vw * 0.06);
        const rawCropY = Math.floor(vh * 0.10);
        const rawCropW = Math.floor(vw * 0.44);
        const rawCropH = Math.floor(vh * 0.72);

        this.roiCtx.save();
        this.roiCtx.translate(240, 0);
        this.roiCtx.scale(-1, 1);
        this.roiCtx.drawImage(this.video, rawCropX, rawCropY, rawCropW, rawCropH, 0, 0, 240, 240);
        this.roiCtx.restore();

        const b64 = this.roiCanvas.toDataURL('image/jpeg', 0.85);

        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: b64 })
        });

        if (response.ok) {
          const data = await response.json();
          this.pushHistory(data.has_hand ? data.class : 'none', data.confidence || 0, data.probabilities || { rock: 0, paper: 0, scissors: 0 });
        }
      } catch (err) {
        console.warn('Backend inference request failed:', err);
      } finally {
        this.isInferencing = false;
      }
    }, 110);
  }

  stop() {
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
      this.loopInterval = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }
}

window.CameraService = CameraService;
