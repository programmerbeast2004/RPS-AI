/**
 * Audio Synthesizer via Web Audio API
 * Generates restrained, polished sound feedback without external asset dependencies.
 */
class SoundService {
  constructor() {
    this.audioCtx = null;
    this.muted = false;
  }

  init() {
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        this.audioCtx = new AudioContext();
      }
    }
    if (this.audioCtx && this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  playTone(freq, duration, type = 'sine', volume = 0.08) {
    if (this.muted) return;
    this.init();
    if (!this.audioCtx) return;

    try {
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);

      gain.gain.setValueAtTime(volume, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, this.audioCtx.currentTime + duration);

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      osc.start();
      osc.stop(this.audioCtx.currentTime + duration);
    } catch (e) {
      // Audio autoplay policy fallback
    }
  }

  tick() {
    this.playTone(720, 0.05, 'triangle', 0.06);
  }

  shoot() {
    this.playTone(980, 0.12, 'sine', 0.1);
  }

  win() {
    this.playTone(880, 0.12, 'sine', 0.09);
    setTimeout(() => this.playTone(1174, 0.18, 'sine', 0.09), 100);
  }

  loss() {
    this.playTone(392, 0.22, 'triangle', 0.08);
  }

  draw() {
    this.playTone(587, 0.12, 'sine', 0.07);
  }
}

window.soundService = new SoundService();

// Unique Client-Side User Identification (Guarantees 100% Isolated Match History)
function getUserId() {
  let uid = localStorage.getItem('rps_user_id');
  if (!uid || uid.length < 5) {
    uid = 'usr_' + Date.now().toString(36) + '_' + Math.random().toString(36).substring(2, 10);
    localStorage.setItem('rps_user_id', uid);
  }
  return uid;
}

window.getUserId = getUserId;
