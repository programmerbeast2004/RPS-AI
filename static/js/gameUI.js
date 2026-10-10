/**
 * Rock Paper Scissors Game State Controller - Hand-Drawn Sketchbook Edition
 * Manages Match State, Dynamic Round Dots, Countdown Ring, Scoring, and Modal Controls.
 */
class GameController {
  constructor(domRefs) {
    this.dom = domRefs;

    // Match Configuration
    this.maxTurns = 5;
    this.currentTurn = 0;
    this.playerScore = 0;
    this.aiScore = 0;
    this.draws = 0;

    // Execution States
    this.isActive = false;
    this.isPaused = false;
    this.isCountingDown = false;
    this.countdownVal = 3;
    this.countdownTimer = null;

    // Auto-Play Settings
    this.isAutoPlay = true;
    this.autoTimer = null;
    this.roundResetTimer = null;

    // Live AI Observation
    this.hasHand = false;
    this.currentLiveMove = 'none';
    this.currentLiveConf = 0.0;

    // Symbols & Representations
    this.icons = {
      rock: '✊',
      paper: '✋',
      scissors: '✌️',
      none: '—'
    };

    this.options = ['rock', 'paper', 'scissors'];
  }

  init() {
    // Preload match outcome images into browser memory
    this.cachedImages = {
      win: new Image(),
      lose: new Image(),
      tie: new Image()
    };
    this.cachedImages.win.src = '/static/assets/image/win.jpeg';
    this.cachedImages.lose.src = '/static/assets/image/lose.jpeg';
    this.cachedImages.tie.src = '/static/assets/image/tie.jpeg';

    const savedFormat = localStorage.getItem('rps_format');
    if (savedFormat) {
      this.setFormat(parseInt(savedFormat, 10), false);
    }
    const savedAuto = localStorage.getItem('rps_auto');
    if (savedAuto !== null) {
      this.isAutoPlay = (savedAuto === 'true');
      this.updateAutoPlayButton();
    }
    this.render();
  }

  setFormat(turns, promptUser = true) {
    if (promptUser && this.isActive && (this.playerScore > 0 || this.aiScore > 0)) {
      if (!confirm('Changing the match length will reset the current score. Proceed?')) {
        return;
      }
    }
    this.maxTurns = turns;
    localStorage.setItem('rps_format', turns);
    if (this.dom.formatText) {
      this.dom.formatText.textContent = `Best of ${turns}`;
    }
    this.resetMatch();
  }

  toggleAutoPlay() {
    this.isAutoPlay = !this.isAutoPlay;
    localStorage.setItem('rps_auto', this.isAutoPlay);
    this.updateAutoPlayButton();

    if (this.isAutoPlay && this.isActive && !this.isPaused && !this.isCountingDown && this.currentTurn < this.maxTurns) {
      this.scheduleNextTurn(1400);
    } else if (!this.isAutoPlay && this.autoTimer) {
      clearTimeout(this.autoTimer);
      this.autoTimer = null;
    }
  }

  updateAutoPlayButton() {
    if (this.dom.autoBtn) {
      this.dom.autoBtn.classList.toggle('active', this.isAutoPlay);
      this.dom.autoBtn.querySelector('span').textContent = this.isAutoPlay ? 'Auto: ON' : 'Auto: OFF';
    }
  }

  startMatch() {
    this.clearTimers();
    this.currentTurn = 0;
    this.playerScore = 0;
    this.aiScore = 0;
    this.draws = 0;
    this.isActive = true;
    this.isPaused = false;
    this.isCountingDown = false;

    this.resetRoundVisuals();
    this.setPromptText('Get Ready!');
    this.render();

    if (this.isAutoPlay) {
      this.scheduleNextTurn(900);
    }
  }

  pauseResume() {
    if (!this.isActive) return;

    this.isPaused = !this.isPaused;
    if (this.isPaused) {
      this.clearTimers();
      this.isCountingDown = false;
      this.setPromptText('Paused');
      if (this.dom.pauseBtnText) this.dom.pauseBtnText.textContent = 'Resume';
      if (this.dom.shootBtn) this.dom.shootBtn.disabled = true;
    } else {
      this.setPromptText('Resumed!');
      if (this.dom.pauseBtnText) this.dom.pauseBtnText.textContent = 'Pause';
      if (this.dom.shootBtn) this.dom.shootBtn.disabled = false;
      if (this.isAutoPlay) {
        this.scheduleNextTurn(1000);
      }
    }
  }

  resetMatch() {
    this.clearTimers();
    this.currentTurn = 0;
    this.playerScore = 0;
    this.aiScore = 0;
    this.draws = 0;
    this.isActive = false;
    this.isPaused = false;
    this.isCountingDown = false;

    if (this.dom.pauseBtnText) this.dom.pauseBtnText.textContent = 'Pause';
    if (this.dom.shootBtn) this.dom.shootBtn.disabled = false;

    this.resetRoundVisuals();
    this.setPromptText('Get Ready!');
    this.setCountdownVal(3);
    this.render();
  }

  /**
   * Resets ONLY the fighter cards back to clean READY state between rounds,
   * without clearing the scoreboard or round count.
   */
  resetRoundVisuals() {
    if (this.dom.aiVisual) this.dom.aiVisual.textContent = '✌️';
    if (this.dom.aiAction) this.dom.aiAction.textContent = 'READY';
    if (this.dom.aiStatusText) this.dom.aiStatusText.textContent = 'Thinking...';
    if (this.dom.aiCard) this.dom.aiCard.classList.remove('win');
  }

  clearTimers() {
    if (this.countdownTimer) {
      clearTimeout(this.countdownTimer);
      this.countdownTimer = null;
    }
    if (this.autoTimer) {
      clearTimeout(this.autoTimer);
      this.autoTimer = null;
    }
    if (this.roundResetTimer) {
      clearTimeout(this.roundResetTimer);
      this.roundResetTimer = null;
    }
  }

  scheduleNextTurn(delayMs = 1800) {
    this.clearTimers();
    if (this.isActive && !this.isPaused && this.currentTurn < this.maxTurns) {
      this.roundResetTimer = setTimeout(() => {
        if (!this.isCountingDown) {
          this.resetRoundVisuals();
          this.setPromptText('Get Ready!');
        }
      }, 1200);

      this.autoTimer = setTimeout(() => {
        this.startTurnCountdown();
      }, delayMs);
    }
  }

  startTurnCountdown() {
    if (!this.isActive || this.isPaused || this.isCountingDown || this.currentTurn >= this.maxTurns) return;

    this.clearTimers();
    this.resetRoundVisuals();

    this.isCountingDown = true;
    this.countdownVal = 3;
    if (this.dom.shootBtn) this.dom.shootBtn.disabled = true;

    this.setPromptText('Show Hand!');
    if (this.dom.aiStatusText) this.dom.aiStatusText.textContent = 'Thinking...';

    this.tickCountdown();
  }

  tickCountdown() {
    if (this.isPaused) return;

    if (this.countdownVal > 0) {
      window.soundService.tick();
      this.setCountdownVal(this.countdownVal);

      // AI shuffle during countdown
      const sample = this.options[Math.floor(Math.random() * 3)];
      if (this.dom.aiVisual) this.dom.aiVisual.textContent = this.icons[sample];
      if (this.dom.aiAction) this.dom.aiAction.textContent = 'THINKING...';

      this.countdownVal--;
      this.countdownTimer = setTimeout(() => this.tickCountdown(), 800);
    } else if (this.countdownVal === 0) {
      window.soundService.shoot();
      this.setCountdownVal('GO!');
      this.setPromptText('Shoot!');
      this.countdownVal--;
      this.countdownTimer = setTimeout(() => this.resolveTurn(), 320);
    }
  }

  resolveTurn() {
    this.isCountingDown = false;
    if (this.dom.shootBtn && !this.isPaused) this.dom.shootBtn.disabled = false;

    // Strict validation: Hand presence required
    if (!this.hasHand || this.currentLiveMove === 'none') {
      window.soundService.loss();
      this.setPromptText('No Hand Detected!');
      this.setCountdownVal('—');
      if (this.isAutoPlay && this.isActive && !this.isPaused) {
        this.scheduleNextTurn(1800);
      }
      return;
    }

    const playerMove = this.currentLiveMove;
    const aiMove = this.options[Math.floor(Math.random() * 3)];
    this.currentTurn++;

    // Display the round resolution
    if (this.dom.aiVisual) this.dom.aiVisual.textContent = this.icons[aiMove];
    if (this.dom.aiAction) this.dom.aiAction.textContent = aiMove.toUpperCase();
    if (this.dom.aiStatusText) this.dom.aiStatusText.textContent = 'Decided!';

    if (this.dom.aiCard) this.dom.aiCard.classList.remove('win');

    if (playerMove === aiMove) {
      window.soundService.draw();
      this.draws++;
      const drawQuotes = ['🎮 PASS THE CONTROLLER!', '🤖 SHARING BRAINCELLS!', '🤝 PERFECT MATCH!', '👀 GLITCH IN THE MATRIX!'];
      const q = drawQuotes[Math.floor(Math.random() * drawQuotes.length)];
      this.setPromptText(`${q} (Both: ${playerMove.toUpperCase()})`);
      this.setCountdownVal('🤝');
    } else if (
      (playerMove === 'rock' && aiMove === 'scissors') ||
      (playerMove === 'paper' && aiMove === 'rock') ||
      (playerMove === 'scissors' && aiMove === 'paper')
    ) {
      window.soundService.win();
      this.playerScore++;
      const winQuotes = ['🔥 GIGACHAD MOVE!', '🚀 EZ CLAP!', '⚡ ROBOT COOKED!', '👑 BIG BRAIN!'];
      const q = winQuotes[Math.floor(Math.random() * winQuotes.length)];
      this.setPromptText(`${q} (${playerMove.toUpperCase()} beats ${aiMove.toUpperCase()})`);
      this.setCountdownVal('🎉');
    } else {
      window.soundService.loss();
      this.aiScore++;
      if (this.dom.aiCard) this.dom.aiCard.classList.add('win');
      const loseQuotes = ['💀 SKILL ISSUE DETECTED!', '🧠 BUFFERING TOO HARD...', '😭 BRO GOT OUTPLAYED!', '🤖 AI: "FREE WIN!"'];
      const q = loseQuotes[Math.floor(Math.random() * loseQuotes.length)];
      this.setPromptText(`${q} (${aiMove.toUpperCase()} beats ${playerMove.toUpperCase()})`);
      this.setCountdownVal('🤖');
    }

    // Update scoreboard and round dots
    this.render();

    // Check Match Completion
    if (this.currentTurn >= this.maxTurns) {
      this.isActive = false;
      if (this.dom.shootBtn) this.dom.shootBtn.disabled = true;
      setTimeout(() => this.showMatchSummary(), 600);
    } else if (this.isAutoPlay && !this.isPaused) {
      this.scheduleNextTurn(2000);
    } else {
      // Clean up after 1.5s in manual mode
      this.roundResetTimer = setTimeout(() => {
        this.resetRoundVisuals();
        this.setPromptText('Get Ready!');
        this.setCountdownVal(3);
      }, 1500);
    }
  }

  setPromptText(text) {
    if (this.dom.centerPromptText) {
      this.dom.centerPromptText.textContent = text;
    }
  }

  setCountdownVal(val) {
    if (this.dom.camCountdown) {
      this.dom.camCountdown.textContent = val;
    }
    // Update SVG arc
    if (this.dom.countdownCircleArc) {
      const maxLen = 264;
      if (typeof val === 'number') {
        const offset = maxLen * (1 - val / 3);
        this.dom.countdownCircleArc.style.strokeDashoffset = offset;
      } else {
        this.dom.countdownCircleArc.style.strokeDashoffset = 0;
      }
    }
  }

  showMatchSummary() {
    let outcome = 'Draw';
    let title = '🤝 IT\'S A STALEMATE!';
    let desc = `Scores were level at the end of the battle (${this.playerScore} — ${this.aiScore}).`;
    let memeSrc = '/static/assets/image/tie.jpeg';
    let memeCaption = '🎮 "IF YOU AIN\'T GONNA PLAY, PASS THE CONTROLLER!"';

    const winCaptions = [
      '🔥 GIGACHAD ENERGY! You crushed the AI to scrap metal!',
      '🚀 EZ CLAP! The AI is crying in binary right now!',
      '👑 Pure skill! The machine was no match for you!',
      '⚡ VICTORY! Pass the tissue to the robot! 😂'
    ];

    const loseCaptions = [
      '💀 Bro got outplayed by 50 lines of Python code...',
      '🧠 Buffering... your brain was loading too slow! 😂',
      '🤖 AI: "Thanks for the free win, human!"',
      '😭 Massive skill issue! The robot just cooked you!',
      '📉 Need a tutorial? Pass the controller to someone else! 🎮'
    ];

    const tieCaptions = [
      '🎮 "IF YOU AIN\'T GONNA PLAY, PASS THE CONTROLLER!"',
      '🤖 Are you two sharing the same braincell? Stalemate!',
      '🤝 Perfectly balanced... neither of you could finish the job!',
      '👀 A glitch in the matrix! Synchronized thoughts!'
    ];

    if (this.playerScore > this.aiScore) {
      window.soundService.win();
      outcome = 'Player';
      title = '🏆 VICTORY ROYALE!';
      desc = `You defeated the AI by ${this.playerScore} to ${this.aiScore} (${this.draws} draws).`;
      memeSrc = '/static/assets/image/win.jpeg';
      memeCaption = winCaptions[Math.floor(Math.random() * winCaptions.length)];
    } else if (this.aiScore > this.playerScore) {
      window.soundService.loss();
      outcome = 'AI';
      title = '💀 SKILL ISSUE DETECTED!';
      desc = `The AI won the match by ${this.aiScore} to ${this.playerScore} (${this.draws} draws).`;
      memeSrc = '/static/assets/image/lose.jpeg';
      memeCaption = loseCaptions[Math.floor(Math.random() * loseCaptions.length)];
    } else {
      window.soundService.draw();
      memeCaption = tieCaptions[Math.floor(Math.random() * tieCaptions.length)];
    }

    if (this.dom.summaryTitle) this.dom.summaryTitle.textContent = title;
    if (this.dom.summaryDesc) this.dom.summaryDesc.textContent = desc;
    if (this.dom.summaryScore) {
      this.dom.summaryScore.textContent = `${this.playerScore} — ${this.aiScore}`;
    }

    // Update Reaction Polaroid Meme Image & Dynamic Caption
    const memeImg = document.getElementById('summaryMemeImg');
    const memeCaptionEl = document.getElementById('summaryMemeCaption');

    if (memeImg) {
      memeImg.onerror = function() {
        // Safe retry or fallback
        if (!this.dataset.retry) {
          this.dataset.retry = '1';
          this.src = memeSrc;
        }
      };
      memeImg.src = memeSrc;
    }

    if (memeCaptionEl) {
      memeCaptionEl.textContent = memeCaption;
    }

    if (this.dom.summaryModal) {
      this.dom.summaryModal.classList.add('open');
    }

    // Persist match via backend and local user storage
    const uid = window.getUserId ? window.getUserId() : 'guest';
    const matchRecord = {
      timestamp: new Date().toLocaleString(),
      format: `Best of ${this.maxTurns}`,
      player_score: this.playerScore,
      ai_score: this.aiScore,
      draws: this.draws,
      winner: outcome
    };

    // Save in user's isolated local storage
    try {
      const localHistoryKey = `rps_history_${uid}`;
      const existing = JSON.parse(localStorage.getItem(localHistoryKey) || '[]');
      existing.unshift(matchRecord);
      localStorage.setItem(localHistoryKey, JSON.stringify(existing.slice(0, 40)));
    } catch (e) {}

    // Send to isolated server endpoint
    fetch('/api/game/record', {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'X-User-ID': uid
      },
      body: JSON.stringify({
        user_id: uid,
        format: `Best of ${this.maxTurns}`,
        player_score: this.playerScore,
        ai_score: this.aiScore,
        draws: this.draws,
        winner: outcome
      })
    }).catch(() => {});
  }

  onPrediction(data) {
    this.hasHand = data.has_hand;
    this.currentLiveMove = data.class || 'none';
    this.currentLiveConf = data.confidence || 0.0;

    // Strict No-Hand Handling
    if (!this.hasHand || this.currentLiveMove === 'none') {
      if (this.dom.guideBox) this.dom.guideBox.classList.remove('active');
      if (this.dom.camStatusBadge) {
        this.dom.camStatusBadge.className = 'hand-detected-pill';
        this.dom.camStatusBadge.textContent = 'NO HAND';
      }
      return;
    }

    // Hand Detected
    const pct = Math.round(this.currentLiveConf * 100);
    const moveName = this.currentLiveMove.toUpperCase();

    if (this.dom.guideBox) this.dom.guideBox.classList.add('active');
    if (this.dom.camStatusBadge) {
      this.dom.camStatusBadge.className = 'hand-detected-pill ready';
      this.dom.camStatusBadge.textContent = `${moveName} (${pct}%)`;
    }
  }

  render() {
    if (this.dom.roundText) {
      this.dom.roundText.textContent = `${this.currentTurn} / ${this.maxTurns}`;
    }

    // Render Round Dots matching screenshot
    if (this.dom.roundDotsContainer) {
      let dotsHtml = '';
      for (let i = 0; i < this.maxTurns; i++) {
        const isActive = i < this.currentTurn || (i === 0 && this.currentTurn === 0);
        dotsHtml += `<span class="round-dot ${isActive ? 'active' : ''}"></span>`;
      }
      this.dom.roundDotsContainer.innerHTML = dotsHtml;
    }

    if (this.dom.scorePlayer) this.dom.scorePlayer.textContent = this.playerScore;
    if (this.dom.scoreAi) this.dom.scoreAi.textContent = this.aiScore;

    if (this.dom.formatText) {
      this.dom.formatText.textContent = `Best of ${this.maxTurns}`;
    }
  }
}

window.GameController = GameController;
