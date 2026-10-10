/**
 * Rock Paper Scissors - Application Entry Point (Hand-Drawn Sketchbook Edition)
 * Wires CameraService, GameController, Web Audio, Modals, Stats History, and UI Event Listeners.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // 1. Gather all DOM references matching the screenshot
  const dom = {
    // Camera Elements
    video: document.getElementById('webcamVideo'),
    guideBox: document.getElementById('guideBox'),
    guidePill: document.getElementById('guidePill'),
    camStatusBadge: document.getElementById('camStatusBadge'),
    camCountdown: document.getElementById('camCountdown'),
    countdownCircleArc: document.getElementById('countdownCircleArc'),
    centerPromptText: document.getElementById('centerPromptText'),

    // Format & Round
    formatText: document.getElementById('formatText'),
    formatDropdownTrigger: document.getElementById('formatDropdownTrigger'),
    formatDropdownMenu: document.getElementById('formatDropdownMenu'),
    format3Btn: document.getElementById('format3Btn'),
    format5Btn: document.getElementById('format5Btn'),
    format10Btn: document.getElementById('format10Btn'),
    roundText: document.getElementById('roundText'),
    roundDotsContainer: document.getElementById('roundDotsContainer'),

    // Scoreboard
    scorePlayer: document.getElementById('scorePlayer'),
    scoreAi: document.getElementById('scoreAi'),

    // AI Display Card
    aiCard: document.getElementById('aiCard'),
    aiVisual: document.getElementById('aiVisual'),
    aiAction: document.getElementById('aiAction'),
    aiStatusText: document.getElementById('aiStatusText'),

    // Control Buttons
    shootBtn: document.getElementById('btnShoot'),
    pauseBtn: document.getElementById('btnPause'),
    pauseBtnText: document.getElementById('pauseBtnText'),
    btnReset: document.getElementById('btnReset'),
    autoBtn: document.getElementById('btnAuto'),

    // Match Summary Modal
    summaryModal: document.getElementById('summaryModal'),
    summaryTitle: document.getElementById('summaryTitle'),
    summaryDesc: document.getElementById('summaryDesc'),
    summaryScore: document.getElementById('summaryScore'),
    summaryMemeImg: document.getElementById('summaryMemeImg'),
    summaryMemeCaption: document.getElementById('summaryMemeCaption'),
    btnModalPlayAgain: document.getElementById('btnModalPlayAgain'),
    btnModalClose: document.getElementById('btnModalClose'),

    // Stats & History Modal
    btnStats: document.getElementById('btnStats'),
    statsModal: document.getElementById('statsModal'),
    btnStatsClose: document.getElementById('btnStatsClose'),
    btnStatsDone: document.getElementById('btnStatsDone'),
    statTotalMatches: document.getElementById('statTotalMatches'),
    statWinRate: document.getElementById('statWinRate'),
    statRecord: document.getElementById('statRecord'),
    historyListContainer: document.getElementById('historyListContainer'),
    historyCountText: document.getElementById('historyCountText'),

    // How to Play Modal
    btnHowToPlay: document.getElementById('btnHowToPlay'),
    howToPlayModal: document.getElementById('howToPlayModal'),
    btnHtpClose: document.getElementById('btnHtpClose'),
    btnHtpDone: document.getElementById('btnHtpDone'),

    // Settings Modal
    btnSettings: document.getElementById('btnSettings'),
    settingsModal: document.getElementById('settingsModal'),
    btnSettingsClose: document.getElementById('btnSettingsClose'),
    btnSettingsDone: document.getElementById('btnSettingsDone'),
    btnToggleSound: document.getElementById('btnToggleSound')
  };

  // 2. Initialize Game Controller
  const game = new GameController(dom);
  window.gameController = game;
  game.init();

  // 3. Initialize Camera Service with Real-Time Prediction Callback
  const camera = new CameraService(dom.video, (predictionData) => {
    game.onPrediction(predictionData);
  });
  window.cameraService = camera;

  const camStarted = await camera.start();
  if (!camStarted) {
    if (dom.centerPromptText) {
      dom.centerPromptText.textContent = 'Camera blocked';
    }
  }

  // 4. Attach Event Listeners
  if (dom.shootBtn) {
    dom.shootBtn.addEventListener('click', () => {
      if (!game.isActive) {
        game.startMatch();
      } else {
        game.startTurnCountdown();
      }
    });
  }

  if (dom.pauseBtn) {
    dom.pauseBtn.addEventListener('click', () => {
      game.pauseResume();
    });
  }

  if (dom.btnReset) {
    dom.btnReset.addEventListener('click', () => {
      game.resetMatch();
    });
  }

  if (dom.autoBtn) {
    dom.autoBtn.addEventListener('click', () => {
      game.toggleAutoPlay();
    });
  }

  // Format Dropdown Controls
  if (dom.formatDropdownTrigger && dom.formatDropdownMenu) {
    dom.formatDropdownTrigger.addEventListener('click', (e) => {
      e.stopPropagation();
      dom.formatDropdownMenu.classList.toggle('open');
    });

    document.addEventListener('click', () => {
      dom.formatDropdownMenu.classList.remove('open');
    });
  }

  const setFormatHandler = (turns) => {
    game.setFormat(turns);
    [dom.format3Btn, dom.format5Btn, dom.format10Btn].forEach(btn => {
      if (btn) btn.classList.toggle('active', btn.id === `format${turns}Btn`);
    });
    if (dom.formatDropdownMenu) dom.formatDropdownMenu.classList.remove('open');
  };

  if (dom.format3Btn) dom.format3Btn.addEventListener('click', () => setFormatHandler(3));
  if (dom.format5Btn) dom.format5Btn.addEventListener('click', () => setFormatHandler(5));
  if (dom.format10Btn) dom.format10Btn.addEventListener('click', () => setFormatHandler(10));

  // Settings Segmented Format Buttons
  document.querySelectorAll('.format-seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.format-seg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const turns = parseInt(btn.getAttribute('data-turns'), 10);
      setFormatHandler(turns);
    });
  });

  // 5. Stats & Match History Modal Loader (User-Isolated)
  async function loadMatchStats() {
    if (!dom.historyListContainer) return;
    try {
      dom.historyListContainer.innerHTML = '<div class="history-empty-msg">Loading match records...</div>';
      const uid = window.getUserId ? window.getUserId() : 'guest';
      
      let history = [];
      try {
        const res = await fetch(`/api/game/history?user_id=${encodeURIComponent(uid)}`, {
          headers: { 'X-User-ID': uid }
        });
        if (res.ok) {
          history = await res.json();
        }
      } catch (netErr) {}

      // If network empty, check client-side isolated storage
      if (!history || history.length === 0) {
        try {
          history = JSON.parse(localStorage.getItem(`rps_history_${uid}`) || '[]');
        } catch (e) {
          history = [];
        }
      }

      if (!history || history.length === 0) {
        if (dom.statTotalMatches) dom.statTotalMatches.textContent = '0';
        if (dom.statWinRate) dom.statWinRate.textContent = '0%';
        if (dom.statRecord) dom.statRecord.textContent = '0W-0L';
        if (dom.historyCountText) dom.historyCountText.textContent = '0 recorded';
        dom.historyListContainer.innerHTML = '<div class="history-empty-msg">No matches recorded yet. Play a game to see your personal stats!</div>';
        return;
      }

      let wins = 0;
      let losses = 0;
      let draws = 0;

      let html = '';
      history.forEach(item => {
        const isWin = item.winner === 'Player';
        const isLoss = item.winner === 'AI';
        const isDraw = item.winner === 'Draw' || !item.winner;

        if (isWin) wins++;
        else if (isLoss) losses++;
        else draws++;

        const tagClass = isWin ? 'won' : (isLoss ? 'lost' : 'draw');
        const tagText = isWin ? 'WON' : (isLoss ? 'LOST' : 'DRAW');

        html += `
          <div class="history-item-row">
            <div class="history-item-left">
              <span class="history-item-format">${item.format || 'Match'}</span>
              <span class="history-item-date">${item.timestamp || 'Recent'}</span>
            </div>
            <div class="history-item-right">
              <span class="history-item-score">${item.player_score || 0} — ${item.ai_score || 0}</span>
              <span class="outcome-tag ${tagClass}">${tagText}</span>
            </div>
          </div>
        `;
      });

      const total = history.length;
      const winRate = total > 0 ? Math.round((wins / total) * 100) : 0;

      if (dom.statTotalMatches) dom.statTotalMatches.textContent = total;
      if (dom.statWinRate) dom.statWinRate.textContent = `${winRate}%`;
      if (dom.statRecord) dom.statRecord.textContent = `${wins}W-${losses}L${draws > 0 ? `-${draws}D` : ''}`;
      if (dom.historyCountText) dom.historyCountText.textContent = `${total} recorded`;

      dom.historyListContainer.innerHTML = html;
    } catch (err) {
      dom.historyListContainer.innerHTML = '<div class="history-empty-msg">Unable to load match records.</div>';
    }
  }

  if (dom.btnStats) {
    dom.btnStats.addEventListener('click', () => {
      if (dom.statsModal) dom.statsModal.classList.add('open');
      loadMatchStats();
    });
  }

  if (dom.btnStatsClose) {
    dom.btnStatsClose.addEventListener('click', () => {
      if (dom.statsModal) dom.statsModal.classList.remove('open');
    });
  }

  if (dom.btnStatsDone) {
    dom.btnStatsDone.addEventListener('click', () => {
      if (dom.statsModal) dom.statsModal.classList.remove('open');
    });
  }

  // How to Play Dialog
  if (dom.btnHowToPlay) {
    dom.btnHowToPlay.addEventListener('click', () => {
      if (dom.howToPlayModal) dom.howToPlayModal.classList.add('open');
    });
  }

  if (dom.btnHtpClose) {
    dom.btnHtpClose.addEventListener('click', () => {
      if (dom.howToPlayModal) dom.howToPlayModal.classList.remove('open');
    });
  }

  if (dom.btnHtpDone) {
    dom.btnHtpDone.addEventListener('click', () => {
      if (dom.howToPlayModal) dom.howToPlayModal.classList.remove('open');
    });
  }

  // Settings Dialog
  if (dom.btnSettings) {
    dom.btnSettings.addEventListener('click', () => {
      if (dom.settingsModal) dom.settingsModal.classList.add('open');
    });
  }

  if (dom.btnSettingsClose) {
    dom.btnSettingsClose.addEventListener('click', () => {
      if (dom.settingsModal) dom.settingsModal.classList.remove('open');
    });
  }

  if (dom.btnSettingsDone) {
    dom.btnSettingsDone.addEventListener('click', () => {
      if (dom.settingsModal) dom.settingsModal.classList.remove('open');
    });
  }

  // Sound Toggle
  if (dom.btnToggleSound) {
    dom.btnToggleSound.addEventListener('click', () => {
      window.soundService.muted = !window.soundService.muted;
      dom.btnToggleSound.textContent = window.soundService.muted ? 'Sound: OFF' : 'Sound: ON';
    });
  }

  // Summary / Match Completion Modal
  if (dom.btnModalPlayAgain) {
    dom.btnModalPlayAgain.addEventListener('click', () => {
      if (dom.summaryModal) dom.summaryModal.classList.remove('open');
      game.startMatch();
    });
  }

  if (dom.btnModalClose) {
    dom.btnModalClose.addEventListener('click', () => {
      if (dom.summaryModal) dom.summaryModal.classList.remove('open');
    });
  }

  // Keyboard accessibility shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.repeat && document.activeElement.tagName !== 'BUTTON') {
      e.preventDefault();
      if (!game.isActive) {
        game.startMatch();
      } else if (!game.isPaused && !game.isCountingDown) {
        game.startTurnCountdown();
      }
    } else if (e.code === 'KeyP') {
      game.pauseResume();
    }
  });

  // Cleanup on unload
  window.addEventListener('beforeunload', () => {
    camera.stop();
    game.clearTimers();
  });
});
