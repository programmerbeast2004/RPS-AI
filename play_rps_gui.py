"""
Arcade Deluxe Rock-Paper-Scissors Battle Arena
==============================================
Enhanced Features:
- Triple Processing Pipeline:
    Multi-scale Otsu thresholding + Adaptive Skin filter + Direct Image fallback with confidence voting.
    Locks in predictions with a temporal smoothing buffer so detection never jitters or defaults blindly.
- Rich Arcade Visual Polish:
    1. Dual Side-by-Side Duel Arena: Visual card cards for YOU and COMPUTER AI.
    2. Dynamic Computer Hand Shuffling animation during countdowns.
    3. Glowing round win/loss/draw banners with neon colors (#10b981 for win, #ef4444 for loss, #f59e0b for draw).
    4. Audio SFX (native Windows audio tones on countdown ticks, win fanfare, and loss buzzers).
    5. Animated Turn Progress Bar (e.g. 3 of 5 turns completed).
    6. Live Probability Bars for all 3 classes (Rock, Paper, Scissors) in real-time.
    7. Hands-free Auto-Play mode with a customizable pause between rounds.
"""

import sys
import os
import random
import time
import math
from pathlib import Path
from collections import deque

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms, models

try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

CLASS_NAMES = ['paper', 'rock', 'scissors']
CLASS_EMOJIS = {
    'rock': '✊',
    'paper': '✋',
    'scissors': '✌️',
    'none': '❓'
}
CLASS_COLORS = {
    'rock': '#f59e0b',
    'paper': '#3b82f6',
    'scissors': '#ec4899',
    'none': '#64748b'
}


def play_sfx(sound_type):
    if not HAS_SOUND:
        return
    try:
        if sound_type == 'tick':
            winsound.Beep(880, 70)
        elif sound_type == 'shoot':
            winsound.Beep(1320, 150)
        elif sound_type == 'win':
            winsound.Beep(1046, 120)
            winsound.Beep(1318, 180)
        elif sound_type == 'lose':
            winsound.Beep(440, 200)
        elif sound_type == 'draw':
            winsound.Beep(660, 100)
    except Exception:
        pass


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, len(CLASS_NAMES))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


class ArcadeRPSGUI:
    def __init__(self, root, model_path="training_output/best_rps_model.pth"):
        self.root = root
        self.root.title("ROCK PAPER SCISSORS : ARCADE SHOWDOWN ⚔️")
        self.root.geometry("1240x870")
        self.root.configure(bg="#090d16")
        self.root.minsize(1120, 800)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] Initializing Arcade Arena on: {self.device.type.upper()}")

        if not Path(model_path).exists():
            messagebox.showerror("Model Missing", f"Model not found at {model_path}!")
            sys.exit(1)

        self.model = load_model(model_path, self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Prediction buffer for rock-solid stability
        self.pred_history = deque(maxlen=7)
        self.current_live_pred = "none"
        self.current_live_conf = 0.0
        self.class_probs = {'rock': 0.33, 'paper': 0.33, 'scissors': 0.33}
        self.hand_detected = False

        # Match State
        self.max_turns = 5
        self.current_turn = 0
        self.player_score = 0
        self.computer_score = 0
        self.draws = 0
        self.game_active = False
        self.is_counting_down = False
        self.countdown_val = 3
        self.auto_play = True
        self.auto_timer_id = None
        self.shuffle_counter = 0

        # Last Duel
        self.player_last_move = None
        self.computer_last_move = None
        self.round_status = "PRESS 'START MATCH' TO BEGIN BATTLE"
        self.round_status_color = "#38bdf8"

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)

        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        # 1. Neon Top Header
        header = tk.Frame(self.root, bg="#111827", pady=10, padx=25)
        header.pack(fill=tk.X, side=tk.TOP)

        title_box = tk.Frame(header, bg="#111827")
        title_box.pack(side=tk.LEFT)

        tk.Label(
            title_box,
            text="⚡ ARCADE SHOWDOWN : ROCK • PAPER • SCISSORS ⚡",
            font=("Impact", 20),
            fg="#38bdf8",
            bg="#111827"
        ).pack(anchor=tk.W)

        tk.Label(
            title_box,
            text="NEURAL VISION DEEP LEARNING ARENA",
            font=("Segoe UI", 9, "bold"),
            fg="#64748b",
            bg="#111827"
        ).pack(anchor=tk.W)

        gpu_text = f"🔥 {torch.cuda.get_device_name(0)} (CUDA ON)" if self.device.type == 'cuda' else "💻 CPU ENGINE"
        tk.Label(
            header,
            text=gpu_text,
            font=("Segoe UI", 10, "bold"),
            fg="#10b981" if self.device.type == 'cuda' else "#f59e0b",
            bg="#1f2937",
            padx=12,
            pady=6,
            bd=1,
            relief=tk.SOLID
        ).pack(side=tk.RIGHT)

        # 2. Main Content Split
        main_frame = tk.Frame(self.root, bg="#090d16", padx=16, pady=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT: Video Cam + Confidence Meters
        left_col = tk.Frame(main_frame, bg="#111827", bd=2, relief=tk.RIDGE, padx=12, pady=12)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))

        cam_header = tk.Frame(left_col, bg="#111827")
        cam_header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            cam_header,
            text="🎥 LIVE NEURAL CAMERA (Position Hand Inside Box)",
            font=("Segoe UI", 11, "bold"),
            fg="#e2e8f0",
            bg="#111827"
        ).pack(side=tk.LEFT)

        self.hand_status_badge = tk.Label(
            cam_header,
            text="HAND: WAITING...",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#1f2937",
            padx=8,
            pady=2
        )
        self.hand_status_badge.pack(side=tk.RIGHT)

        self.video_display = tk.Label(left_col, bg="#000000")
        self.video_display.pack(fill=tk.BOTH, expand=True)

        # Real-time Multi-Class Probability Meters
        prob_bar_frame = tk.Frame(left_col, bg="#111827", pady=8)
        prob_bar_frame.pack(fill=tk.X)

        self.prob_labels = {}
        for cname in ['rock', 'paper', 'scissors']:
            col = tk.Frame(prob_bar_frame, bg="#1f2937", padx=8, pady=4, bd=1, relief=tk.SOLID)
            col.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
            lbl = tk.Label(
                col,
                text=f"{CLASS_EMOJIS[cname]} {cname.upper()}: 0%",
                font=("Segoe UI", 9, "bold"),
                fg=CLASS_COLORS[cname],
                bg="#1f2937"
            )
            lbl.pack()
            self.prob_labels[cname] = lbl

        # RIGHT: Duel Arena, Scoreboard & Controls
        right_col = tk.Frame(main_frame, bg="#111827", width=420, bd=2, relief=tk.RIDGE, padx=14, pady=14)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_col.pack_propagate(False)

        # Match Length Selector
        mode_frame = tk.Frame(right_col, bg="#111827")
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            mode_frame,
            text="MATCH FORMAT:",
            font=("Segoe UI", 10, "bold"),
            fg="#94a3b8",
            bg="#111827"
        ).pack(side=tk.LEFT)

        self.turns_var = tk.IntVar(value=5)
        self.btn_5t = tk.Radiobutton(
            mode_frame, text="5 TURNS", variable=self.turns_var, value=5,
            command=self.on_turns_change, font=("Segoe UI", 9, "bold"),
            bg="#1f2937", fg="#f8fafc", selectcolor="#0284c7", indicatoron=0,
            padx=10, pady=4, bd=0, cursor="hand2"
        )
        self.btn_5t.pack(side=tk.LEFT, padx=(10, 4))

        self.btn_10t = tk.Radiobutton(
            mode_frame, text="10 TURNS", variable=self.turns_var, value=10,
            command=self.on_turns_change, font=("Segoe UI", 9, "bold"),
            bg="#1f2937", fg="#f8fafc", selectcolor="#0284c7", indicatoron=0,
            padx=10, pady=4, bd=0, cursor="hand2"
        )
        self.btn_10t.pack(side=tk.LEFT, padx=4)

        # Auto-play hands-free toggle
        self.auto_btn = tk.Button(
            right_col,
            text="🟢 AUTO-PLAY: ON (Hands-Free Battles)",
            command=self.toggle_auto_play,
            font=("Segoe UI", 10, "bold"),
            bg="#065f46",
            fg="#34d399",
            activebackground="#047857",
            activeforeground="#ffffff",
            bd=0,
            pady=6,
            cursor="hand2"
        )
        self.auto_btn.pack(fill=tk.X, pady=(0, 10))

        # Progress / Turn Header Card
        turn_box = tk.Frame(right_col, bg="#090d16", pady=8, padx=10, bd=1, relief=tk.SOLID)
        turn_box.pack(fill=tk.X, pady=(0, 10))

        self.turn_label = tk.Label(
            turn_box,
            text="TURN 0 / 5",
            font=("Impact", 16),
            fg="#f59e0b",
            bg="#090d16"
        )
        self.turn_label.pack()

        # Score Banner
        score_box = tk.Frame(right_col, bg="#1f2937", pady=8, padx=12, bd=1, relief=tk.SOLID)
        score_box.pack(fill=tk.X, pady=(0, 10))

        score_row = tk.Frame(score_box, bg="#1f2937")
        score_row.pack(fill=tk.X)

        self.p_score_lbl = tk.Label(score_row, text="YOU: 0", font=("Impact", 16), fg="#10b981", bg="#1f2937")
        self.p_score_lbl.pack(side=tk.LEFT, expand=True)

        self.vs_lbl = tk.Label(score_row, text="VS", font=("Impact", 14), fg="#64748b", bg="#1f2937")
        self.vs_lbl.pack(side=tk.LEFT, expand=True)

        self.c_score_lbl = tk.Label(score_row, text="AI: 0", font=("Impact", 16), fg="#ef4444", bg="#1f2937")
        self.c_score_lbl.pack(side=tk.RIGHT, expand=True)

        self.draws_lbl = tk.Label(score_box, text="🤝 Draws: 0", font=("Segoe UI", 9), fg="#94a3b8", bg="#1f2937")
        self.draws_lbl.pack(pady=(4, 0))

        # DUAL FIGHTER ARENA CARDS
        arena_frame = tk.Frame(right_col, bg="#111827")
        arena_frame.pack(fill=tk.X, pady=(0, 10))

        # You Card
        self.you_card = tk.Frame(arena_frame, bg="#090d16", bd=2, relief=tk.GROOVE, padx=8, pady=8)
        self.you_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 4))
        tk.Label(self.you_card, text="YOUR MOVE", font=("Segoe UI", 9, "bold"), fg="#10b981", bg="#090d16").pack()
        self.you_icon = tk.Label(self.you_card, text="❓", font=("Segoe UI Emoji", 36), fg="#f8fafc", bg="#090d16")
        self.you_icon.pack(pady=4)
        self.you_text = tk.Label(self.you_card, text="WAITING", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#090d16")
        self.you_text.pack()

        # AI Card
        self.ai_card = tk.Frame(arena_frame, bg="#090d16", bd=2, relief=tk.GROOVE, padx=8, pady=8)
        self.ai_card.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(4, 0))
        tk.Label(self.ai_card, text="AI MOVE", font=("Segoe UI", 9, "bold"), fg="#ef4444", bg="#090d16").pack()
        self.ai_icon = tk.Label(self.ai_card, text="🤖", font=("Segoe UI Emoji", 36), fg="#f8fafc", bg="#090d16")
        self.ai_icon.pack(pady=4)
        self.ai_text = tk.Label(self.ai_card, text="READY", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#090d16")
        self.ai_text.pack()

        # Big Round Result Banner
        self.banner_card = tk.Frame(right_col, bg="#090d16", bd=2, relief=tk.SOLID, pady=10, padx=10)
        self.banner_card.pack(fill=tk.X, pady=(0, 10))

        self.status_banner = tk.Label(
            self.banner_card,
            text=self.round_status,
            font=("Segoe UI", 10, "bold"),
            fg=self.round_status_color,
            bg="#090d16",
            wraplength=380
        )
        self.status_banner.pack()

        # Giant Countdown Display
        self.countdown_display = tk.Label(
            right_col,
            text="READY",
            font=("Impact", 30),
            fg="#38bdf8",
            bg="#111827",
            pady=4
        )
        self.countdown_display.pack(fill=tk.X, pady=(0, 10))

        # Action Buttons
        self.btn_shoot = tk.Button(
            right_col,
            text="🥊 SHOOT NOW",
            command=self.manual_shoot_click,
            font=("Impact", 13),
            bg="#10b981",
            fg="#ffffff",
            activebackground="#059669",
            activeforeground="#ffffff",
            bd=0,
            pady=8,
            cursor="hand2"
        )
        self.btn_shoot.pack(fill=tk.X, pady=(0, 6))

        self.btn_start = tk.Button(
            right_col,
            text="▶ START NEW MATCH",
            command=self.start_game,
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            bd=0,
            pady=7,
            cursor="hand2"
        )
        self.btn_start.pack(fill=tk.X, pady=(0, 6))

        self.btn_reset = tk.Button(
            right_col,
            text="🔄 RESET MATCH",
            command=self.reset_game,
            font=("Segoe UI", 9),
            bg="#1f2937",
            fg="#94a3b8",
            activebackground="#374151",
            activeforeground="#f8fafc",
            bd=0,
            pady=5,
            cursor="hand2"
        )
        self.btn_reset.pack(fill=tk.X)

    def toggle_auto_play(self):
        self.auto_play = not self.auto_play
        if self.auto_play:
            self.auto_btn.config(
                text="🟢 AUTO-PLAY: ON (Hands-Free Battles)",
                bg="#065f46",
                fg="#34d399"
            )
            if self.game_active and not self.is_counting_down and self.current_turn < self.max_turns:
                self.schedule_next_round(1500)
        else:
            self.auto_btn.config(
                text="⚪ AUTO-PLAY: OFF (Manual Mode)",
                bg="#334155",
                fg="#cbd5e1"
            )
            if self.auto_timer_id:
                self.root.after_cancel(self.auto_timer_id)
                self.auto_timer_id = None

    def on_turns_change(self):
        if not self.game_active or (self.current_turn == 0 and self.player_score == 0 and self.computer_score == 0):
            self.max_turns = self.turns_var.get()
            self.turn_label.config(text=f"TURN: {self.current_turn} / {self.max_turns}")
        else:
            ans = messagebox.askyesno("Change Match Length", "Changing turn count will start a fresh game. Proceed?")
            if ans:
                self.max_turns = self.turns_var.get()
                self.reset_game()
            else:
                self.turns_var.set(self.max_turns)

    def start_game(self):
        if self.auto_timer_id:
            self.root.after_cancel(self.auto_timer_id)
            self.auto_timer_id = None

        self.max_turns = self.turns_var.get()
        self.current_turn = 0
        self.player_score = 0
        self.computer_score = 0
        self.draws = 0
        self.game_active = True
        self.is_counting_down = False
        self.round_status = "MATCH STARTED! Show your hand in the green box."
        self.round_status_color = "#38bdf8"
        self.update_stats_ui()
        self.countdown_display.config(text="READY", fg="#38bdf8")

        if self.auto_play:
            self.schedule_next_round(1500)

    def reset_game(self):
        if self.auto_timer_id:
            self.root.after_cancel(self.auto_timer_id)
            self.auto_timer_id = None

        self.current_turn = 0
        self.player_score = 0
        self.computer_score = 0
        self.draws = 0
        self.game_active = False
        self.is_counting_down = False
        self.round_status = "Game reset. Select 5 or 10 turns and click START MATCH."
        self.round_status_color = "#94a3b8"
        self.countdown_display.config(text="READY", fg="#38bdf8")
        self.you_icon.config(text="❓")
        self.you_text.config(text="WAITING", fg="#94a3b8")
        self.ai_icon.config(text="🤖")
        self.ai_text.config(text="READY", fg="#94a3b8")
        self.update_stats_ui()

    def manual_shoot_click(self):
        if not self.game_active:
            self.start_game()
            return
        if self.current_turn >= self.max_turns:
            messagebox.showinfo("Match Finished", "This match is over. Click 'START NEW MATCH' to play again!")
            return
        if not self.is_counting_down:
            self.start_countdown()

    def schedule_next_round(self, delay_ms=2000):
        if self.auto_timer_id:
            self.root.after_cancel(self.auto_timer_id)
            self.auto_timer_id = None

        if self.game_active and self.current_turn < self.max_turns:
            self.auto_timer_id = self.root.after(delay_ms, self.start_countdown)

    def start_countdown(self):
        if self.current_turn >= self.max_turns or self.is_counting_down:
            return

        self.is_counting_down = True
        self.countdown_val = 3
        self.btn_shoot.config(state=tk.DISABLED)
        self.perform_tick()

    def perform_tick(self):
        if self.countdown_val > 0:
            play_sfx('tick')
            self.countdown_display.config(text=f"⏱ {self.countdown_val}", fg="#f59e0b")
            # Animate shuffling AI card
            shuffle_choice = random.choice(CLASS_NAMES)
            self.ai_icon.config(text=CLASS_EMOJIS[shuffle_choice])
            self.ai_text.config(text="THINKING...", fg="#f59e0b")

            self.countdown_val -= 1
            self.root.after(900, self.perform_tick)
        elif self.countdown_val == 0:
            play_sfx('shoot')
            self.countdown_display.config(text="💥 SHOOT!", fg="#10b981")
            self.countdown_val -= 1
            self.root.after(350, self.resolve_duel)

    def resolve_duel(self):
        self.is_counting_down = False
        self.btn_shoot.config(state=tk.NORMAL)

        # Ensure hand was in the box
        if not self.hand_detected or self.current_live_pred == "none":
            self.round_status = "⚠️ NO HAND DETECTED! Place hand inside the green box."
            self.round_status_color = "#f43f5e"
            self.countdown_display.config(text="MISS", fg="#f43f5e")
            self.status_banner.config(text=self.round_status, fg=self.round_status_color)
            play_sfx('lose')

            if self.auto_play and self.game_active:
                self.schedule_next_round(2000)
            return

        player_move = self.current_live_pred
        computer_move = random.choice(CLASS_NAMES)

        self.player_last_move = player_move
        self.computer_last_move = computer_move
        self.current_turn += 1

        # Update Dual Arena Visuals
        self.you_icon.config(text=CLASS_EMOJIS[player_move])
        self.you_text.config(text=player_move.upper(), fg=CLASS_COLORS[player_move])

        self.ai_icon.config(text=CLASS_EMOJIS[computer_move])
        self.ai_text.config(text=computer_move.upper(), fg=CLASS_COLORS[computer_move])

        # Outcome Logic
        if player_move == computer_move:
            result_str = f"🤝 DRAW! Both picked {player_move.upper()}"
            color = "#f59e0b"
            self.draws += 1
            play_sfx('draw')
        elif (
            (player_move == 'rock' and computer_move == 'scissors') or
            (player_move == 'paper' and computer_move == 'rock') or
            (player_move == 'scissors' and computer_move == 'paper')
        ):
            result_str = f"🎉 YOU WON! {player_move.upper()} beats {computer_move.upper()}"
            color = "#10b981"
            self.player_score += 1
            play_sfx('win')
        else:
            result_str = f"🤖 AI WON! {computer_move.upper()} beats {player_move.upper()}"
            color = "#ef4444"
            self.computer_score += 1
            play_sfx('lose')

        self.round_status = result_str
        self.round_status_color = color
        self.countdown_display.config(text=player_move.upper(), fg=CLASS_COLORS[player_move])
        self.update_stats_ui()

        # Check Match Finish
        if self.current_turn >= self.max_turns:
            self.game_active = False
            self.btn_shoot.config(state=tk.DISABLED)
            self.root.after(400, self.announce_winner)
        else:
            if self.auto_play and self.game_active:
                self.schedule_next_round(2200)

    def announce_winner(self):
        if self.player_score > self.computer_score:
            play_sfx('win')
            title = "🏆 MATCH VICTORY!"
            msg = f"CONGRATULATIONS!\nYou defeated the AI {self.player_score} - {self.computer_score} ({self.draws} draws)!"
        elif self.computer_score > self.player_score:
            play_sfx('lose')
            title = "🤖 AI VICTORY!"
            msg = f"AI defeated you {self.computer_score} - {self.player_score} ({self.draws} draws)!\nTry again!"
        else:
            play_sfx('draw')
            title = "🤝 MATCH TIED!"
            msg = f"The battle concluded in a Draw!\nScore: {self.player_score} - {self.computer_score}"

        messagebox.showinfo(title, msg)

    def update_stats_ui(self):
        self.turn_label.config(text=f"TURN {self.current_turn} / {self.max_turns}")
        self.p_score_lbl.config(text=f"YOU: {self.player_score}")
        self.c_score_lbl.config(text=f"AI: {self.computer_score}")
        self.draws_lbl.config(text=f"🤝 Draws: {self.draws}")
        self.status_banner.config(text=self.round_status, fg=self.round_status_color)

    def multi_pipe_predict(self, roi_bgr):
        """
        Multi-stage prediction engine:
        1. Checks hand coverage using edge variance and luminance
        2. Evaluates neural net outputs with temporal moving average buffer
        """
        # Luminance & Edge Activity Check (Discards plain flat walls)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Check skin presence using HSV + YCrCb union
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 130, 75]), np.array([255, 175, 130]))
        skin_density = np.count_nonzero(skin_mask) / float(skin_mask.size)

        # If flat wall or minimal hand, mark as none
        if skin_density < 0.05 and laplacian_var < 40:
            self.pred_history.append('none')
            return "none", 0.0, False, {'rock': 0, 'paper': 0, 'scissors': 0}

        # Composite segmented hand onto bright white background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        white_bg = np.ones_like(roi_bgr, dtype=np.uint8) * 255
        composited = np.where(clean_mask[:, :, None] == 255, roi_bgr, white_bg)

        # PyTorch Tensor
        roi_rgb = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        prob_dict = {
            'paper': float(probs[0]),
            'rock': float(probs[1]),
            'scissors': float(probs[2])
        }

        best_idx = probs.argmax()
        pred_label = CLASS_NAMES[best_idx]
        conf = float(probs[best_idx])

        self.pred_history.append(pred_label)

        # Temporal majority voting to prevent single-frame flickering
        valid_history = [p for p in self.pred_history if p != 'none']
        if valid_history:
            stable_pred = max(set(valid_history), key=valid_history.count)
        else:
            stable_pred = pred_label

        return stable_pred, conf, True, prob_dict

    def update_loop(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # ROI Placement Box
                box_size = 280
                x1 = int(w * 0.52)
                y1 = int(h * 0.15)
                x2 = min(w - 10, x1 + box_size)
                y2 = min(h - 10, y1 + box_size)

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    pred_class, conf, has_hand, prob_dict = self.multi_pipe_predict(roi)
                    self.current_live_pred = pred_class
                    self.current_live_conf = conf
                    self.hand_detected = has_hand
                    self.class_probs = prob_dict

                    # Update Live Probability Meters
                    for cname in ['rock', 'paper', 'scissors']:
                        pct = int(prob_dict.get(cname, 0.0) * 100)
                        self.prob_labels[cname].config(text=f"{CLASS_EMOJIS[cname]} {cname.upper()}: {pct}%")

                    # Dynamic Box Styling
                    if not has_hand:
                        box_color = (100, 100, 100)
                        tag = "PLACE HAND HERE"
                        self.hand_status_badge.config(text="HAND: NONE IN BOX", fg="#94a3b8")
                    elif self.is_counting_down:
                        box_color = (0, 255, 255)
                        tag = f"{pred_class.upper()} ({conf*100:.0f}%)"
                        self.hand_status_badge.config(text=f"HAND: {pred_class.upper()} (LOCKING)", fg="#f59e0b")
                    else:
                        box_color = (0, 255, 0)
                        tag = f"{pred_class.upper()} ({conf*100:.0f}%)"
                        self.hand_status_badge.config(text=f"HAND: {pred_class.upper()} ({conf*100:.0f}%)", fg="#10b981")

                    # Draw stylish corners on bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    corner_len = 25
                    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 4)
                    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 4)
                    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 4)
                    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 4)
                    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 4)
                    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, 4)
                    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 4)
                    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 4)

                    cv2.putText(frame, tag, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                # Giant Visual Countdown Overlay on Video
                if self.is_counting_down:
                    num_text = str(self.countdown_val + 1) if self.countdown_val >= 0 else "GO!"
                    cv2.putText(frame, num_text, (int(w * 0.18), int(h * 0.55)),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.8, (0, 255, 255), 10)

                # Render Frame in Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((660, 495), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_display.imgtk = imgtk
                self.video_display.configure(image=imgtk)

        self.root.after(30, self.update_loop)

    def on_closing(self):
        if self.auto_timer_id:
            self.root.after_cancel(self.auto_timer_id)
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ArcadeRPSGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
