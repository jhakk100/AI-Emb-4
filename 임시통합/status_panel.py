"""Temporary on-screen RGB LED and per-stage state indicators."""

import time
import tkinter as tk
from tkinter import ttk

COLORS = {"off": "#64748b", "white": "#ffffff", "yellow": "#facc15",
          "blue": "#38bdf8", "green": "#4ade80", "red": "#fb7185", "purple": "#c084fc"}


class StatusPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.color, self.blink = "off", False
        top = ttk.Frame(self)
        top.pack(fill="x")
        self.led = tk.Canvas(top, width=36, height=36, bg="#f4f6fa", highlightthickness=0)
        self.led.pack(side="left")
        self.bulb = self.led.create_oval(6, 6, 30, 30, fill=COLORS["off"], outline="#334155", width=2)
        self.led_text = tk.StringVar(value="임시 LED · 카메라 꺼짐")
        ttk.Label(top, textvariable=self.led_text, font=("맑은 고딕", 11, "bold")).pack(side="left", padx=6)
        self.phase = tk.StringVar(value="현재 단계: 카메라 시작 대기")
        ttk.Label(self, textvariable=self.phase).pack(anchor="w", pady=(0, 4))
        row = ttk.Frame(self)
        row.pack(fill="x")
        self.details, self.stage_labels = [], []
        for index, title in enumerate(("1 · 대기 조건", "2 · 검출 / Crop", "3 · 임베딩", "4 · DB / 인증")):
            card = ttk.Frame(row, padding=(7, 4))
            card.grid(row=0, column=index, sticky="nsew", padx=(0, 4))
            row.columnconfigure(index, weight=1, uniform="stage")
            ttk.Label(card, text=title, font=("맑은 고딕", 10, "bold")).pack(anchor="w")
            detail = tk.StringVar(value="대기")
            label = ttk.Label(card, textvariable=detail, wraplength=140, foreground="#64748b")
            label.pack(anchor="w")
            self.details.append(detail)
            self.stage_labels.append(label)

    def set_led(self, color, text, blink=False):
        self.color, self.blink = color, blink
        self.led_text.set(f"임시 LED · {text}")
        self.tick()

    def set_stage(self, stage, detail, color="off"):
        self.details[stage - 1].set(detail)
        text_colors = {"off": "#64748b", "white": "#64748b", "yellow": "#a16207",
                       "blue": "#0369a1", "green": "#15803d", "red": "#be123c", "purple": "#7e22ce"}
        self.stage_labels[stage - 1].configure(foreground=text_colors[color])

    def idle(self, text="카메라 시작 대기"):
        self.phase.set(f"현재 단계: {text}")
        self.set_led("off", text)
        for stage in range(1, 5):
            self.set_stage(stage, "대기")

    def tick(self):
        lit = not self.blink or int(time.monotonic() * 2) % 2 == 0
        self.led.itemconfigure(self.bulb, fill=COLORS[self.color] if lit else "#334155")
