"""Local webcam face enrollment and recognition. Run with python app.py."""

import argparse
from queue import Empty, Full, Queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from face_core import DEFAULT_THRESHOLD, Enrollment, FaceEngine, FaceStore, REGISTRATION_SAMPLES, open_camera
from models import ensure_models


class CameraWorker(threading.Thread):
    def __init__(self, index, frames, events):
        super().__init__(daemon=True)
        self.index, self.frames, self.events = index, frames, events
        self.stop_event = threading.Event()

    def run(self):
        capture = None
        try:
            paths = ensure_models(lambda text: self.events.put(("status", text)))
            if self.stop_event.is_set():
                return
            engine = FaceEngine(*paths)
            self.events.put(("status", f"카메라 {self.index} 연결 중…"))
            capture = open_camera(self.index)
            failures = 0
            while not self.stop_event.is_set():
                started = time.monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError("카메라 영상이 끊겼습니다. 다른 앱의 카메라 사용 여부를 확인하세요.")
                    self.stop_event.wait(0.1)
                    continue
                failures = 0
                # Keep aspect ratio; cap processing resolution for CPU inference.
                height, width = frame.shape[:2]
                scale = min(640 / width, 480 / height, 1.0)
                if scale < 1:
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                # Recognition and display use the same mirrored frame.
                frame = cv2.flip(frame, 1)
                faces = engine.analyze(frame)
                packet = (frame, faces, time.monotonic(), 1 / max(time.monotonic() - started, 0.001))
                try:
                    self.frames.put_nowait(packet)
                except Full:
                    try:
                        self.frames.get_nowait()
                    except Empty:
                        pass
                    self.frames.put_nowait(packet)
                self.stop_event.wait(max(0, 1 / 20 - (time.monotonic() - started)))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            if capture is not None:
                capture.release()
            self.events.put(("stopped", ""))


class FaceApp:
    def __init__(self, root, camera=0, store=None):
        self.root = root
        self.store = store if store is not None else FaceStore()
        self.worker = None
        self.frames, self.events = Queue(maxsize=1), Queue()
        self.enrollment = None
        self.last_frame_at = 0
        self.photo = None
        self.camera = tk.StringVar(value=str(camera))
        self.name = tk.StringVar()
        self.threshold = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.status = tk.StringVar(value="카메라 시작을 누르면 미리보기가 켜집니다.")
        self.result = tk.StringVar(value="인식 대기 중")
        self.threshold_text = tk.StringVar(value=f"{DEFAULT_THRESHOLD:.2f}")
        self._build()
        self._refresh_people()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(40, self._poll)

    def _build(self):
        self.root.title("내 얼굴 인식 · Local Face")
        self.root.geometry("1020x760")
        self.root.minsize(980, 740)
        self.root.configure(bg="#f4f6fa")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6fa")
        style.configure("TLabel", background="#f4f6fa", font=("맑은 고딕", 10))
        style.configure("TButton", font=("맑은 고딕", 10), padding=5)
        style.configure("Title.TLabel", font=("맑은 고딕", 22, "bold"))
        style.configure("Result.TLabel", font=("맑은 고딕", 13, "bold"))

        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="내 얼굴 인식", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="얼굴 등록 → 임베딩 생성 → 실시간 인식 · 노트북 안에서 처리합니다.").pack(anchor="w", pady=(4, 16))
        content = ttk.Frame(outer)
        content.pack(fill="both", expand=True)
        left, right = ttk.Frame(content), ttk.Frame(content, padding=(20, 0, 0, 0))
        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="y")

        self.canvas = tk.Canvas(left, width=640, height=480, bg="#141d2d", highlightthickness=0)
        self.canvas.pack(anchor="n")
        self._placeholder()
        ttk.Label(left, textvariable=self.result, style="Result.TLabel", wraplength=640).pack(anchor="w", pady=(12, 4))
        ttk.Label(left, textvariable=self.status, wraplength=640).pack(anchor="w")

        ttk.Label(right, text="1. 카메라 연결", style="Result.TLabel").pack(anchor="w")
        row = ttk.Frame(right)
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="카메라 번호").pack(side="left")
        self.camera_input = ttk.Spinbox(row, from_=0, to=9, width=5, textvariable=self.camera)
        self.camera_input.pack(side="right")
        self.start_button = ttk.Button(right, text="카메라 시작", command=self.start_camera)
        self.start_button.pack(fill="x")
        self.stop_button = ttk.Button(right, text="카메라 중지", command=self.stop_camera, state="disabled")
        self.stop_button.pack(fill="x", pady=(6, 12))

        ttk.Label(right, text="2. 내 얼굴 등록", style="Result.TLabel").pack(anchor="w")
        self.name_input = ttk.Entry(right, textvariable=self.name, width=25, font=("맑은 고딕", 11))
        self.name_input.pack(fill="x", pady=8)
        self.register_button = ttk.Button(right, text="이름 입력 후 얼굴 등록", command=self.register, state="disabled")
        self.register_button.pack(fill="x")
        self.cancel_button = ttk.Button(right, text="등록 취소", command=self.cancel_enrollment, state="disabled")
        self.cancel_button.pack(fill="x", pady=4)
        self.progress = ttk.Progressbar(right, maximum=REGISTRATION_SAMPLES)
        self.progress.pack(fill="x", pady=(0, 6))
        ttk.Label(right, text="얼굴 한 명 · 선명한 샘플 15개\n고개를 조금씩 움직여주세요.").pack(anchor="w")

        ttk.Label(right, text="등록된 얼굴", style="Result.TLabel").pack(anchor="w", pady=(12, 6))
        self.people = tk.Listbox(right, height=3, font=("맑은 고딕", 11), exportselection=False)
        self.people.pack(fill="x")
        self.delete_button = ttk.Button(right, text="선택한 등록 삭제", command=self.delete_person)
        self.delete_button.pack(fill="x", pady=6)

        ttk.Label(right, text="인식 기준 · 높을수록 엄격").pack(anchor="w", pady=(6, 0))
        ttk.Scale(right, from_=0.30, to=0.80, variable=self.threshold,
                  command=lambda value: self.threshold_text.set(f"{float(value):.2f}")).pack(fill="x")
        ttk.Label(right, textvariable=self.threshold_text).pack(anchor="e")
        ttk.Label(outer, text="사진·영상은 저장하지 않습니다. 등록한 임베딩은 data/embeddings.json에 저장됩니다.").pack(anchor="w", pady=(16, 0))

    def _placeholder(self, text="카메라 미리보기\n오른쪽의 ‘카메라 시작’을 눌러주세요."):
        self.canvas.delete("all")
        self.canvas.create_text(320, 240, text=text, fill="#cbd5e1", font=("맑은 고딕", 15), justify="center", width=580)
        self.photo = None

    def _refresh_people(self):
        self.people.delete(0, "end")
        for name in sorted(self.store.people):
            self.people.insert("end", name)

    def start_camera(self):
        if self.worker is not None:
            return
        try:
            index = int(self.camera.get())
            if not 0 <= index <= 9:
                raise ValueError
        except ValueError:
            messagebox.showerror("카메라 번호", "0~9 사이의 번호를 입력하세요.")
            return
        self.frames, self.events = Queue(maxsize=1), Queue()
        self.last_frame_at = 0
        self.start_button.configure(state="disabled")
        self.camera_input.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("모델 준비 중… 첫 실행에는 약 37 MB 다운로드가 필요합니다.")
        self.worker = CameraWorker(index, self.frames, self.events)
        self.worker.start()

    def stop_camera(self):
        if self.worker is not None:
            self.worker.stop_event.set()
            self.stop_button.configure(state="disabled")
            self.register_button.configure(state="disabled")
            self.cancel_enrollment()
            self.status.set("카메라를 종료하고 있습니다…")
            self.result.set("인식 중지")
            self._placeholder("카메라 중지 중…")

    def register(self):
        if self.worker is None or self.worker.stop_event.is_set() or time.monotonic() - self.last_frame_at > 2:
            messagebox.showinfo("얼굴 등록", "먼저 카메라 영상을 확인해주세요.")
            return
        name = self.name.get().strip()
        if not name or len(name) > 40:
            messagebox.showinfo("얼굴 등록", "이름을 1~40자로 입력하세요.")
            return
        if name in self.store.people and not messagebox.askyesno("다시 등록", f"‘{name}’의 기존 임베딩을 새로 등록할까요?"):
            return
        self.enrollment = Enrollment(name)
        self.progress["value"] = 0
        self.register_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.name_input.configure(state="disabled")
        self.delete_button.configure(state="disabled")
        self.status.set("정면을 바라보며 얼굴을 화면에 맞춰주세요.")

    def cancel_enrollment(self):
        self.enrollment = None
        self.progress["value"] = 0
        self.cancel_button.configure(state="disabled")
        self.name_input.configure(state="normal")
        self.delete_button.configure(state="normal")
        if self.worker and not self.worker.stop_event.is_set():
            self.register_button.configure(state="normal")

    def delete_person(self):
        selected = self.people.curselection()
        if not selected:
            return
        name = self.people.get(selected[0])
        if messagebox.askyesno("등록 삭제", f"‘{name}’의 저장된 임베딩을 삭제할까요?"):
            try:
                self.store.delete(name)
            except OSError as exc:
                messagebox.showerror("삭제 실패", str(exc))
                return
            self._refresh_people()
            self.status.set(f"‘{name}’의 등록을 삭제했습니다.")

    def _handle_frame(self, frame, faces, captured_at, fps):
        self.last_frame_at = captured_at
        if self.enrollment:
            status = self.enrollment.accept(faces, captured_at)
            self.status.set(status)
            self.progress["value"] = len(self.enrollment.samples)
            if captured_at - self.enrollment.started > 60:
                self.cancel_enrollment()
                self.status.set("등록 시간이 초과되었습니다. 조명과 위치를 조절한 후 다시 등록하세요.")
            elif len(self.enrollment.samples) == REGISTRATION_SAMPLES:
                name = self.enrollment.name
                try:
                    self.store.enroll(name, self.enrollment.samples)
                except (ValueError, OSError) as exc:
                    self.cancel_enrollment()
                    messagebox.showerror("등록 실패", str(exc))
                else:
                    self.cancel_enrollment()
                    self._refresh_people()
                    self.status.set(f"‘{name}’ 등록 완료 · 128차원 임베딩을 저장했습니다.")
        else:
            self.register_button.configure(state="normal")

        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        height, width = frame.shape[:2]
        offset_x, offset_y = (640 - width) // 2, (480 - height) // 2
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.photo)
        labels = []
        for face in faces:
            name, score = self.store.match(face.embedding, self.threshold.get())
            color = "#4ade80" if name else "#fbbf24"
            label = name or "미등록"
            if score is not None:
                label += f" · {score:.3f}"
            labels.append(label)
            x, y, w, h = face.box
            x, y = x + offset_x, y + offset_y
            self.canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=3)
            text = self.canvas.create_text(max(4, x), max(4, y - 26), text=label, anchor="nw",
                                           fill=color, font=("맑은 고딕", 11, "bold"))
            background = self.canvas.create_rectangle(self.canvas.bbox(text), fill="#141d2d", outline="")
            self.canvas.tag_lower(background, text)
        self.result.set(" / ".join(labels) if labels else "얼굴이 감지되지 않았습니다.")
        self.canvas.create_text(10, 468, anchor="sw", text=f"{fps:.0f} FPS · 유사도 기준 {self.threshold.get():.2f}",
                                fill="white", font=("맑은 고딕", 10))

    def _poll(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except Empty:
                break
            if kind == "status":
                self.status.set(value)
            elif kind == "error":
                self.status.set(value)
                self._placeholder("카메라를 연결하지 못했습니다.\n상태 메시지를 확인한 후 다시 시작하세요.")
                messagebox.showerror("카메라 연결 오류", value)
            elif kind == "stopped":
                self.worker = None
                self.cancel_enrollment()
                self.start_button.configure(state="normal")
                self.camera_input.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.register_button.configure(state="disabled")
                self.result.set("인식 대기 중")
                self._placeholder()
        if self.worker and not self.worker.stop_event.is_set():
            try:
                packet = self.frames.get_nowait()
            except Empty:
                pass
            else:
                self._handle_frame(*packet)
            if self.last_frame_at and time.monotonic() - self.last_frame_at > 2:
                self.register_button.configure(state="disabled")
                self.result.set("카메라 영상을 기다리고 있습니다…")
                self._placeholder("카메라 영상을 기다리고 있습니다…")
        self.root.after(40, self._poll)

    def close(self):
        if self.worker:
            self.worker.stop_event.set()
            # Keep the process alive until the capture is released.
            self.root.withdraw()
            self._wait_for_close()
        else:
            self.root.destroy()

    def _wait_for_close(self):
        if self.worker and self.worker.is_alive():
            self.root.after(50, self._wait_for_close)
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Local webcam face enrollment and recognition")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    args = parser.parse_args()
    root = tk.Tk()
    try:
        FaceApp(root, args.camera)
    except (ValueError, OSError) as exc:
        root.withdraw()
        messagebox.showerror("등록 데이터 오류", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
