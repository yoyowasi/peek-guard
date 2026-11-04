"""노트북 웹캠으로 화면 쪽을 보는 얼굴 수를 세어 작업 표시줄 트레이 아이콘에 표시한다.

초록 1 = 1명 보는 중(보통 나), 빨강 2+ = 다른 사람도 보는 중(Windows 알림), 회색 0 = 아무도 안 봄,
회색 II = 일시정지(카메라 꺼짐), 검정 ! = 카메라를 쓸 수 없음.
창 없이 백그라운드에서 돈다. 카메라 화면은 우클릭 메뉴에서 '카메라 화면 보기'를 직접 골랐을 때만 뜬다.
'감시 간격'을 고르면 그 간격마다 카메라를 잠깐 켜서 확인하고 끈다(LED가 켜져 있는 시간이 줄어든다).
감도·간격 등은 settings.json에서 바꾸며, 저장하면 실행 중에도 바로 반영된다(README 참고).
"""
import ctypes
import json
import math
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import pystray
from PIL import Image, ImageDraw, ImageFont

CHECK_INTERVAL = 0.5     # 초, 계속 감시할 때 감지 주기
INTERVAL_CHOICES = (0, 5, 10, 30)
SCAN_FRAMES = 3          # 간격 감시에서 카메라를 켤 때마다 볼 프레임 수
SCAN_GAP = 0.15          # 초, 그 프레임 사이 간격

# settings.json에 없는 항목은 이 값을 쓴다. 각 항목의 뜻은 README에 있다.
DEFAULT_SETTINGS = {
    "camera_index": 0,
    "scan_every_seconds": 10,
    "max_yaw_ratio": 1.0,
    "min_face_ratio": 0.06,
    "score_threshold": 0.8,
    "alert_frames": 2,
    "notify": True,
    "notify_cooldown_seconds": 30,
}

HERE = Path(__file__).parent
SETTINGS_PATH = HERE / "settings.json"
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
MODEL_PATH = HERE / "face_detection_yunet_2023mar.onnx"
PREVIEW_WINDOW = "Peek Guard preview (Esc)"  # OpenCV 창 제목은 한글이 깨진다

GREEN, RED, GRAY, BLACK = (34, 160, 80), (215, 40, 40), (130, 130, 130), (30, 30, 30)
try:
    FONT = ImageFont.truetype("arialbd.ttf", 40)
except OSError:
    FONT = ImageFont.load_default()


def make_icon(color, text):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((2, 2, 62, 62), fill=color)
    draw.text((32, 33), text, fill="white", font=FONT, anchor="mm")
    return img


def load_settings():
    """settings.json을 읽어 기본값에 덮어쓴다. 파일이 없으면 기본값으로 만든다. 잘못된 내용이면 ValueError."""
    if not SETTINGS_PATH.exists():
        save_settings(DEFAULT_SETTINGS)
    user = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))  # 메모장이 붙이는 BOM도 읽는다
    if not isinstance(user, dict):
        raise ValueError("{ ... } 형태의 JSON이어야 합니다")
    for key, value in user.items():
        if key not in DEFAULT_SETTINGS:
            raise ValueError(f"모르는 항목: {key}")
        if isinstance(DEFAULT_SETTINGS[key], bool):
            valid = isinstance(value, bool)
        else:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
        if not valid:
            raise ValueError(f"{key} 값이 잘못됨: {value!r}")
    return {**DEFAULT_SETTINGS, **user}


def save_settings(settings):
    SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_model():
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_detector():
    return cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320), DEFAULT_SETTINGS["score_threshold"])


def open_camera(index):
    # 해상도를 지정하면 여는 데 1초 넘게 더 걸려 LED가 그만큼 오래 켜진다. 기본 해상도를 그대로 쓴다.
    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def detect(detector, frame, cfg):
    """(화면 쪽을 보는 얼굴, 그 밖의 얼굴) 목록을 돌려준다. 얼굴 = YuNet 결과 한 줄."""
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    detector.setScoreThreshold(cfg["score_threshold"])
    _, faces = detector.detect(frame)
    viewers, others = [], []
    for face in faces if faces is not None else []:
        if face[2] < w * cfg["min_face_ratio"]:
            continue
        # 카메라가 화면 위에 있으니, 코가 두 눈 가운데에 가까우면(카메라 쪽 얼굴) 화면을 보는 것으로 본다
        (viewers if yaw_ratio(face) <= cfg["max_yaw_ratio"] else others).append(face)
    return viewers, others


def yaw_ratio(face):
    """코가 두 눈 가운데에서 벗어난 정도(눈 간격 대비). 정면이면 0, 고개를 돌릴수록 커진다."""
    rex, rey, lex, ley, nose_x = face[4:9]
    eye_dist = math.hypot(lex - rex, ley - rey)
    return abs(nose_x - (rex + lex) / 2) / eye_dist if eye_dist > 1 else math.inf


class PeekGuard:
    def __init__(self):
        self.cfg = dict(DEFAULT_SETTINGS)
        self.settings_error = None
        self.paused = False
        self.preview = False
        self.status = "시작 중"
        self._shown = None
        self._settings_mtime = None
        self._lock = threading.Lock()   # 메뉴(저장)와 감시 스레드(다시 읽기)가 settings.json을 함께 쓴다
        self._stop = threading.Event()
        self._wake = threading.Event()  # 메뉴를 바꾸면 쉬던 감시 스레드를 바로 깨운다
        self._worker = threading.Thread(target=self._run, daemon=True)
        self.icon = pystray.Icon(
            "peek_guard", make_icon(GRAY, ""), "Peek Guard",
            menu=pystray.Menu(
                pystray.MenuItem(lambda item: self.status, None, enabled=False),
                pystray.MenuItem(lambda item: f"설정 파일 오류: {self.settings_error}", None, enabled=False,
                                 visible=lambda item: bool(self.settings_error)),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("카메라 화면 보기", self._toggle("preview"), checked=lambda i: self.preview),
                pystray.MenuItem("감시 간격", pystray.Menu(*map(self._interval_item, INTERVAL_CHOICES))),
                pystray.MenuItem("알림", self._toggle_setting("notify"), checked=lambda i: self.cfg["notify"]),
                pystray.MenuItem("설정 파일 열기", self._open_settings),
                pystray.MenuItem("일시정지", self._toggle("paused"), checked=lambda i: self.paused),
                pystray.MenuItem("종료", self._quit),
            ),
        )

    def run(self):
        def setup(icon):
            icon.visible = True
            self._worker.start()

        self.icon.run(setup=setup)
        self._worker.join(timeout=3)

    def _toggle(self, name):
        def action(icon, item):
            setattr(self, name, not getattr(self, name))
            self._wake.set()
        return action

    def _toggle_setting(self, key):
        def action(icon, item):
            self._set(key, not self.cfg[key])
        return action

    def _interval_item(self, seconds):
        def action(icon, item):
            self._set("scan_every_seconds", seconds)
        label = "계속 (LED 계속 켜짐)" if seconds == 0 else f"{seconds}초마다"
        return pystray.MenuItem(label, action, checked=lambda i: self.cfg["scan_every_seconds"] == seconds,
                                radio=True)

    def _set(self, key, value):
        """메뉴에서 바꾼 값을 적용하고 settings.json에도 저장한다. 파일이 잘못돼 있으면 사용자가 고칠 수 있게 덮어쓰지 않는다."""
        with self._lock:
            self.cfg = {**self.cfg, key: value}
            if not self.settings_error:
                save_settings(self.cfg)
                self._settings_mtime = SETTINGS_PATH.stat().st_mtime
        self._wake.set()

    def _reload_if_changed(self):
        """settings.json이 바뀌었으면(또는 처음이면) 다시 읽는다. 읽었으면 True."""
        with self._lock:
            mtime = SETTINGS_PATH.stat().st_mtime if SETTINGS_PATH.exists() else None
            if mtime is not None and mtime == self._settings_mtime:
                return False
            try:
                self.cfg = load_settings()
                self.settings_error = None
            except (OSError, ValueError) as e:  # 잘못 고친 파일은 알리고 이전 설정으로 계속 돈다
                self.settings_error = str(e)
                self.icon.notify(f"settings.json을 읽지 못해 이전 설정으로 동작합니다.\n{e}", "Peek Guard")
            self._settings_mtime = SETTINGS_PATH.stat().st_mtime if SETTINGS_PATH.exists() else None
        self.icon.update_menu()
        return True

    def _open_settings(self, icon, item):
        if not SETTINGS_PATH.exists():
            save_settings(self.cfg)
        subprocess.Popen(["notepad.exe", str(SETTINGS_PATH)])

    def _quit(self, icon, item):
        self._stop.set()
        self._wake.set()
        icon.stop()

    def _sleep(self, seconds):
        self._wake.wait(seconds)
        self._wake.clear()

    def _show(self, color, text, status):
        if self._shown == (color, text, status):
            return
        self._shown = (color, text, status)
        self.status = status
        self.icon.icon = make_icon(color, text)
        self.icon.title = f"Peek Guard - {status}"
        self.icon.update_menu()

    def _wait(self, seconds, preview_open):
        # 미리보기 창이 떠 있으면 waitKey로 기다려야 창이 응답한다
        if not preview_open:
            self._sleep(seconds)
            return
        if cv2.waitKey(int(seconds * 1000)) == 27 or \
                cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self.preview = False
            self.icon.update_menu()

    def _run(self):
        detector = create_detector()
        cap = None
        frames_left = 0
        preview_open = False
        hits = misses = 0
        alert = False
        last_notify = 0.0
        try:
            while not self._stop.is_set():
                if self._reload_if_changed() and cap is not None:
                    cap.release()  # 카메라 번호가 바뀌었을 수 있으니 다시 연다
                    cap = None
                cfg = self.cfg
                alert_frames = max(1, cfg["alert_frames"])

                if preview_open and not self.preview:
                    try:
                        cv2.destroyWindow(PREVIEW_WINDOW)
                    except cv2.error:  # X 버튼으로 닫은 창은 이미 없어서 다시 닫으면 오류가 난다
                        pass
                    preview_open = False
                scan_every = 0 if self.preview else cfg["scan_every_seconds"]  # 미리보기 중에는 계속 감시

                if self.paused:
                    if cap is not None:
                        cap.release()
                        cap = None
                    self._show(GRAY, "II", "일시정지 (카메라 꺼짐)")
                    self._sleep(None)  # 메뉴를 바꿀 때까지 쉰다
                    continue

                if cap is None:
                    cap = open_camera(cfg["camera_index"])
                    frames_left = SCAN_FRAMES
                ok, frame = False, None
                if cap is not None and cap.grab():  # 버퍼에 쌓인 지난 프레임 하나를 버린다
                    ok, frame = cap.read()
                if not ok:
                    if cap is not None:
                        cap.release()
                        cap = None
                    self._show(BLACK, "!", "카메라를 쓸 수 없음 (다른 앱이 사용 중일 수 있음)")
                    self._sleep(3)
                    continue

                viewers, others = detect(detector, frame, cfg)
                n = len(viewers)
                if n >= 2:
                    hits, misses = hits + 1, 0
                else:
                    hits, misses = 0, misses + 1

                if not alert and hits >= alert_frames:
                    alert = True
                    if cfg["notify"] and time.monotonic() - last_notify > cfg["notify_cooldown_seconds"]:
                        self.icon.notify(f"{n}명이 화면을 보고 있습니다.", "Peek Guard")
                        last_notify = time.monotonic()
                elif alert and misses >= alert_frames:
                    alert = False

                if alert:
                    color, text, status = RED, str(max(n, 2)), f"다른 사람도 보는 중 ({n}명)"
                elif n == 1:
                    color, text, status = GREEN, "1", "1명 보는 중"  # 누가 나인지는 모르므로 '나만'이라 하지 않는다
                else:
                    color, text, status = GRAY, str(n), "화면을 보는 사람 없음" if n == 0 else f"{n}명 보는 중"
                self._show(color, text, status + (f" · {scan_every:g}초마다 확인" if scan_every else ""))

                if self.preview:
                    for faces, color in ((viewers, (0, 200, 0)), (others, (160, 160, 160))):
                        for face in faces:
                            x, y, fw, fh = map(int, face[:4])
                            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
                            # 방향 수치: max_yaw_ratio 이하면 '보는 중'(초록). 설정값을 맞출 때 참고한다
                            cv2.putText(frame, f"{yaw_ratio(face):.2f}", (x, max(y - 8, 16)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.imshow(PREVIEW_WINDOW, frame)
                    preview_open = True

                frames_left -= 1
                if scan_every and frames_left <= 0:
                    cap.release()  # 카메라(LED)를 끄고 다음 확인까지 쉰다
                    cap = None
                    self._sleep(scan_every)
                else:
                    self._wait(SCAN_GAP if scan_every else CHECK_INTERVAL, preview_open)
        except Exception as e:  # 감시가 멈춘 채 아이콘만 남지 않도록 알린다
            self._show(BLACK, "!", f"오류로 감시 멈춤 ({type(e).__name__}) - 다시 실행하세요")
            raise
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()


def main():
    try:
        ensure_model()
        PeekGuard().run()
    except Exception as e:  # pythonw로 실행하면 콘솔이 없으니 메시지 상자로 알린다
        ctypes.windll.user32.MessageBoxW(0, f"{type(e).__name__}: {e}", "Peek Guard 오류", 0x10)
        raise


if __name__ == "__main__":
    main()
