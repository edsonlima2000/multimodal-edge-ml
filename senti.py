from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import queue
import re
import threading
import time
import unicodedata

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sounddevice as sd
from tensorflow.keras.models import load_model
from vosk import KaldiRecognizer, Model


FACE_MODEL_PATH = Path("models/blaze_face_short_range.tflite")
EMOTION_MODEL_PATH = Path("models/fer2013_mini_XCEPTION.102-0.66.hdf5")
ASR_MODEL_PATH = Path("models/vosk-model-small-pt-0.3")
EMOJI_FONT_PATH = Path(r"C:\Windows\Fonts\seguiemj.ttf")
UI_FONT_PATH = Path(r"C:\Windows\Fonts\segoeui.ttf")

# MiniXception treinado no dataset FER-2013.
EMOTION_LABELS = {
    0: "\U0001F620 Raiva (Angry)",
    1: "\U0001F922 Nojo (Disgust)",
    2: "\U0001F628 Medo (Fear)",
    3: "\U0001F60A Feliz (Happy)",
    4: "\U0001F622 Triste (Sad)",
    5: "\U0001F632 Surpreso (Surprise)",
    6: "\U0001F610 Neutro (Neutral)",
}

EMOTION_COLORS = {
    0: (60, 70, 220),
    1: (40, 150, 40),
    2: (180, 90, 180),
    3: (0, 190, 255),
    4: (220, 120, 60),
    5: (0, 220, 220),
    6: (160, 160, 160),
}

VIDEO_SENTIMENTS = {
    0: "Negativo",
    1: "Negativo",
    2: "Negativo",
    3: "Positivo",
    4: "Negativo",
    5: "Neutro",
    6: "Neutro",
}

SENTIMENT_COLORS = {
    "Positivo": (50, 170, 90),
    "Negativo": (210, 70, 70),
    "Neutro": (120, 120, 120),
    "Indisponivel": (120, 120, 120),
}

POSITIVE_HINTS = {
    "adorei": 3,
    "adoro": 3,
    "amei": 3,
    "amo": 3,
    "bom": 1,
    "boa": 1,
    "excelente": 2,
    "feliz": 2,
    "gostei": 2,
    "gostei muito": 3,
    "legal": 1,
    "maravilhoso": 2,
    "maravilhosa": 2,
    "muito bom": 2,
    "muito boa": 2,
    "otimo": 2,
    "otima": 2,
    "perfeito": 2,
    "perfeita": 2,
    "satisfeito": 2,
    "satisfeita": 2,
}

NEGATIVE_HINTS = {
    "demorado": 1,
    "demora": 1,
    "detestei": 3,
    "detesto": 3,
    "horrivel": 2,
    "insatisfeito": 2,
    "insatisfeita": 2,
    "lento": 1,
    "medo": 1,
    "nao gostei": 3,
    "nojo": 2,
    "odeiei": 3,
    "odeio": 3,
    "odiei": 3,
    "pessimo": 2,
    "pessima": 2,
    "problema": 1,
    "raiva": 2,
    "ruim": 2,
    "terrivel": 2,
    "triste": 2,
}


@dataclass
class AudioState:
    text: str = "Aguardando audio..."
    sentiment_label: str = "Indisponivel"
    sentiment_color: tuple[int, int, int] = SENTIMENT_COLORS["Indisponivel"]
    status: str = "Audio nao iniciado"


class AudioTranscriber:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.state = AudioState()
        self.partial_text = ""
        self.final_segments: deque[str] = deque(maxlen=3)
        self.processing_thread: threading.Thread | None = None
        self.stream: sd.RawInputStream | None = None
        self.recognizer: KaldiRecognizer | None = None
        self.model: Model | None = None
        self.sample_rate = 16000

    def start(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo de ASR nao encontrado: {self.model_path}")

        device_info = sd.query_devices(kind="input")
        self.sample_rate = int(device_info["default_samplerate"])
        self.model = Model(str(self.model_path))
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(False)

        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        )
        self.stream.start()

        with self.state_lock:
            self.state.status = "Microfone ativo"

        self.processing_thread = threading.Thread(target=self._run, daemon=True)
        self.processing_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=1.0)
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        with self.state_lock:
            self.state.status = "Microfone encerrado"

    def get_state(self) -> AudioState:
        with self.state_lock:
            return AudioState(
                text=self.state.text,
                sentiment_label=self.state.sentiment_label,
                sentiment_color=self.state.sentiment_color,
                status=self.state.status,
            )

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            with self.state_lock:
                self.state.status = f"Audio com aviso: {status}"
        self.audio_queue.put(bytes(indata))

    def _run(self) -> None:
        assert self.recognizer is not None

        while not self.stop_event.is_set():
            try:
                data = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                transcript = result.get("text", "").strip()
                if transcript:
                    self.final_segments.append(transcript)
                self.partial_text = ""
            else:
                partial = json.loads(self.recognizer.PartialResult())
                self.partial_text = partial.get("partial", "").strip()

            combined_text = " ".join(self.final_segments).strip()
            if self.partial_text:
                combined_text = f"{combined_text} {self.partial_text}".strip()

            sentiment_label, sentiment_color = classify_text_sentiment(combined_text)
            display_text = combined_text if combined_text else "Aguardando audio..."

            with self.state_lock:
                self.state.text = display_text
                self.state.sentiment_label = sentiment_label
                self.state.sentiment_color = sentiment_color


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized.lower().strip()


def classify_text_sentiment(text: str) -> tuple[str, tuple[int, int, int]]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return "Indisponivel", SENTIMENT_COLORS["Indisponivel"]

    def score_hints(hints: dict[str, int]) -> int:
        score = 0
        for hint, weight in hints.items():
            pattern = rf"(?<!\w){re.escape(hint)}(?!\w)"
            matches = re.findall(pattern, normalized_text)
            score += len(matches) * weight
        return score

    positive_score = score_hints(POSITIVE_HINTS)
    negative_score = score_hints(NEGATIVE_HINTS)

    if positive_score > negative_score:
        return "Positivo", SENTIMENT_COLORS["Positivo"]
    if negative_score > positive_score:
        return "Negativo", SENTIMENT_COLORS["Negativo"]
    return "Neutro", SENTIMENT_COLORS["Neutro"]


def preprocess_face(gray_face: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    resized_face = cv2.resize(gray_face, input_size)
    normalized_face = resized_face.astype("float32") / 255.0
    normalized_face = (normalized_face - 0.5) * 2.0
    normalized_face = np.expand_dims(normalized_face, axis=(0, -1))
    return normalized_face


def clamp_bbox(
    bbox: object,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, int(bbox.origin_x))
    y1 = max(0, int(bbox.origin_y))
    x2 = min(frame_width, x1 + int(bbox.width))
    y2 = min(frame_height, y1 + int(bbox.height))
    return x1, y1, x2, y2


def load_font(size: int, emoji: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = EMOJI_FONT_PATH if emoji else UI_FONT_PATH
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = load_font(size=28, emoji=True)
    x, y = origin
    left, top, right, bottom = draw.textbbox((x, y), label, font=font)
    padding = 8
    box = (
        left - padding,
        top - padding,
        right + padding,
        bottom + padding,
    )
    draw.rounded_rectangle(box, radius=10, fill=(*color, 215))
    draw.text((x, y), label, font=font, fill=(255, 255, 255))


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current_line = words[0]

    for word in words[1:]:
        candidate = f"{current_line} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = candidate
            continue

        lines.append(current_line)
        current_line = word
        if len(lines) == max_lines - 1:
            break

    remaining_words = []
    if len(lines) < max_lines:
        remaining_words = words[len(" ".join(lines + [current_line]).split()):]

    final_line = " ".join([current_line] + remaining_words).strip()
    if len(lines) >= max_lines:
        return lines[:max_lines]

    while final_line:
        bbox = draw.textbbox((0, 0), final_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            break
        final_line = final_line[:-2].rstrip() + "..."

    lines.append(final_line)
    return lines[:max_lines]


def draw_top_transcription_bar(
    draw: ImageDraw.ImageDraw,
    frame_width: int,
    transcript: str,
) -> None:
    title_font = load_font(size=22)
    body_font = load_font(size=22)
    x1, y1, x2 = 20, 18, frame_width - 20
    wrapped_lines = wrap_text_to_width(
        draw,
        transcript,
        body_font,
        max_width=(x2 - x1) - 32,
        max_lines=2,
    )

    title_bbox = draw.textbbox((0, 0), "Transcricao", font=title_font)
    body_heights = [draw.textbbox((0, 0), line or " ", font=body_font)[3] for line in wrapped_lines]
    panel_height = 18 + (title_bbox[3] - title_bbox[1]) + 10 + sum(body_heights) + max(0, len(body_heights) - 1) * 6 + 18
    panel_box = (x1, y1, x2, y1 + panel_height)
    draw.rounded_rectangle(panel_box, radius=18, fill=(18, 18, 18, 220))

    current_y = y1 + 14
    draw.text((x1 + 16, current_y), "Transcricao", font=title_font, fill=(255, 255, 255))
    current_y += (title_bbox[3] - title_bbox[1]) + 10

    for index, line in enumerate(wrapped_lines):
        draw.text((x1 + 16, current_y), line, font=body_font, fill=(240, 240, 240))
        current_y += body_heights[index] + 6


def draw_bottom_dashboard(
    draw: ImageDraw.ImageDraw,
    frame_width: int,
    frame_height: int,
    video_emotion: str,
    video_sentiment: str,
    audio_sentiment: str,
    audio_status: str,
) -> None:
    title_font = load_font(size=24)
    body_font = load_font(size=20)
    emoji_font = load_font(size=20, emoji=True)
    x1, x2 = 20, frame_width - 20
    panel_height = 118
    y1 = frame_height - panel_height - 20
    y2 = frame_height - 20
    draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=(18, 18, 18, 220))

    draw.text((x1 + 16, y1 + 12), "SENTI", font=title_font, fill=(255, 255, 255))
    draw.text((x1 + 16, y1 + 46), f"Emocao do video: {video_emotion}", font=emoji_font, fill=(240, 240, 240))

    chip_y = y1 + 78
    chips = [
        ("Video", video_sentiment, SENTIMENT_COLORS.get(video_sentiment, (120, 120, 120))),
        ("Audio", audio_sentiment, SENTIMENT_COLORS.get(audio_sentiment, (120, 120, 120))),
        ("Status", audio_status, (120, 90, 50) if "ativo" in audio_status.lower() else (120, 120, 120)),
    ]

    chip_x = x1 + 16
    for prefix, value, color in chips:
        text = f"{prefix}: {value}"
        bbox = draw.textbbox((0, 0), text, font=body_font)
        chip_width = (bbox[2] - bbox[0]) + 24
        chip_box = (chip_x, chip_y, chip_x + chip_width, chip_y + 32)
        draw.rounded_rectangle(chip_box, radius=12, fill=(*color, 205))
        draw.text((chip_x + 12, chip_y + 5), text, font=body_font, fill=(255, 255, 255))
        chip_x += chip_width + 12


def annotate_frame(
    frame: np.ndarray,
    face_annotations: list[dict[str, object]],
    audio_state: AudioState,
) -> np.ndarray:
    for annotation in face_annotations:
        x1, y1, x2, y2 = annotation["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), annotation["color"], 2)

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image, "RGBA")

    for annotation in face_annotations:
        x1, y1, _, _ = annotation["bbox"]
        draw_label(
            draw,
            annotation["emotion_label"],
            origin=(x1, max(0, y1 - 34)),
            color=annotation["color"],
        )

    if face_annotations:
        primary_face = face_annotations[0]
        video_emotion = primary_face["emotion_label"]
        video_sentiment = primary_face["video_sentiment"]
    else:
        video_emotion = "Sem rosto detectado"
        video_sentiment = "Indisponivel"

    frame_width, frame_height = frame.shape[1], frame.shape[0]
    draw_top_transcription_bar(draw, frame_width, audio_state.text)
    draw_bottom_dashboard(
        draw,
        frame_width,
        frame_height,
        video_emotion=video_emotion,
        video_sentiment=video_sentiment,
        audio_sentiment=audio_state.sentiment_label,
        audio_status=audio_state.status,
    )

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def validate_assets() -> None:
    required_paths = (
        FACE_MODEL_PATH,
        EMOTION_MODEL_PATH,
        ASR_MODEL_PATH,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Arquivos obrigatorios ausentes: {', '.join(missing)}")


def main() -> None:
    validate_assets()

    base_options = mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
    face_options = mp.tasks.vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )

    emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)
    emotion_input_size = tuple(emotion_classifier.input_shape[1:3])

    audio_transcriber = AudioTranscriber(ASR_MODEL_PATH)
    audio_transcriber.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        audio_transcriber.stop()
        raise RuntimeError("Nao foi possivel abrir a camera 0.")

    with mp.tasks.vision.FaceDetector.create_from_options(face_options) as detector:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Falha ao ler frame da camera.")
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.monotonic() * 1000)
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)

                frame_height, frame_width = frame.shape[:2]
                face_annotations: list[dict[str, object]] = []

                for detection in detection_result.detections:
                    x1, y1, x2, y2 = clamp_bbox(
                        detection.bounding_box,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    if x2 <= x1 or y2 <= y1:
                        continue

                    gray_face = gray_frame[y1:y2, x1:x2]
                    if gray_face.size == 0:
                        continue

                    face_tensor = preprocess_face(gray_face, emotion_input_size)
                    emotion_scores = emotion_classifier.predict(face_tensor, verbose=0)[0]
                    emotion_index = int(np.argmax(emotion_scores))

                    face_annotations.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "emotion_label": EMOTION_LABELS[emotion_index],
                            "video_sentiment": VIDEO_SENTIMENTS[emotion_index],
                            "color": EMOTION_COLORS[emotion_index],
                        }
                    )

                frame = annotate_frame(frame, face_annotations, audio_transcriber.get_state())
                cv2.imshow("SENTI", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            audio_transcriber.stop()


if __name__ == "__main__":
    main()
