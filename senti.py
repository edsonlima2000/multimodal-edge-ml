from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time

import cv2
from huggingface_hub import snapshot_download
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sounddevice as sd
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vosk import KaldiRecognizer, Model


FACE_MODEL_PATH = Path("models/blaze_face_short_range.tflite")
EMOTION_MODEL_PATH = Path("models/fer2013_mini_XCEPTION.102-0.66.hdf5")
ASR_MODEL_PATH = Path("models/vosk-model-small-pt-0.3")
TEXT_SENTIMENT_MODEL_PATH = Path("models/SYAS1-PTBR")
TEXT_SENTIMENT_MODEL_ID = "manushya-ai/SYAS1-PTBR"
EMOJI_FONT_PATH = Path(r"C:\Windows\Fonts\seguiemj.ttf")
UI_FONT_PATH = Path(r"C:\Windows\Fonts\segoeui.ttf")
EMOTION_INFERENCE_EVERY_N_FRAMES = 8
FACE_DETECTION_SCALE = 0.5
EMOTION_INFERENCE_MIN_INTERVAL_S = 1.0
AUDIO_SENTIMENT_PARTIAL_MIN_WORDS = 8
AUDIO_SENTIMENT_PARTIAL_WORD_STEP = 5
AUDIO_SENTIMENT_MAX_WORDS = 100

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

MODEL_INFERENCE_LOCK = threading.Lock()

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


@dataclass
class AudioState:
    text: str = "Aguardando audio..."
    sentiment_label: str = "Indisponivel"
    sentiment_color: tuple[int, int, int] = SENTIMENT_COLORS["Indisponivel"]
    status: str = "Audio nao iniciado"


@dataclass
class EmotionState:
    emotion_label: str = "Sem emocao"
    video_sentiment: str = "Indisponivel"
    color: tuple[int, int, int] = SENTIMENT_COLORS["Indisponivel"]


class TextSentimentAnalyzer:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, text: str) -> tuple[str, tuple[int, int, int]]:
        if not text.strip():
            return "Indisponivel", SENTIMENT_COLORS["Indisponivel"]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with MODEL_INFERENCE_LOCK:
            with torch.no_grad():
                logits = self.model(**inputs).logits
                prediction = int(torch.argmax(logits, dim=1).item())

        label = self.model.config.id2label[prediction]
        return label, SENTIMENT_COLORS.get(label, SENTIMENT_COLORS["Neutro"])


class AudioTranscriber:
    def __init__(self, model_path: Path, sentiment_analyzer: TextSentimentAnalyzer) -> None:
        self.model_path = model_path
        self.sentiment_analyzer = sentiment_analyzer
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
        self.last_classified_text = ""
        self.last_classified_word_count = 0

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

            is_final_result = False
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                transcript = result.get("text", "").strip()
                if transcript:
                    self.final_segments.append(transcript)
                self.partial_text = ""
                is_final_result = True
            else:
                partial = json.loads(self.recognizer.PartialResult())
                self.partial_text = partial.get("partial", "").strip()

            combined_text = " ".join(self.final_segments).strip()
            if self.partial_text:
                combined_text = f"{combined_text} {self.partial_text}".strip()

            analysis_text = limit_text_to_last_words(combined_text, AUDIO_SENTIMENT_MAX_WORDS)
            current_word_count = len(analysis_text.split())
            should_classify_partial = (
                bool(self.partial_text)
                and current_word_count >= AUDIO_SENTIMENT_PARTIAL_MIN_WORDS
                and (current_word_count - self.last_classified_word_count) >= AUDIO_SENTIMENT_PARTIAL_WORD_STEP
            )
            should_classify = (
                analysis_text != self.last_classified_text
                and (is_final_result or should_classify_partial)
            )

            if should_classify:
                sentiment_label, sentiment_color = self.sentiment_analyzer.predict(analysis_text)
                self.last_classified_text = analysis_text
                self.last_classified_word_count = current_word_count
            else:
                with self.state_lock:
                    sentiment_label = self.state.sentiment_label
                    sentiment_color = self.state.sentiment_color
            display_text = combined_text if combined_text else "Aguardando audio..."

            with self.state_lock:
                self.state.text = display_text
                self.state.sentiment_label = sentiment_label
                self.state.sentiment_color = sentiment_color


def limit_text_to_last_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[-max_words:]).strip()


class EmotionClassifierWorker:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.latest_state = EmotionState()
        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.model = load_model(model_path, compile=False)
        self.input_size = tuple(self.model.input_shape[1:3])

    def start(self) -> None:
        self.worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def submit(self, gray_face: np.ndarray) -> None:
        face_copy = gray_face.copy()
        try:
            self.input_queue.put_nowait(face_copy)
        except queue.Full:
            try:
                _ = self.input_queue.get_nowait()
            except queue.Empty:
                pass
            self.input_queue.put_nowait(face_copy)

    def get_state(self) -> EmotionState:
        with self.state_lock:
            return EmotionState(
                emotion_label=self.latest_state.emotion_label,
                video_sentiment=self.latest_state.video_sentiment,
                color=self.latest_state.color,
            )

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                gray_face = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            face_tensor = preprocess_face(gray_face, self.input_size)
            with MODEL_INFERENCE_LOCK:
                emotion_scores = self.model.predict(face_tensor, verbose=0)[0]
            emotion_index = int(np.argmax(emotion_scores))

            with self.state_lock:
                self.latest_state = EmotionState(
                    emotion_label=EMOTION_LABELS[emotion_index],
                    video_sentiment=VIDEO_SENTIMENTS[emotion_index],
                    color=EMOTION_COLORS[emotion_index],
                )


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


def scale_bbox(
    bbox: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    scaled_x1 = max(0, min(frame_width, int(x1 * scale_x)))
    scaled_y1 = max(0, min(frame_height, int(y1 * scale_y)))
    scaled_x2 = max(0, min(frame_width, int(x2 * scale_x)))
    scaled_y2 = max(0, min(frame_height, int(y2 * scale_y)))
    return scaled_x1, scaled_y1, scaled_x2, scaled_y2


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


def ensure_text_sentiment_model() -> Path:
    if TEXT_SENTIMENT_MODEL_PATH.exists():
        return TEXT_SENTIMENT_MODEL_PATH

    snapshot_download(
        repo_id=TEXT_SENTIMENT_MODEL_ID,
        local_dir=str(TEXT_SENTIMENT_MODEL_PATH),
    )
    return TEXT_SENTIMENT_MODEL_PATH


def main() -> None:
    validate_assets()
    text_sentiment_model_path = ensure_text_sentiment_model()

    base_options = mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
    face_options = mp.tasks.vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )

    text_sentiment_analyzer = TextSentimentAnalyzer(text_sentiment_model_path)
    emotion_worker = EmotionClassifierWorker(EMOTION_MODEL_PATH)
    emotion_worker.start()

    audio_transcriber = AudioTranscriber(ASR_MODEL_PATH, text_sentiment_analyzer)
    audio_transcriber.start()
    frame_index = 0
    last_face_submit_time = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        emotion_worker.stop()
        audio_transcriber.stop()
        raise RuntimeError("Nao foi possivel abrir a camera 0.")

    with mp.tasks.vision.FaceDetector.create_from_options(face_options) as detector:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Falha ao ler frame da camera.")
                    break

                frame_height, frame_width = frame.shape[:2]
                processing_width = max(1, int(frame_width * FACE_DETECTION_SCALE))
                processing_height = max(1, int(frame_height * FACE_DETECTION_SCALE))
                processing_frame = cv2.resize(frame, (processing_width, processing_height))

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_processing_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_processing_frame,
                )
                timestamp_ms = int(time.monotonic() * 1000)
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                face_annotations: list[dict[str, object]] = []
                frame_index += 1

                for detection in detection_result.detections:
                    scaled_bbox = clamp_bbox(
                        detection.bounding_box,
                        frame_width=processing_width,
                        frame_height=processing_height,
                    )
                    x1, y1, x2, y2 = scale_bbox(
                        scaled_bbox,
                        scale_x=frame_width / processing_width,
                        scale_y=frame_height / processing_height,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    if x2 <= x1 or y2 <= y1:
                        continue

                    gray_face = gray_frame[y1:y2, x1:x2]
                    if gray_face.size == 0:
                        continue

                    current_emotion_state = emotion_worker.get_state()
                    should_run_emotion = (
                        frame_index % EMOTION_INFERENCE_EVERY_N_FRAMES == 0
                        and (time.monotonic() - last_face_submit_time) >= EMOTION_INFERENCE_MIN_INTERVAL_S
                    )

                    if should_run_emotion:
                        emotion_worker.submit(gray_face)
                        last_face_submit_time = time.monotonic()

                    face_annotations.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "emotion_label": current_emotion_state.emotion_label,
                            "video_sentiment": current_emotion_state.video_sentiment,
                            "color": current_emotion_state.color,
                        }
                    )

                frame = annotate_frame(frame, face_annotations, audio_transcriber.get_state())
                cv2.imshow("SENTI", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            emotion_worker.stop()
            audio_transcriber.stop()


if __name__ == "__main__":
    main()
