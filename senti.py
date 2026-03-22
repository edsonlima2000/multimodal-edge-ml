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
import opensmile
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
VOICE_ANALYSIS_WINDOW_S = 2.0
VOICE_ANALYSIS_STEP_S = 1.0
VOICE_BASELINE_MIN_WINDOWS = 3
VOICE_BASELINE_HISTORY = 20
VOICE_MIN_RMS = 0.01
SENTIMENT_TIMELINE_WINDOW_S = 60.0
SENTIMENT_TIMELINE_SAMPLE_STEP_S = 0.5
BOTTOM_PANEL_HEIGHT_RATIO = 0.25

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

FUSION_WEIGHTS = {
    "video": 0.55,
    "voice": 0.38,
    "text": 0.07,
}

VOICE_SENTIMENTS = {
    "Alegria": "Positivo",
    "Tristeza": "Negativo",
    "Raiva": "Negativo",
    "Medo": "Negativo",
    "Surpresa": "Neutro",
    "Nojo": "Negativo",
    "Neutra": "Neutro",
    "Indisponivel": "Indisponivel",
}

VOICE_EMOTION_COLORS = {
    "Alegria": (0, 190, 255),
    "Tristeza": (220, 120, 60),
    "Raiva": (60, 70, 220),
    "Medo": (180, 90, 180),
    "Surpresa": (0, 220, 220),
    "Nojo": (40, 150, 40),
    "Neutra": (160, 160, 160),
    "Indisponivel": (120, 120, 120),
}


@dataclass
class AudioState:
    text: str = "Aguardando audio..."
    sentiment_label: str = "Indisponivel"
    sentiment_color: tuple[int, int, int] = SENTIMENT_COLORS["Indisponivel"]
    sentiment_confidence: float = 0.0
    voice_emotion_label: str = "Indisponivel"
    voice_emotion_color: tuple[int, int, int] = VOICE_EMOTION_COLORS["Indisponivel"]
    voice_emotion_confidence: float = 0.0
    status: str = "Audio nao iniciado"


@dataclass
class EmotionState:
    emotion_label: str = "Sem emocao"
    video_sentiment: str = "Indisponivel"
    color: tuple[int, int, int] = SENTIMENT_COLORS["Indisponivel"]
    confidence: float = 0.0


@dataclass
class SentimentTimelinePoint:
    timestamp: float
    score: float
    label: str
    confidence: float


class TextSentimentAnalyzer:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, text: str) -> tuple[str, tuple[int, int, int], float]:
        if not text.strip():
            return "Indisponivel", SENTIMENT_COLORS["Indisponivel"], 0.0

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
                probabilities = torch.softmax(logits, dim=1)
                prediction = int(torch.argmax(probabilities, dim=1).item())
                confidence = float(probabilities[0, prediction].item())

        label = self.model.config.id2label[prediction]
        return label, SENTIMENT_COLORS.get(label, SENTIMENT_COLORS["Neutro"]), confidence


class VoiceEmotionAnalyzer:
    def __init__(self) -> None:
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.history: deque[dict[str, float]] = deque(maxlen=VOICE_BASELINE_HISTORY)

    def predict(self, audio_signal: np.ndarray, sample_rate: int) -> tuple[str, tuple[int, int, int], float]:
        metrics = self._extract_metrics(audio_signal, sample_rate)
        if metrics is None:
            return "Indisponivel", VOICE_EMOTION_COLORS["Indisponivel"], 0.0

        if len(self.history) < VOICE_BASELINE_MIN_WINDOWS:
            self.history.append(metrics)
            return "Indisponivel", VOICE_EMOTION_COLORS["Indisponivel"], 0.0

        z = self._compute_zscores(metrics)
        emotion_scores = self._score_emotions(z)
        emotion_label, confidence = self._select_emotion(emotion_scores, z)
        self.history.append(metrics)
        return emotion_label, VOICE_EMOTION_COLORS.get(emotion_label, VOICE_EMOTION_COLORS["Neutra"]), confidence

    def _extract_metrics(self, audio_signal: np.ndarray, sample_rate: int) -> dict[str, float] | None:
        signal = audio_signal.astype("float32")
        if signal.size and float(np.max(np.abs(signal))) > 1.5:
            signal = signal / 32768.0

        rms = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0
        if rms < VOICE_MIN_RMS:
            return None

        features = self.smile.process_signal(signal, sampling_rate=sample_rate).iloc[0]
        metrics = {
            "pitch_mean": float(features["F0semitoneFrom27.5Hz_sma3nz_amean"]),
            "pitch_peak": float(features["F0semitoneFrom27.5Hz_sma3nz_percentile80.0"]),
            "pitch_range": float(features["F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"]),
            "loudness_mean": float(features["loudness_sma3_amean"]),
            "loudness_peak": float(features["loudness_sma3_percentile80.0"]),
            "speech_rate": float(features["VoicedSegmentsPerSec"]),
            "segment_duration": float(features["MeanVoicedSegmentLengthSec"]),
            "sound_level": float(features["equivalentSoundLevel_dBp"]),
        }
        if any(np.isnan(value) or np.isinf(value) for value in metrics.values()):
            return None
        return metrics

    def _compute_zscores(self, metrics: dict[str, float]) -> dict[str, float]:
        zscores: dict[str, float] = {}
        for key, value in metrics.items():
            history_values = np.array([sample[key] for sample in self.history], dtype=np.float32)
            mean = float(np.mean(history_values))
            std = float(np.std(history_values))
            safe_std = max(std, 1e-3)
            zscores[key] = (value - mean) / safe_std
        return zscores

    def _score_emotions(self, z: dict[str, float]) -> dict[str, float]:
        return {
            "Alegria": (
                1.25 * z["pitch_mean"]
                + 1.15 * z["loudness_mean"]
                + 1.00 * z["speech_rate"]
                - 0.75 * z["segment_duration"]
                + 0.45 * z["pitch_peak"]
            ),
            "Tristeza": (
                -1.20 * z["pitch_mean"]
                - 1.15 * z["loudness_mean"]
                - 0.90 * z["pitch_range"]
                - 0.25 * z["speech_rate"]
            ),
            "Raiva": (
                1.70 * z["loudness_peak"]
                + 0.90 * z["loudness_mean"]
                + 0.40 * z["sound_level"]
                + 0.30 * z["pitch_range"]
            ),
            "Medo": (
                1.65 * z["pitch_range"]
                - 0.90 * z["loudness_mean"]
                + 0.80 * z["speech_rate"]
                - 0.65 * z["segment_duration"]
            ),
            "Surpresa": (
                1.75 * z["pitch_peak"]
                + 1.00 * z["pitch_mean"]
                + 0.35 * z["pitch_range"]
                - 0.25 * z["loudness_mean"]
            ),
            "Nojo": (
                1.80 * z["segment_duration"]
                - 0.90 * z["speech_rate"]
                + 0.20 * z["loudness_mean"]
            ),
        }

    def _select_emotion(self, emotion_scores: dict[str, float], z: dict[str, float]) -> tuple[str, float]:
        max_abs_deviation = max(abs(value) for value in z.values())
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        best_score = emotion_scores[best_emotion]
        ranked_scores = sorted(emotion_scores.values(), reverse=True)
        score_margin = ranked_scores[0] - ranked_scores[1] if len(ranked_scores) > 1 else ranked_scores[0]
        normalized_deviation = min(max_abs_deviation / 2.5, 1.0)
        normalized_score = min(best_score / 2.5, 1.0)
        normalized_margin = min(score_margin / 1.5, 1.0)
        confidence = max(0.0, min(1.0, 0.45 * normalized_score + 0.35 * normalized_margin + 0.20 * normalized_deviation))

        if max_abs_deviation < 0.65:
            return "Neutra", max(0.0, min(1.0, 1.0 - (max_abs_deviation / 0.65)))
        if best_score < 1.2 or score_margin < 0.35:
            return "Indisponivel", 0.0
        if best_emotion in {"Nojo", "Raiva"} and best_score < 1.45:
            return "Neutra", confidence * 0.5
        return best_emotion, confidence


class VoiceEmotionWorker:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.latest_label = "Indisponivel"
        self.latest_color = VOICE_EMOTION_COLORS["Indisponivel"]
        self.latest_confidence = 0.0
        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.analyzer = VoiceEmotionAnalyzer()
        self.worker_thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def submit(self, audio_signal: np.ndarray) -> None:
        signal_copy = audio_signal.copy()
        try:
            self.input_queue.put_nowait(signal_copy)
        except queue.Full:
            try:
                _ = self.input_queue.get_nowait()
            except queue.Empty:
                pass
            self.input_queue.put_nowait(signal_copy)

    def get_state(self) -> tuple[str, tuple[int, int, int], float]:
        with self.state_lock:
            return self.latest_label, self.latest_color, self.latest_confidence

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                audio_signal = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            label, color, confidence = self.analyzer.predict(audio_signal, self.sample_rate)
            with self.state_lock:
                self.latest_label = label
                self.latest_color = color
                self.latest_confidence = confidence


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
        self.voice_buffer: deque[np.ndarray] = deque()
        self.voice_buffer_sample_count = 0
        self.total_voice_samples_seen = 0
        self.last_voice_analysis_total_samples = 0
        self.voice_worker: VoiceEmotionWorker | None = None

    def start(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo de ASR nao encontrado: {self.model_path}")

        device_info = sd.query_devices(kind="input")
        self.sample_rate = int(device_info["default_samplerate"])
        self.model = Model(str(self.model_path))
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(False)
        self.voice_worker = VoiceEmotionWorker(self.sample_rate)
        self.voice_worker.start()

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
        if self.voice_worker is not None:
            self.voice_worker.stop()
        with self.state_lock:
            self.state.status = "Microfone encerrado"

    def get_state(self) -> AudioState:
        with self.state_lock:
            return AudioState(
                text=self.state.text,
                sentiment_label=self.state.sentiment_label,
                sentiment_color=self.state.sentiment_color,
                sentiment_confidence=self.state.sentiment_confidence,
                voice_emotion_label=self.state.voice_emotion_label,
                voice_emotion_color=self.state.voice_emotion_color,
                voice_emotion_confidence=self.state.voice_emotion_confidence,
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

            self._update_voice_buffer(np.frombuffer(data, dtype=np.int16))
            self._maybe_submit_voice_window()

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
                sentiment_label, sentiment_color, sentiment_confidence = self.sentiment_analyzer.predict(analysis_text)
                self.last_classified_text = analysis_text
                self.last_classified_word_count = current_word_count
            else:
                with self.state_lock:
                    sentiment_label = self.state.sentiment_label
                    sentiment_color = self.state.sentiment_color
                    sentiment_confidence = self.state.sentiment_confidence
            display_text = combined_text if combined_text else "Aguardando audio..."

            with self.state_lock:
                self.state.text = display_text
                self.state.sentiment_label = sentiment_label
                self.state.sentiment_color = sentiment_color
                self.state.sentiment_confidence = sentiment_confidence
                if self.voice_worker is not None:
                    voice_label, voice_color, voice_confidence = self.voice_worker.get_state()
                    self.state.voice_emotion_label = voice_label
                    self.state.voice_emotion_color = voice_color
                    self.state.voice_emotion_confidence = voice_confidence

    def _update_voice_buffer(self, audio_chunk: np.ndarray) -> None:
        if audio_chunk.size == 0:
            return

        self.voice_buffer.append(audio_chunk.copy())
        self.voice_buffer_sample_count += int(audio_chunk.size)
        self.total_voice_samples_seen += int(audio_chunk.size)

        max_buffer_samples = int(self.sample_rate * max(VOICE_ANALYSIS_WINDOW_S * 3, 6))
        while self.voice_buffer and self.voice_buffer_sample_count > max_buffer_samples:
            removed = self.voice_buffer.popleft()
            self.voice_buffer_sample_count -= int(removed.size)

    def _maybe_submit_voice_window(self) -> None:
        if self.voice_worker is None:
            return

        window_samples = int(self.sample_rate * VOICE_ANALYSIS_WINDOW_S)
        step_samples = int(self.sample_rate * VOICE_ANALYSIS_STEP_S)
        if self.voice_buffer_sample_count < window_samples:
            return
        if (self.total_voice_samples_seen - self.last_voice_analysis_total_samples) < step_samples:
            return

        recent_window = self._get_recent_voice_window(window_samples)
        self.voice_worker.submit(recent_window.astype(np.float32))
        self.last_voice_analysis_total_samples = self.total_voice_samples_seen

    def _get_recent_voice_window(self, window_samples: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        collected = 0

        for chunk in reversed(self.voice_buffer):
            chunks.append(chunk)
            collected += int(chunk.size)
            if collected >= window_samples:
                break

        recent_audio = np.concatenate(list(reversed(chunks))) if chunks else np.zeros(window_samples, dtype=np.int16)
        return recent_audio[-window_samples:]


def limit_text_to_last_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[-max_words:]).strip()


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def format_with_confidence(label: str, confidence: float) -> str:
    if label == "Indisponivel":
        return label
    return f"{label} [{round(clamp_confidence(confidence) * 100)}%]"


def sentiment_to_class(label: str) -> str | None:
    if label == "Positivo":
        return "Positivo"
    if label == "Negativo":
        return "Negativo"
    if label == "Neutro":
        return "Neutro"
    return None


def fuse_multimodal_sentiment(
    video_sentiment: str,
    video_confidence: float,
    text_sentiment: str,
    text_confidence: float,
    voice_emotion: str,
    voice_confidence: float,
) -> tuple[str, float]:
    modal_votes = {
        "video": (sentiment_to_class(video_sentiment), clamp_confidence(video_confidence)),
        "text": (sentiment_to_class(text_sentiment), clamp_confidence(text_confidence)),
        "voice": (sentiment_to_class(VOICE_SENTIMENTS.get(voice_emotion, "Indisponivel")), clamp_confidence(voice_confidence)),
    }

    class_scores = {
        "Positivo": 0.0,
        "Negativo": 0.0,
        "Neutro": 0.0,
    }

    total_vote_mass = 0.0
    for modal_name, (sentiment_class, confidence) in modal_votes.items():
        if sentiment_class is None or confidence <= 0:
            continue
        weighted_confidence = FUSION_WEIGHTS[modal_name] * confidence
        class_scores[sentiment_class] += weighted_confidence
        total_vote_mass += weighted_confidence

    if total_vote_mass <= 0:
        return "Indisponivel", 0.0

    final_label = max(class_scores, key=class_scores.get)
    final_confidence = class_scores[final_label] / total_vote_mass
    return final_label, final_confidence


def sentiment_to_timeline_score(label: str, confidence: float) -> float:
    clamped_confidence = clamp_confidence(confidence)
    if label == "Positivo":
        return clamped_confidence
    if label == "Negativo":
        return -clamped_confidence
    if label == "Neutro":
        return 0.0
    return 0.0


def prune_sentiment_timeline(timeline: deque[SentimentTimelinePoint], now: float) -> None:
    while timeline and (now - timeline[0].timestamp) > SENTIMENT_TIMELINE_WINDOW_S:
        timeline.popleft()


def update_sentiment_timeline(
    timeline: deque[SentimentTimelinePoint],
    label: str,
    confidence: float,
    now: float,
) -> None:
    prune_sentiment_timeline(timeline, now)
    if label not in {"Positivo", "Negativo", "Neutro"}:
        return

    point = SentimentTimelinePoint(
        timestamp=now,
        score=sentiment_to_timeline_score(label, confidence),
        label=label,
        confidence=clamp_confidence(confidence),
    )
    if timeline and (now - timeline[-1].timestamp) < SENTIMENT_TIMELINE_SAMPLE_STEP_S:
        timeline[-1] = SentimentTimelinePoint(
            timestamp=timeline[-1].timestamp,
            score=point.score,
            label=point.label,
            confidence=point.confidence,
        )
    else:
        timeline.append(point)
    prune_sentiment_timeline(timeline, now)


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
                confidence=self.latest_state.confidence,
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
            emotion_confidence = float(emotion_scores[emotion_index])

            with self.state_lock:
                self.latest_state = EmotionState(
                    emotion_label=EMOTION_LABELS[emotion_index],
                    video_sentiment=VIDEO_SENTIMENTS[emotion_index],
                    color=EMOTION_COLORS[emotion_index],
                    confidence=emotion_confidence,
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
    font = load_font(size=14, emoji=True)
    x, y = origin
    left, top, right, bottom = draw.textbbox((x, y), label, font=font)
    padding = 6
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
    title_font = load_font(size=18)
    body_font = load_font(size=14)
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


def draw_sentiment_timeline_chart(
    draw: ImageDraw.ImageDraw,
    chart_box: tuple[int, int, int, int],
    timeline: deque[SentimentTimelinePoint],
    now: float,
) -> None:
    title_font = load_font(size=14)
    label_font = load_font(size=10)
    x1, y1, x2, y2 = chart_box
    draw.rounded_rectangle(chart_box, radius=16, fill=(24, 24, 24, 210))

    title_text = "Sentimento multimodal | ultimo minuto"
    draw.text((x1 + 12, y1 + 8), title_text, font=title_font, fill=(245, 245, 245))

    plot_left = x1 + 42
    plot_right = x2 - 14
    plot_top = y1 + 26
    plot_bottom = y2 - 18
    draw.rounded_rectangle((plot_left, plot_top, plot_right, plot_bottom), radius=12, outline=(70, 70, 70, 220), width=1)

    guide_levels = [
        (1.0, "P100"),
        (0.5, "P50"),
        (0.0, "0"),
        (-0.5, "N50"),
        (-1.0, "N100"),
    ]
    plot_height = max(1, plot_bottom - plot_top)
    plot_width = max(1, plot_right - plot_left)

    def score_to_y(score: float) -> int:
        normalized = (1.0 - score) / 2.0
        return int(plot_top + normalized * plot_height)

    for guide_score, guide_label in guide_levels:
        guide_y = score_to_y(guide_score)
        guide_color = (95, 95, 95, 225) if guide_score == 0.0 else (58, 58, 58, 200)
        draw.line((plot_left, guide_y, plot_right, guide_y), fill=guide_color, width=2 if guide_score == 0.0 else 1)
        draw.text((x1 + 8, guide_y - 5), guide_label, font=label_font, fill=(175, 175, 175))

    quarter_step = plot_width / 4.0
    for quarter_index in range(1, 4):
        grid_x = int(plot_left + (quarter_step * quarter_index))
        draw.line((grid_x, plot_top, grid_x, plot_bottom), fill=(46, 46, 46, 160), width=1)

    draw.text((plot_left, plot_bottom + 3), "-60s", font=label_font, fill=(165, 165, 165))
    now_bbox = draw.textbbox((0, 0), "agora", font=label_font)
    draw.text((plot_right - (now_bbox[2] - now_bbox[0]), plot_bottom + 3), "agora", font=label_font, fill=(165, 165, 165))

    if not timeline:
        empty_text = "Aguardando historico suficiente para plotagem"
        empty_bbox = draw.textbbox((0, 0), empty_text, font=label_font)
        empty_x = plot_left + max(0, (plot_width - (empty_bbox[2] - empty_bbox[0])) // 2)
        empty_y = plot_top + max(0, (plot_height - (empty_bbox[3] - empty_bbox[1])) // 2)
        draw.text((empty_x, empty_y), empty_text, font=label_font, fill=(145, 145, 145))
        return

    timeline_points = [point for point in timeline if (now - point.timestamp) <= SENTIMENT_TIMELINE_WINDOW_S]
    if len(timeline_points) == 1:
        timeline_points = [timeline_points[0], timeline_points[0]]

    polyline_points: list[tuple[int, int]] = []
    for point in timeline_points:
        elapsed = max(0.0, min(SENTIMENT_TIMELINE_WINDOW_S, now - point.timestamp))
        x_position = int(plot_right - ((elapsed / SENTIMENT_TIMELINE_WINDOW_S) * plot_width))
        polyline_points.append((x_position, score_to_y(point.score)))

    for index in range(1, len(polyline_points)):
        segment_color = SENTIMENT_COLORS.get(timeline_points[index].label, (160, 160, 160))
        draw.line([polyline_points[index - 1], polyline_points[index]], fill=(*segment_color, 235), width=3)

    latest_x, latest_y = polyline_points[-1]
    latest_color = SENTIMENT_COLORS.get(timeline_points[-1].label, (160, 160, 160))
    draw.ellipse((latest_x - 4, latest_y - 4, latest_x + 4, latest_y + 4), fill=(*latest_color, 255), outline=(255, 255, 255, 220))


def get_right_dashboard_box(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    panel_width = 230
    x2 = frame_width - 20
    x1 = max(20, x2 - panel_width)
    y1 = 120
    y2 = frame_height - 20
    return x1, y1, x2, y2


def draw_bottom_timeline_panel(
    draw: ImageDraw.ImageDraw,
    frame_width: int,
    frame_height: int,
    right_panel_x1: int,
    timeline: deque[SentimentTimelinePoint],
    now: float,
) -> None:
    chart_x1 = 20
    chart_x2 = max(chart_x1 + 180, right_panel_x1 - 14)
    chart_height = max(72, int(frame_height * BOTTOM_PANEL_HEIGHT_RATIO))
    chart_height = min(chart_height, max(72, int(frame_height * 0.25)))
    chart_y2 = frame_height - 20
    chart_y1 = max(120, chart_y2 - chart_height)
    draw_sentiment_timeline_chart(
        draw,
        chart_box=(chart_x1, chart_y1, chart_x2, chart_y2),
        timeline=timeline,
        now=now,
    )


def draw_bottom_dashboard(
    draw: ImageDraw.ImageDraw,
    frame_width: int,
    frame_height: int,
    video_emotion: str,
    video_sentiment: str,
    video_confidence: float,
    text_sentiment: str,
    text_confidence: float,
    voice_emotion: str,
    voice_confidence: float,
    voice_color: tuple[int, int, int],
    final_sentiment: str,
    final_confidence: float,
    audio_status: str,
) -> None:
    title_font = load_font(size=18)
    body_font = load_font(size=14)
    emoji_font = load_font(size=14, emoji=True)
    x1, y1, x2, y2 = get_right_dashboard_box(frame_width, frame_height)
    draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=(18, 18, 18, 220))

    current_y = y1 + 14
    draw.text((x1 + 14, current_y), "SENTI", font=title_font, fill=(255, 255, 255))
    current_y += 28
    final_text = f"Final: {format_with_confidence(final_sentiment, final_confidence)}"
    final_color = SENTIMENT_COLORS.get(final_sentiment, SENTIMENT_COLORS["Indisponivel"])
    final_box = (x1 + 14, current_y, x2 - 14, current_y + 32)
    draw.rounded_rectangle(final_box, radius=10, fill=(*final_color, 210))
    draw.text((x1 + 24, current_y + 8), final_text, font=body_font, fill=(255, 255, 255))
    current_y += 42
    draw.text((x1 + 14, current_y), "Emocao do video", font=body_font, fill=(210, 210, 210))
    current_y += 20
    emotion_lines = wrap_text_to_width(draw, video_emotion, emoji_font, 230 - 28, 2)
    for line in emotion_lines:
        draw.text((x1 + 14, current_y), line, font=emoji_font, fill=(240, 240, 240))
        current_y += 18

    current_y += 6
    chips = [
        ("Video", format_with_confidence(video_sentiment, video_confidence), SENTIMENT_COLORS.get(video_sentiment, (120, 120, 120))),
        ("Texto", format_with_confidence(text_sentiment, text_confidence), SENTIMENT_COLORS.get(text_sentiment, (120, 120, 120))),
        ("Voz", format_with_confidence(voice_emotion, voice_confidence), voice_color),
        ("Status", audio_status, (120, 90, 50) if "ativo" in audio_status.lower() else (120, 120, 120)),
    ]

    for prefix, value, color in chips:
        text = f"{prefix}: {value}"
        chip_box = (x1 + 14, current_y, x2 - 14, current_y + 30)
        draw.rounded_rectangle(chip_box, radius=10, fill=(*color, 205))
        draw.text((x1 + 24, current_y + 7), text, font=body_font, fill=(255, 255, 255))
        current_y += 38


def annotate_frame(
    frame: np.ndarray,
    face_annotations: list[dict[str, object]],
    audio_state: AudioState,
    sentiment_timeline: deque[SentimentTimelinePoint],
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
        video_confidence = float(primary_face["confidence"])
    else:
        video_emotion = "Sem rosto detectado"
        video_sentiment = "Indisponivel"
        video_confidence = 0.0

    final_sentiment, final_confidence = fuse_multimodal_sentiment(
        video_sentiment=video_sentiment,
        video_confidence=video_confidence,
        text_sentiment=audio_state.sentiment_label,
        text_confidence=audio_state.sentiment_confidence,
        voice_emotion=audio_state.voice_emotion_label,
        voice_confidence=audio_state.voice_emotion_confidence,
    )
    now = time.monotonic()
    update_sentiment_timeline(sentiment_timeline, final_sentiment, final_confidence, now)

    frame_width, frame_height = frame.shape[1], frame.shape[0]
    draw_top_transcription_bar(draw, frame_width, audio_state.text)
    right_panel_x1, _, _, _ = get_right_dashboard_box(frame_width, frame_height)
    draw_bottom_timeline_panel(
        draw,
        frame_width,
        frame_height,
        right_panel_x1=right_panel_x1,
        timeline=sentiment_timeline,
        now=now,
    )
    draw_bottom_dashboard(
        draw,
        frame_width,
        frame_height,
        video_emotion=video_emotion,
        video_sentiment=video_sentiment,
        video_confidence=video_confidence,
        text_sentiment=audio_state.sentiment_label,
        text_confidence=audio_state.sentiment_confidence,
        voice_emotion=audio_state.voice_emotion_label,
        voice_confidence=audio_state.voice_emotion_confidence,
        voice_color=audio_state.voice_emotion_color,
        final_sentiment=final_sentiment,
        final_confidence=final_confidence,
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
    sentiment_timeline: deque[SentimentTimelinePoint] = deque()

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
                            "confidence": current_emotion_state.confidence,
                        }
                    )

                frame = annotate_frame(frame, face_annotations, audio_transcriber.get_state(), sentiment_timeline)
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
