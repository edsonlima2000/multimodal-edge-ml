from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model


FACE_MODEL_PATH = Path("models/blaze_face_short_range.tflite")
EMOTION_MODEL_PATH = Path("models/fer2013_mini_XCEPTION.102-0.66.hdf5")
EMOJI_FONT_PATH = Path(r"C:\Windows\Fonts\seguiemj.ttf")

# MiniXception treinado no dataset FER-2013.
EMOTION_LABELS = {
    0: "😠 Raiva (Angry)",
    1: "🤢 Nojo (Disgust)",
    2: "😨 Medo (Fear)",
    3: "😊 Feliz (Happy)",
    4: "😢 Triste (Sad)",
    5: "😲 Surpreso (Surprise)",
    6: "😐 Neutro (Neutral)",
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


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if EMOJI_FONT_PATH.exists():
        return ImageFont.truetype(str(EMOJI_FONT_PATH), size=size)
    return ImageFont.load_default()


def draw_label(
    frame: np.ndarray,
    label: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> np.ndarray:
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    font = load_font(size=28)

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

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def main() -> None:
    if not FACE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {FACE_MODEL_PATH}")
    if not EMOTION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {EMOTION_MODEL_PATH}")

    base_options = mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
    face_options = mp.tasks.vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )

    emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)
    emotion_input_size = tuple(emotion_classifier.input_shape[1:3])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
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
                    emotion_label = EMOTION_LABELS[emotion_index]
                    emotion_color = EMOTION_COLORS[emotion_index]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), emotion_color, 2)
                    frame = draw_label(
                        frame,
                        emotion_label,
                        origin=(x1, max(0, y1 - 34)),
                        color=emotion_color,
                    )

                cv2.imshow("MediaPipe + MiniXception", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
