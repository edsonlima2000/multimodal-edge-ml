from pathlib import Path
import time

import cv2
import mediapipe as mp


MODEL_PATH = Path("models/blaze_face_short_range.tflite")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_PATH}")

    base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir a camera 0.")

    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Falha ao ler frame da camera.")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(time.time() * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)

                for detection in result.detections:
                    mp.tasks.vision.drawing_utils.draw_detection(frame, detection)

                cv2.imshow("Hello MediaPipe", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
