import cv2
import numpy as np
import os
import urllib.request

# Sheriff Robot View
# Webcam face detection and approximate XYZ position estimation.
# Note: XYZ values are estimates, not calibrated physical measurements.

WINDOW_NAME = "Sheriff Robot - Face XYZ"
MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_detection_yunet_2023mar.onnx"
)
MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/"
    "main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

CAMERA_INDEX = 0
MIRROR = True

# Approximate assumptions for distance estimation.
FACE_WIDTH_CM = 15.0
FOCAL_FACTOR = 1.2
SMOOTHING = 0.25


def download_model():
    """Download YuNet model if it is not already present."""
    if os.path.isfile(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 10000:
        return

    print("Downloading YuNet face detection model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the face model. Check your internet connection."
        ) from exc

    if not os.path.isfile(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 10000:
        raise RuntimeError(
            "The downloaded model file appears incomplete. Delete it and run again."
        )


def main():
    download_model()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions and close other apps using it."
        )

    window_created = False
    smooth_xyz = None

    try:
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the webcam.")

        if MIRROR:
            first_frame = cv2.flip(first_frame, 1)

        height, width = first_frame.shape[:2]

        detector = cv2.FaceDetectorYN.create(
            MODEL_FILE, "", (width, height), 0.65, 0.3, 5000
        )

        print("Program started.")
        print("X: left/right, Y: up/down, Z: estimated distance")
        print("Press Q or ESC to quit.")

        while True:
            success, frame = cap.read()
            if not success:
                print("Could not read a webcam frame.")
                break

            if MIRROR:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)

            origin_x, origin_y = w // 2, h // 2
            cv2.drawMarker(
                frame, (origin_x, origin_y), (255, 255, 255),
                cv2.MARKER_CROSS, 20, 2
            )
            cv2.putText(
                frame, "Camera center (0, 0)",
                (origin_x + 10, max(20, origin_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            if faces is not None and len(faces) > 0:
                # Select the detected face with the highest confidence.
                face = max(faces, key=lambda item: item[-1])
                x, y, fw, fh = map(int, face[:4])
                confidence = float(face[-1])
                cx, cy = x + fw // 2, y + fh // 2

                # Approximate distance from apparent face width.
                focal_px = FOCAL_FACTOR * w
                z_cm = focal_px * FACE_WIDTH_CM / fw if fw > 0 else 0.0

                # Camera-centered coordinates; positive Y points down.
                x_cm = (cx - w / 2) * z_cm / focal_px
                y_cm = (cy - h / 2) * z_cm / focal_px

                xyz = np.array([x_cm, y_cm, z_cm], dtype=np.float64)
                if smooth_xyz is None:
                    smooth_xyz = xyz.copy()
                else:
                    smooth_xyz = (
                        SMOOTHING * xyz + (1.0 - SMOOTHING) * smooth_xyz
                    )

                X, Y, Z = smooth_xyz

                cv2.rectangle(
                    frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2
                )
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.line(
                    frame, (origin_x, origin_y), (cx, cy), (255, 0, 255), 2
                )

                lines = [
                    (f"Confidence: {confidence:.2f}", (0, 255, 0)),
                    (f"X: {X:+.1f} cm", (0, 0, 255)),
                    (f"Y: {Y:+.1f} cm", (0, 255, 0)),
                    (f"Z: {Z:.1f} cm (estimated)", (255, 100, 0)),
                ]
                for i, (label, color) in enumerate(lines):
                    cv2.putText(
                        frame, label, (20, 30 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                    )
                cv2.putText(
                    frame, "FACE DETECTED", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            else:
                smooth_xyz = None
                cv2.putText(
                    frame, "NO FACE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            cv2.imshow(WINDOW_NAME, frame)
            window_created = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if (
                window_created
                and cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1
            ):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Program closed.")


if __name__ == "__main__":
    main()
