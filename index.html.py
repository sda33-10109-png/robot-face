import cv2
import numpy as np
import os
import urllib.request

# ==========================================
# 1. 기본 설정
# ==========================================
WINDOW_NAME = "Sheriff Robot - Face XYZ"

MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_detection_yunet_2023mar.onnx"
)

MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/"
    "main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

CAMERA_INDEX = 0
MIRROR = True

# 거리 계산용 근사값
# 일반적인 웹캠의 수평 초점거리를 가정한 값
FACE_WIDTH_CM = 15.0
FOCAL_FACTOR = 1.2

# 값의 흔들림을 줄이는 설정
SMOOTHING = 0.25

# ==========================================
# 2. YuNet 모델 다운로드
# ==========================================
if not os.path.isfile(MODEL_FILE):
    print("얼굴 인식 모델 다운로드 중...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    except Exception as e:
        raise RuntimeError(
            "모델 다운로드 실패. 인터넷 연결을 확인하세요."
        ) from e

if os.path.getsize(MODEL_FILE) < 10000:
    raise RuntimeError(
        "모델 파일이 너무 작습니다. 다운로드 파일을 확인하세요."
    )

# ==========================================
# 3. 카메라와 얼굴 검출기 준비
# ==========================================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "카메라를 열 수 없습니다. 다른 프로그램이 카메라를 "
        "사용하고 있는지 확인하세요."
    )

ok, first_frame = cap.read()

if not ok:
    cap.release()
    raise RuntimeError("카메라 화면을 읽을 수 없습니다.")

if MIRROR:
    first_frame = cv2.flip(first_frame, 1)

height, width = first_frame.shape[:2]

try:
    detector = cv2.FaceDetectorYN.create(
        MODEL_FILE,
        "",
        (width, height),
        0.65,
        0.3,
        5000
    )
except cv2.error as e:
    cap.release()
    raise RuntimeError(
        "YuNet 검출기를 초기화하지 못했습니다. "
        "OpenCV 설치 및 모델 파일을 확인하세요."
    ) from e

# XYZ 좌표의 평활화 상태
smooth_xyz = None

print("프로그램 실행 완료")
print("X: 좌우 위치")
print("Y: 상하 위치")
print("Z: 카메라와 얼굴 사이의 추정 거리")
print("Q 또는 ESC: 종료")

# ==========================================
# 4. 실시간 얼굴 인식 및 XYZ 계산
# ==========================================
try:
    while True:
        success, frame = cap.read()

        if not success:
            print("카메라 프레임 읽기 실패")
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]

        # 카메라 화면 크기에 맞춰 검출기 설정
        detector.setInputSize((W, H))

        # YuNet 얼굴 검출
        _, faces = detector.detect(frame)

        # 화면 중앙을 좌표 원점으로 설정
        origin_x = W // 2
        origin_y = H // 2

        cv2.drawMarker(
            frame,
            (origin_x, origin_y),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
            2
        )

        cv2.putText(
            frame,
            "Camera Center (0, 0)",
            (origin_x + 10, origin_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        if faces is not None and len(faces) > 0:
            # 가장 신뢰도가 높은 얼굴 선택
            face = max(faces, key=lambda item: item[-1])

            x, y, fw, fh = face[:4]
            confidence = float(face[-1])

            x, y, fw, fh = map(
                int, (x, y, fw, fh)
            )

            # 얼굴 중심 좌표
            cx = x + fw // 2
            cy = y + fh // 2

            # ----------------------------------
            # Z: 얼굴 크기로 거리 추정
            # ----------------------------------
            focal_px = FOCAL_FACTOR * W

            if fw > 0:
                z_cm = (
                    focal_px * FACE_WIDTH_CM / fw
                )
            else:
                z_cm = 0.0

            # ----------------------------------
            # X, Y: 카메라 중심 기준 위치 추정
            # ----------------------------------
            x_cm = (cx - W / 2) * z_cm / focal_px
            y_cm = (cy - H / 2) * z_cm / focal_px

            xyz = np.array(
                [x_cm, y_cm, z_cm],
                dtype=np.float64
            )

            # 좌표를 부드럽게 표시
            if smooth_xyz is None:
                smooth_xyz = xyz.copy()
            else:
                smooth_xyz = (
                    SMOOTHING * xyz
                    + (1.0 - SMOOTHING) * smooth_xyz
                )

            X, Y, Z = smooth_xyz

            # 얼굴 표시
            cv2.rectangle(
                frame,
                (x, y),
                (x + fw, y + fh),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 255, 255),
                -1
            )

            # 중심에서 얼굴 중심까지 선 표시
            cv2.line(
                frame,
                (origin_x, origin_y),
                (cx, cy),
                (255, 0, 255),
                2
            )

            # 검출 신뢰도
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            # XYZ 좌표
            cv2.putText(
                frame,
                f"X: {X:+.1f} cm",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Y: {Y:+.1f} cm",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Z: {Z:.1f} cm (estimated)",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 100, 0),
                2
            )

            cv2.putText(
                frame,
                "FACE DETECTED",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:
            # 얼굴을 놓쳤을 때 이전 좌표 초기화
            smooth_xyz = None

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if cv2.getWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램 종료")