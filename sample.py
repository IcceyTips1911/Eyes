import cv2
import pandas as pd
import time
from datetime import datetime


# -----------------------------
# Configuration
# -----------------------------

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MIN_CONTOUR_AREA = 3000
THRESHOLD_VALUE = 30

OUTPUT_FILE = "motion_data_log.csv"

WINDOW_NAME = "WSL Motion Tracking"


# -----------------------------
# Initialize camera
# -----------------------------

video_capture = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2
)

if not video_capture.isOpened():
    video_capture.release()

    # Fallback to OpenCV's default backend
    video_capture = cv2.VideoCapture(CAMERA_INDEX)

if not video_capture.isOpened():
    raise RuntimeError(
        "Could not open camera. If running under WSL2, "
        "verify webcam USB passthrough and check /dev/video0."
    )


# Request MJPEG format, which may resolve green or
# incorrectly decoded frames on some webcams.
video_capture.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

video_capture.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

video_capture.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

# Reduce buffering where supported.
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera initialized.")
print("Backend:", video_capture.getBackendName())
print(
    "Reported resolution:",
    video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
    "x",
    video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print("Preparing camera...")


# -----------------------------
# Create display window
# -----------------------------

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1280,
    720
)


# -----------------------------
# Camera warm-up
# -----------------------------

frame = None

for i in range(30):
    success, frame = video_capture.read()

    if not success or frame is None:
        print("Waiting for camera frame...")
        time.sleep(0.1)
        continue

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        video_capture.release()
        cv2.destroyAllWindows()
        raise SystemExit

    time.sleep(0.03)


if frame is None:
    video_capture.release()
    cv2.destroyAllWindows()

    raise RuntimeError(
        "Camera opened, but no frames could be captured."
    )


print("Camera ready.")
print("Tracking started.")
print("Press Q to stop and save data.")
print("Press B to recalibrate the background.")


# -----------------------------
# Variables
# -----------------------------

motion_log = []

background = None


# -----------------------------
# Background calibration
# -----------------------------

def calibrate_background(current_frame):
    """Create a grayscale background reference."""

    gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (21, 21),
        0
    )

    return gray


# Use the most recent camera frame.
background = calibrate_background(frame)


# -----------------------------
# Main tracking loop
# -----------------------------

try:
    while True:

        # -----------------------------
        # Read full-color camera frame
        # -----------------------------

        success, frame = video_capture.read()

        if not success or frame is None:
            print("Failed to read frame.")
            time.sleep(0.05)
            continue

        # Keep the original color frame for display.
        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # Create a separate grayscale copy for detection.
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (21, 21),
            0
        )

        # -----------------------------
        # Detect differences
        # -----------------------------

        difference = cv2.absdiff(
            background,
            gray
        )

        threshold = cv2.threshold(
            difference,
            THRESHOLD_VALUE,
            255,
            cv2.THRESH_BINARY
        )[1]

        threshold = cv2.dilate(
            threshold,
            None,
            iterations=2
        )

        # -----------------------------
        # Find moving objects
        # -----------------------------

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]

        movement_detected = False

        # -----------------------------
        # Process movement
        # -----------------------------

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < MIN_CONTOUR_AREA:
                continue

            movement_detected = True

            x, y, w, h = cv2.boundingRect(contour)

            center_x = x + (w // 2)
            center_y = y + (h // 2)

            # Record movement data.
            motion_log.append({
                "Timestamp": timestamp,
                "X_Coord": center_x,
                "Y_Coord": center_y,
                "Area_Size": round(area, 2)
            })

            # Draw a center point on the original image.
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

        # -----------------------------
        # Display status
        # -----------------------------

        if movement_detected:
            status = "MOVEMENT DETECTED"
            status_color = (0, 0, 255)
        else:
            status = "NO MOVEMENT"
            status_color = (255, 255, 255)

        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        # -----------------------------
        # Display full camera feed
        # -----------------------------

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # -----------------------------
        # Keyboard controls
        # -----------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("b"):
            background = calibrate_background(frame)
            print("Background recalibrated.")

finally:

    # -----------------------------
    # Release camera
    # -----------------------------

    video_capture.release()
    cv2.destroyAllWindows()

    # -----------------------------
    # Save movement data
    # -----------------------------

    if motion_log:
        df = pd.DataFrame(motion_log)

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print()
        print(f"Motion data saved to: {OUTPUT_FILE}")
        print(f"Records collected: {len(df)}")

    else:
        print()
        print("No motion was recorded.")
