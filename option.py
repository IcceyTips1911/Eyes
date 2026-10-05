import cv2
import pandas as pd
import time
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MIN_CONTOUR_AREA = 3000
THRESHOLD_VALUE = 30

OUTPUT_FILE = "motion_data_log.csv"


# ============================================================
# INITIALIZE CAMERA
# ============================================================

print()
print("========================================")
print("       WSL MOTION TRACKING")
print("========================================")
print()

print("Initializing camera...")

# Use V4L2 backend under Linux/WSL
video_capture = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2
)

if not video_capture.isOpened():

    raise RuntimeError(
        "\nCould not open camera.\n\n"
        "If running under WSL2, make sure your webcam "
        "is attached using usbipd.\n"
    )


# ============================================================
# CAMERA FORMAT
# ============================================================

# Request MJPEG format.
#
# Some webcams produce corrupted/green frames when OpenCV
# receives an unsupported raw pixel format.
#
# MJPEG is generally a safer format for USB webcams.

video_capture.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)


# ============================================================
# CAMERA RESOLUTION
# ============================================================

video_capture.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

video_capture.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


# ============================================================
# CAMERA INFORMATION
# ============================================================

actual_width = video_capture.get(
    cv2.CAP_PROP_FRAME_WIDTH
)

actual_height = video_capture.get(
    cv2.CAP_PROP_FRAME_HEIGHT
)

backend = video_capture.getBackendName()


print()
print("Camera initialized.")
print()
print("Camera information:")
print("----------------------------------------")
print(f"Requested resolution : {FRAME_WIDTH} x {FRAME_HEIGHT}")
print(
    f"Actual resolution    : "
    f"{int(actual_width)} x {int(actual_height)}"
)
print(f"Backend              : {backend}")
print("----------------------------------------")
print()


# ============================================================
# CAMERA WARM-UP
# ============================================================

print("Preparing camera...")

for i in range(30):

    success, frame = video_capture.read()

    if not success:

        print(
            "Failed to read camera during startup."
        )

        break


    # --------------------------------------------------------
    # Display camera while warming up
    # --------------------------------------------------------

    cv2.imshow(
        "WSL Motion Tracking",
        frame
    )


    # --------------------------------------------------------
    # Allow user to quit
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        video_capture.release()

        cv2.destroyAllWindows()

        exit()


    time.sleep(0.03)


print("Camera ready.")
print("Tracking started.")
print()
print("Press 'q' to stop and save data.")
print()


# ============================================================
# VARIABLES
# ============================================================

motion_log = []

background = None


# ============================================================
# CREATE BACKGROUND FRAME
# ============================================================

success, frame = video_capture.read()


if not success:

    video_capture.release()

    cv2.destroyAllWindows()

    raise RuntimeError(
        "Could not capture background frame."
    )


# ============================================================
# RESIZE FRAME
# ============================================================

frame = cv2.resize(
    frame,
    (FRAME_WIDTH, FRAME_HEIGHT)
)


# ============================================================
# CONVERT BACKGROUND TO GRAYSCALE
# ============================================================

background_gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# BLUR BACKGROUND
# ============================================================

background_gray = cv2.GaussianBlur(
    background_gray,
    (21, 21),
    0
)


background = background_gray.copy()


# ============================================================
# MAIN TRACKING LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # READ CAMERA FRAME
        # ====================================================

        success, frame = video_capture.read()


        if not success:

            print(
                "Failed to read frame."
            )

            break


        # ====================================================
        # RESIZE FRAME
        # ====================================================

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )


        # ====================================================
        # CONVERT FRAME TO GRAYSCALE
        # ====================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ====================================================
        # REDUCE IMAGE NOISE
        # ====================================================

        gray = cv2.GaussianBlur(
            gray,
            (21, 21),
            0
        )


        # ====================================================
        # DETECT DIFFERENCES
        # ====================================================

        difference = cv2.absdiff(
            background,
            gray
        )


        # ====================================================
        # THRESHOLD MOVEMENT
        # ====================================================

        threshold = cv2.threshold(
            difference,
            THRESHOLD_VALUE,
            255,
            cv2.THRESH_BINARY
        )[1]


        # ====================================================
        # REMOVE SMALL GAPS / NOISE
        # ====================================================

        threshold = cv2.dilate(
            threshold,
            None,
            iterations=2
        )


        # ====================================================
        # FIND MOVEMENT CONTOURS
        # ====================================================

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]


        movement_detected = False


        # ====================================================
        # PROCESS MOVEMENT
        # ====================================================

        for contour in contours:

            # ------------------------------------------------
            # Calculate contour area
            # ------------------------------------------------

            area = cv2.contourArea(
                contour
            )


            # ------------------------------------------------
            # Ignore small movement
            # ------------------------------------------------

            if area < MIN_CONTOUR_AREA:

                continue


            movement_detected = True


            # ------------------------------------------------
            # Get bounding box
            # ------------------------------------------------

            x, y, w, h = cv2.boundingRect(
                contour
            )


            # ------------------------------------------------
            # Calculate center
            # ------------------------------------------------

            center_x = x + (w // 2)

            center_y = y + (h // 2)


            # ------------------------------------------------
            # Save tracking data
            # ------------------------------------------------

            motion_log.append({

                "Timestamp": timestamp,

                "X_Coord": center_x,

                "Y_Coord": center_y,

                "Area_Size": round(
                    area,
                    2
                )

            })


            # ------------------------------------------------
            # Draw center point
            # ------------------------------------------------

            cv2.circle(

                frame,

                (
                    center_x,
                    center_y
                ),

                5,

                (0, 0, 255),

                -1
            )


        # ====================================================
        # DISPLAY STATUS
        # ====================================================

        if movement_detected:

            cv2.putText(

                frame,

                "MOVEMENT DETECTED",

                (
                    10,
                    30
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2
            )

        else:

            cv2.putText(

                frame,

                "NO MOVEMENT",

                (
                    10,
                    30
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )


        # ====================================================
        # DISPLAY FULL CAMERA
        # ====================================================

        cv2.imshow(

            "WSL Motion Tracking",

            frame

        )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


finally:

    # ========================================================
    # RELEASE CAMERA
    # ========================================================

    video_capture.release()


    # ========================================================
    # CLOSE OPENCV WINDOWS
    # ========================================================

    cv2.destroyAllWindows()


    # ========================================================
    # SAVE MOTION DATA
    # ========================================================

    if motion_log:

        df = pd.DataFrame(
            motion_log
        )


        df.to_csv(

            OUTPUT_FILE,

            index=False

        )


        print()
        print("========================================")
        print("Motion tracking stopped.")
        print("========================================")
        print()

        print(
            f"Motion data saved to: "
            f"{OUTPUT_FILE}"
        )

        print(
            f"Records collected: "
            f"{len(df)}"
        )

        print()

    else:

        print()
        print("========================================")
        print("Motion tracking stopped.")
        print("========================================")
        print()

        print(
            "No motion was recorded."
        )

        print()
