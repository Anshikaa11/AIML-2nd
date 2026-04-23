"""
Face Attendance System Backend (OpenCV-only version)

Uses OpenCV's Haar Cascade for face detection and LBPH
(Local Binary Patterns Histograms) for face recognition.
No dlib or face_recognition dependency required.

Auto-captures faces during registration — no key presses needed.
"""

import os
import time
import datetime
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    raise ImportError(
        "OpenCV is required. Install it with:\n"
        "  !python -m pip install opencv-contrib-python"
    )

# Check that the face module is available (from opencv-contrib-python)
if not hasattr(cv2, "face"):
    raise ImportError(
        "opencv-contrib-python is required for face recognition.\n"
        "Install it with:\n"
        "  !python -m pip install opencv-contrib-python"
    )


class FaceAttendanceSystem:
    """
    A complete face-attendance back-end that the Jupyter dashboard calls.

    Directory layout created under `data_dir`:
        data/
        ├── users.csv            # columns: name, registered_at, image_dir
        ├── attendance.csv       # columns: name, date, timestamp
        ├── faces/               # cropped grayscale face images per user
        │   └── <username>/
        └── model.yml            # trained LBPH recognizer model
    """

    FACE_CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    NUM_CAPTURES = 15          # number of face samples to grab during registration
    CONFIDENCE_THRESHOLD = 80  # lower = stricter match (LBPH distance)

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.faces_dir = os.path.join(data_dir, "faces")
        self.users_csv = os.path.join(data_dir, "users.csv")
        self.attendance_csv = os.path.join(data_dir, "attendance.csv")
        self.model_path = os.path.join(data_dir, "model.yml")

        # Ensure directories exist
        os.makedirs(self.faces_dir, exist_ok=True)

        # Haar cascade detector
        self.face_cascade = cv2.CascadeClassifier(self.FACE_CASCADE_FILE)

        # LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._model_loaded = False

        # DataFrames the notebook reads directly
        self.users_df = pd.DataFrame()
        self.attendance_df = pd.DataFrame()

        # label <-> name mapping
        self._label_to_name: dict[int, str] = {}

        # Bootstrap
        self._init_csvs()
        self.load_data()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _init_csvs(self):
        """Create CSV files with headers if they don't exist yet."""
        if not os.path.exists(self.users_csv):
            pd.DataFrame(columns=["name", "registered_at", "image_dir"]).to_csv(
                self.users_csv, index=False
            )
        if not os.path.exists(self.attendance_csv):
            pd.DataFrame(columns=["name", "date", "timestamp"]).to_csv(
                self.attendance_csv, index=False
            )

    def load_data(self):
        """Reload CSVs and retrain / reload the LBPH model."""
        self.users_df = pd.read_csv(self.users_csv)
        self.attendance_df = pd.read_csv(self.attendance_csv)

        # Rebuild label map
        self._label_to_name.clear()
        for idx, row in self.users_df.iterrows():
            self._label_to_name[int(idx)] = row["name"]

        # Reload trained model if it exists
        self._model_loaded = False
        if os.path.exists(self.model_path) and not self.users_df.empty:
            try:
                self.recognizer.read(self.model_path)
                self._model_loaded = True
            except Exception:
                self._train_model()

    def _train_model(self):
        """Train the LBPH recognizer from saved face images."""
        faces = []
        labels = []

        for label, row in self.users_df.iterrows():
            user_dir = row["image_dir"]
            if not os.path.isdir(user_dir):
                continue
            for img_name in os.listdir(user_dir):
                img_path = os.path.join(user_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(int(label))

        if faces:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.write(self.model_path)
            self._model_loaded = True
        else:
            self._model_loaded = False

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    def register_user(self, name: str) -> bool:
        """
        Open the webcam and AUTO-CAPTURE face samples.
        No key presses required — faces are captured automatically
        whenever a face is detected. A preview window shows progress.
        Press 'q' in the webcam window to cancel (if window is visible).
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam.")
            return False

        # Let the camera warm up
        time.sleep(1)

        # Prepare user directory
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        user_dir = os.path.join(self.faces_dir, safe_name)
        os.makedirs(user_dir, exist_ok=True)

        count = 0
        max_frames = 300  # safety limit (~10 seconds at 30fps)
        frame_num = 0

        print(
            f"📸 Webcam opened. Auto-capturing {self.NUM_CAPTURES} face samples for '{name}'.\n"
            f"   Look at the camera and slowly move your head.\n"
            f"   A window may appear in your taskbar — you can ignore it."
        )

        while count < self.NUM_CAPTURES and frame_num < max_frames:
            ret, frame = cap.read()
            frame_num += 1
            if not ret:
                print("❌ Failed to read from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100)
            )

            for (x, y, w, h) in detected:
                face_roi = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_roi, (200, 200))

                img_path = os.path.join(user_dir, f"{count}.jpg")
                cv2.imwrite(img_path, face_resized)
                count += 1
                print(f"   Captured {count}/{self.NUM_CAPTURES}...")

                # Draw rectangle on preview
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Captured {count}/{self.NUM_CAPTURES}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if count >= self.NUM_CAPTURES:
                    break

            # Show preview (may or may not be visible depending on environment)
            try:
                cv2.imshow("Registering face...", frame)
                key = cv2.waitKey(200) & 0xFF
                if key == ord("q"):
                    print("🚫 Registration cancelled by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
            except Exception:
                # If imshow fails (e.g. headless), just add a delay
                time.sleep(0.2)

        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if count == 0:
            print("⚠️ No face detected. Try again in better lighting.")
            return False

        # Append to CSV
        new_row = pd.DataFrame(
            [
                {
                    "name": name,
                    "registered_at": datetime.datetime.now().isoformat(),
                    "image_dir": user_dir,
                }
            ]
        )
        new_row.to_csv(self.users_csv, mode="a", header=False, index=False)

        # Retrain model with the new user
        self.load_data()
        self._train_model()

        print(f"✅ Registered '{name}' with {count} face samples.")
        return True

    def remove_user(self, name: str) -> bool:
        """Remove a user by name (case-insensitive) and retrain the model."""
        self.load_data()
        mask = self.users_df["name"].str.lower() == name.lower()
        if not mask.any():
            print(f"⚠️ User '{name}' not found.")
            return False

        # Delete face image directories
        import shutil

        for img_dir in self.users_df.loc[mask, "image_dir"]:
            if os.path.isdir(img_dir):
                shutil.rmtree(img_dir)

        self.users_df = self.users_df[~mask]
        self.users_df.to_csv(self.users_csv, index=False)

        # Retrain without the removed user
        self.load_data()
        self._train_model()

        print(f"✅ User '{name}' removed.")
        return True

    # ------------------------------------------------------------------
    # Live tracking
    # ------------------------------------------------------------------

    def run_tracking(self):
        """
        Open webcam, detect/recognise faces, and log attendance.
        Press 'q' in the webcam window to stop.
        Automatically stops after 60 seconds if no key press is possible.
        """
        self.load_data()

        if not self._model_loaded:
            print("⚠️ No trained model. Register at least one user first!")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam.")
            return

        today = datetime.date.today().isoformat()
        logged_today: set[str] = set()

        # Pre-load already logged names for today
        if not self.attendance_df.empty and "date" in self.attendance_df.columns:
            logged_today = set(
                self.attendance_df.loc[
                    self.attendance_df["date"] == today, "name"
                ]
            )

        print(
            "🎥 Tracking started.\n"
            "   Press 'q' in the webcam window to stop.\n"
            "   Auto-stops after 60 seconds."
        )

        start_time = time.time()
        timeout = 60  # auto-stop after 60 seconds

        while True:
            # Auto-stop safety
            if time.time() - start_time > timeout:
                print("⏱️ Auto-stopped after 60 seconds.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in detected:
                face_roi = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_roi, (200, 200))

                name = "Unknown"
                confidence = 0
                try:
                    label, confidence = self.recognizer.predict(face_resized)
                    if confidence < self.CONFIDENCE_THRESHOLD:
                        name = self._label_to_name.get(label, "Unknown")
                except Exception:
                    pass

                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(
                    frame, (x, y + h - 35), (x + w, y + h), color, cv2.FILLED
                )

                display_text = name if name == "Unknown" else f"{name} ({confidence:.0f})"
                cv2.putText(
                    frame, display_text, (x + 6, y + h - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1,
                )

                # Log attendance (once per person per day)
                if name != "Unknown" and name not in logged_today:
                    now = datetime.datetime.now()
                    row = pd.DataFrame(
                        [
                            {
                                "name": name,
                                "date": today,
                                "timestamp": now.isoformat(),
                            }
                        ]
                    )
                    row.to_csv(
                        self.attendance_csv, mode="a", header=False, index=False
                    )
                    logged_today.add(name)
                    print(f"[{now:%H:%M:%S}] ✅ Attendance marked for {name}")

            try:
                cv2.imshow("Face Attendance Tracking - press 'q' to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except Exception:
                time.sleep(0.03)

        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.load_data()
        print("⏹️ Tracking stopped.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_attendance(self):
        """Wipe the attendance log."""
        pd.DataFrame(columns=["name", "date", "timestamp"]).to_csv(
            self.attendance_csv, index=False
        )
        self.load_data()
        print("🗑️ Attendance logs cleared.")
