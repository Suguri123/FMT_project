from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import av
import cv2
import mediapipe as mp


class MotionCaptureSession:
    # Thread-safe MediaPipe processor for one Streamlit browser session.
    def __init__(self, data_dir: str | Path, max_saved_files: int = 10) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_saved_files = max_saved_files

        self._state_lock = threading.Lock()
        self._processor_lock = threading.Lock()
        self._holistic = None
        self._countdown_started_at: float | None = None
        self._recording_started_at: float | None = None
        self._record_duration = 5.0
        self._base_name = 'motion'
        self._frames: list[dict] = []
        self._completed_filename: str | None = None
        self._last_error: str | None = None

        self._mp_holistic = mp.solutions.holistic
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils

    def start_recording(self, duration: float, base_name: str) -> bool:
        with self._state_lock:
            if self._countdown_started_at is not None or self._recording_started_at is not None:
                return False

            self._record_duration = float(duration)
            self._base_name = Path(base_name).name or f'motion_{int(time.time())}'
            self._frames = []
            self._completed_filename = None
            self._last_error = None
            self._countdown_started_at = time.time()
            return True

    def status(self) -> dict:
        now = time.time()
        with self._state_lock:
            countdown_remaining = None
            if self._countdown_started_at is not None:
                countdown_remaining = max(0.0, 3.0 - (now - self._countdown_started_at))

            recording_remaining = None
            if self._recording_started_at is not None:
                recording_remaining = max(
                    0.0,
                    self._record_duration - (now - self._recording_started_at),
                )

            return {
                'countdown_remaining': countdown_remaining,
                'recording_remaining': recording_remaining,
                'last_error': self._last_error,
            }

    def take_completed_filename(self) -> str | None:
        with self._state_lock:
            filename = self._completed_filename
            self._completed_filename = None
            return filename

    def stop_stream(self) -> None:
        with self._processor_lock:
            with self._state_lock:
                self._countdown_started_at = None
                self._recording_started_at = None
                self._frames = []
                holistic = self._holistic
                self._holistic = None

            if holistic is not None:
                holistic.close()

    def process_video_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = cv2.flip(frame.to_ndarray(format='bgr24'), 1)
        completed_recording: tuple[list[dict], str] | None = None

        try:
            with self._processor_lock:
                holistic = self._get_holistic()
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_image.flags.writeable = False
                results = holistic.process(rgb_image)
                self._draw_landmarks(image, results)

                now = time.time()
                frame_data = self._serialize_results(results, now)
                overlay_text, overlay_color = self._update_recording_state(now, frame_data)

                cv2.putText(
                    image,
                    overlay_text,
                    (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    overlay_color,
                    2,
                )

                with self._state_lock:
                    if self._recording_started_at is None and self._frames:
                        completed_recording = (self._frames, self._base_name)
                        self._frames = []

            if completed_recording is not None:
                frames, base_name = completed_recording
                filename = self._save_recording(frames, base_name)
                with self._state_lock:
                    self._completed_filename = filename
                    self._last_error = None

        except Exception as exc:
            with self._state_lock:
                self._last_error = str(exc)
                self._countdown_started_at = None
                self._recording_started_at = None
                self._frames = []
            cv2.putText(
                image,
                'TRACKING ERROR',
                (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(image, format='bgr24')

    def _get_holistic(self):
        with self._state_lock:
            if self._holistic is None:
                self._holistic = self._mp_holistic.Holistic(
                    static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=1,
                    smooth_landmarks=False,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                )
            return self._holistic

    def _draw_landmarks(self, image, results) -> None:
        if results.pose_landmarks:
            self._mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self._mp_holistic.POSE_CONNECTIONS,
            )
        if results.left_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
            )
        if results.right_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
            )

    @staticmethod
    def _landmarks_to_dict(landmarks) -> list[dict]:
        if landmarks is None:
            return []
        return [
            {'id': index, 'x': point.x, 'y': point.y, 'z': point.z}
            for index, point in enumerate(landmarks.landmark)
        ]

    def _serialize_results(self, results, timestamp: float) -> dict:
        pose = self._landmarks_to_dict(results.pose_landmarks)
        left_hand = self._landmarks_to_dict(results.left_hand_landmarks)
        right_hand = self._landmarks_to_dict(results.right_hand_landmarks)
        hands = [hand for hand in (left_hand, right_hand) if hand]
        return {
            'time': timestamp,
            'pose': pose,
            'left_hand': left_hand,
            'right_hand': right_hand,
            'hands': hands,
        }

    def _update_recording_state(
        self,
        now: float,
        frame_data: dict,
    ) -> tuple[str, tuple[int, int, int]]:
        with self._state_lock:
            self._last_error = None
            if self._countdown_started_at is not None:
                countdown_remaining = 3.0 - (now - self._countdown_started_at)
                if countdown_remaining > 0:
                    seconds = max(1, math.ceil(countdown_remaining))
                    return (f'Starting in {seconds}...', (0, 255, 255))

                self._countdown_started_at = None
                self._recording_started_at = now

            if self._recording_started_at is not None:
                self._frames.append(frame_data)
                remaining = self._record_duration - (now - self._recording_started_at)
                if remaining <= 0:
                    self._recording_started_at = None
                return (f'Recording: {max(0.0, remaining):.1f}s', (0, 0, 255))

            tracking = bool(
                frame_data['pose']
                or frame_data['left_hand']
                or frame_data['right_hand']
            )
            label = 'ACTIVE TRACKING' if tracking else 'SEARCHING BODY/HAND'
            return (label, (0, 255, 0))

    def _save_recording(self, frames: list[dict], base_name: str) -> str:
        path = self._unique_path(base_name)
        with path.open('w', encoding='utf-8') as output:
            json.dump(frames, output, ensure_ascii=False)

        saved_files = sorted(
            self.data_dir.glob('*.json'),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for old_file in saved_files[self.max_saved_files :]:
            try:
                old_file.unlink()
            except OSError:
                pass

        return path.name

    def _unique_path(self, base_name: str) -> Path:
        candidate = self.data_dir / f'{base_name}.json'
        if not candidate.exists():
            return candidate

        index = 2
        while True:
            candidate = self.data_dir / f'{base_name}_{index}.json'
            if not candidate.exists():
                return candidate
            index += 1
