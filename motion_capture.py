from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import av
import cv2
import mediapipe as mp

try:
    import serial
except ImportError:
    serial = None


FIST_FINGER_IDS = {
    'index': {'mcp': 5, 'pip': 6, 'tip': 8},
    'middle': {'mcp': 9, 'pip': 10, 'tip': 12},
    'ring': {'mcp': 13, 'pip': 14, 'tip': 16},
    'pinky': {'mcp': 17, 'pip': 18, 'tip': 20},
}

LEFT_FIST_THRESHOLD = 0.60
ARDUINO_BAUD_RATE = 9600


class MotionCaptureSession:
    # Thread-safe MediaPipe processor for one Streamlit browser session.
    def __init__(self, data_dir: str | Path, max_saved_files: int = 10) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_saved_files = max_saved_files

        self._state_lock = threading.Lock()
        self._processor_lock = threading.Lock()
        self._hands_processor = None
        self._countdown_started_at: float | None = None
        self._recording_started_at: float | None = None
        self._record_duration = 5.0
        self._base_name = 'motion'
        self._frames: list[dict] = []
        self._completed_filename: str | None = None
        self._last_error: str | None = None
        self._arduino_enabled = False
        self._arduino_port = 'COM3'
        self._arduino = None
        self._arduino_last_command: str | None = None
        self._arduino_last_sent_at = 0.0
        self._arduino_status = '실시간 Arduino 전송 꺼짐'
        self._arduino_error: str | None = None

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
                'arduino_enabled': self._arduino_enabled,
                'arduino_status': self._arduino_status,
                'arduino_error': self._arduino_error,
            }

    def configure_arduino_realtime(self, enabled: bool, port: str) -> None:
        port = (port or 'COM3').strip()
        with self._processor_lock:
            with self._state_lock:
                port_changed = port != self._arduino_port
                self._arduino_enabled = enabled
                self._arduino_port = port
                if not enabled:
                    self._arduino_status = '실시간 Arduino 전송 꺼짐'
                    self._arduino_error = None

            if not enabled or port_changed:
                self._close_arduino()

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
                hands_processor = self._hands_processor
                self._hands_processor = None
                arduino = self._arduino
                self._arduino = None
                self._arduino_last_command = None

            if hands_processor is not None:
                hands_processor.close()
            if arduino is not None:
                arduino.close()

    def process_video_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = cv2.flip(frame.to_ndarray(format='bgr24'), 1)
        completed_recording: tuple[list[dict], str] | None = None

        try:
            with self._processor_lock:
                hands_processor = self._get_hands_processor()
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_image.flags.writeable = False
                results = hands_processor.process(rgb_image)
                self._draw_landmarks(image, results)

                now = time.time()
                frame_data = self._serialize_results(results, now)
                self._update_arduino_from_frame(frame_data, now)
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

    def _get_hands_processor(self):
        with self._state_lock:
            if self._hands_processor is None:
                self._hands_processor = self._mp_hands.Hands(
                    static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=1,
                    max_num_hands=2,
                )
            return self._hands_processor

    def _get_arduino(self):
        if serial is None:
            raise RuntimeError('pyserial이 설치되어 있지 않습니다. pip install pyserial을 실행하세요.')

        if self._arduino is None or not self._arduino.is_open:
            self._arduino = serial.Serial(self._arduino_port, ARDUINO_BAUD_RATE, timeout=1)
            time.sleep(2)
            self._arduino_last_command = None
        return self._arduino

    def _close_arduino(self) -> None:
        arduino = self._arduino
        self._arduino = None
        self._arduino_last_command = None
        if arduino is not None:
            try:
                arduino.close()
            except Exception:
                pass

    def _draw_landmarks(self, image, results) -> None:
        for landmarks in results.multi_hand_landmarks or []:
            self._mp_drawing.draw_landmarks(
                image,
                landmarks,
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
        left_hand = []
        right_hand = []
        hands = []

        hand_landmarks = results.multi_hand_landmarks or []
        handedness_items = results.multi_handedness or []
        for index, landmarks in enumerate(hand_landmarks):
            hand_data = self._landmarks_to_dict(landmarks)
            hands.append(hand_data)

            label = ''
            if index < len(handedness_items) and handedness_items[index].classification:
                label = handedness_items[index].classification[0].label

            if label == 'Left' and not left_hand:
                left_hand = hand_data
            elif label == 'Right' and not right_hand:
                right_hand = hand_data
            elif not left_hand:
                left_hand = hand_data
            elif not right_hand:
                right_hand = hand_data

        return {
            'time': timestamp,
            'pose': [],
            'left_hand': left_hand,
            'right_hand': right_hand,
            'hands': hands,
        }

    @staticmethod
    def _landmark_distance(first, second) -> float:
        return (
            (first['x'] - second['x']) ** 2
            + (first['y'] - second['y']) ** 2
            + (first['z'] - second['z']) ** 2
        ) ** 0.5

    def _is_left_hand_fist(self, frame_data: dict) -> bool:
        hand_points = frame_data.get('left_hand') or []
        landmarks = {point['id']: point for point in hand_points}
        wrist = landmarks.get(0)
        middle_mcp = landmarks.get(9)
        if not wrist or not middle_mcp:
            return False

        palm_size = self._landmark_distance(wrist, middle_mcp)
        if palm_size <= 0:
            return False

        folded_count = 0
        checked_count = 0
        for ids in FIST_FINGER_IDS.values():
            mcp = landmarks.get(ids['mcp'])
            pip = landmarks.get(ids['pip'])
            tip = landmarks.get(ids['tip'])
            if not mcp or not pip or not tip:
                continue

            checked_count += 1
            tip_to_wrist = self._landmark_distance(tip, wrist)
            pip_to_wrist = self._landmark_distance(pip, wrist)
            tip_to_mcp = self._landmark_distance(tip, mcp)
            if tip_to_wrist <= pip_to_wrist * 1.02 and tip_to_mcp < palm_size * 0.90:
                folded_count += 1

        if checked_count < 4:
            return False
        return folded_count / checked_count >= LEFT_FIST_THRESHOLD

    def _update_arduino_from_frame(self, frame_data: dict, now: float) -> None:
        with self._state_lock:
            enabled = self._arduino_enabled
            if not enabled:
                return

        command = 'ON' if self._is_left_hand_fist(frame_data) else 'OFF'
        if command == self._arduino_last_command and now - self._arduino_last_sent_at < 0.5:
            return

        try:
            arduino = self._get_arduino()
            arduino.write(f'{command}\n'.encode('ascii'))
            arduino.flush()
            self._arduino_last_command = command
            self._arduino_last_sent_at = now
            with self._state_lock:
                self._arduino_error = None
                self._arduino_status = f'{self._arduino_port}로 {command} 전송 중'
        except Exception as exc:
            self._close_arduino()
            with self._state_lock:
                self._arduino_error = str(exc)
                self._arduino_status = 'Arduino 전송 오류'

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
                frame_data['left_hand']
                or frame_data['right_hand']
            )
            label = 'ACTIVE HAND TRACKING' if tracking else 'SEARCHING HAND'
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
