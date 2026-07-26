from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import av
import cv2
import mediapipe as mp

from physical_ai import PhysicalAIModel

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
RIGHT_DIRECTION_ANIMATION_SECONDS = 0.8


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
        self._arduino_port = 'COM22'
        self._arduino = None
        self._arduino_last_command: str | None = None
        self._arduino_last_sent_at = 0.0
        self._last_frame_command = 'OFF'
        self._last_frame_command_at = 0.0
        self._arduino_status = '실시간 Arduino 전송 꺼짐'
        self._arduino_error: str | None = None
        self._physical_ai = PhysicalAIModel(Path(__file__).with_name('physical_ai_model.joblib'))
        self._physical_ai_model_path = str(Path(__file__).with_name('physical_ai_model.joblib'))
        self._physical_ai_category = '전체'
        self._physical_ai_prediction = 'AI 모델 대기 중'
        self._right_direction_armed = False
        self._right_direction_started_at: float | None = None
        self._left_direction_started_at: float | None = None

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils

    def set_physical_ai_model(self, model_path: str | Path, category: str = '전체') -> bool:
        model_path = str(Path(model_path))
        if model_path == self._physical_ai_model_path:
            if category != self._physical_ai_category:
                self._physical_ai_prediction = 'AI 모델 대기 중'
            self._physical_ai_category = category or '전체'
            return self._physical_ai.available
        self._physical_ai = PhysicalAIModel(model_path)
        self._physical_ai_model_path = model_path
        self._physical_ai_category = category or '전체'
        self._physical_ai_prediction = 'AI 모델 대기 중'
        return self._physical_ai.available

    def start_recording(self, duration: float, base_name: str) -> bool:
        with self._state_lock:
            if self._countdown_started_at is not None or self._recording_started_at is not None:
                return False

            self._record_duration = float(duration)
            relative_name = Path(base_name)
            if relative_name.is_absolute() or '..' in relative_name.parts:
                relative_name = Path(f'motion_{int(time.time())}')
            self._base_name = str(relative_name).replace('\\', '/') or f'motion_{int(time.time())}'
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
                'physical_ai_available': self._physical_ai.available,
                'physical_ai_prediction': self._physical_ai_prediction,
            }

    def configure_arduino_realtime(self, enabled: bool, port: str) -> None:
        port = (port or 'COM22').strip()
        with self._processor_lock:
            with self._state_lock:
                port_changed = port != self._arduino_port
                self._arduino_enabled = enabled
                self._arduino_port = port
                if not enabled:
                    self._arduino_status = '실시간 Arduino 전송 꺼짐'
                    self._arduino_error = None
                    self._right_direction_armed = False
                    self._right_direction_started_at = None
                    self._left_direction_started_at = None

            if not enabled or port_changed:
                self._close_arduino()

    def send_arduino_command(self, port: str, command: str) -> None:
        port = (port or 'COM22').strip()
        command = command.strip().upper()
        bar_commands = {f'{prefix}{i}' for prefix in ('L', 'R') for i in range(1, 9)}
        bar_commands.update(
            f'L{left_count}R{right_count}'
            for left_count in range(0, 9)
            for right_count in range(0, 9)
            if left_count or right_count
        )
        if command not in {'L', 'R', 'LR', 'OFF', 'ON'} and command not in bar_commands:
            raise ValueError('Arduino command must be L, R, LR, L1-L8, R1-R8, ON, or OFF.')

        with self._processor_lock:
            with self._state_lock:
                if port != self._arduino_port:
                    self._arduino_port = port
                    self._close_arduino()

            self._write_arduino_command(command, time.time(), '전송 완료')

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
            self._arduino = serial.Serial(
                self._arduino_port,
                ARDUINO_BAUD_RATE,
                timeout=0.1,
                write_timeout=1.0,
            )
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

    def _write_arduino_command(self, command: str, now: float, status_suffix: str) -> None:
        arduino = self._get_arduino()
        arduino.write(f'{command}\n'.encode('ascii'))
        arduino.flush()
        with self._state_lock:
            self._arduino_last_command = command
            self._arduino_last_sent_at = now
            self._arduino_error = None
            self._arduino_status = f'{self._arduino_port}로 {command} {status_suffix}'

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
        unlabeled_hands = []

        for index, landmarks in enumerate(hand_landmarks):
            hand_data = self._landmarks_to_dict(landmarks)
            hands.append(hand_data)
            if not hand_data:
                continue

            label = ''
            if index < len(handedness_items) and handedness_items[index].classification:
                label = handedness_items[index].classification[0].label

            if label == 'Left' and not left_hand:
                left_hand = hand_data
            elif label == 'Right' and not right_hand:
                right_hand = hand_data
            else:
                center_x = sum(point['x'] for point in hand_data) / len(hand_data)
                unlabeled_hands.append((center_x, hand_data))

        if unlabeled_hands:
            unlabeled_hands.sort(key=lambda item: item[0])
            for center_x, hand_data in unlabeled_hands:
                if center_x < 0.5 and not left_hand:
                    left_hand = hand_data
                elif center_x >= 0.5 and not right_hand:
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

    def _is_hand_fist(self, hand_points: list[dict]) -> bool:
        landmarks = {point['id']: point for point in hand_points or []}
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

    def _is_hand_pointing_direction(self, hand_points: list[dict], direction: str) -> bool:
        if self._is_hand_fist(hand_points):
            return False

        landmarks = {point['id']: point for point in hand_points or []}
        wrist = landmarks.get(0)
        middle_mcp = landmarks.get(9)
        if not wrist or not middle_mcp:
            return False

        palm_size = self._landmark_distance(wrist, middle_mcp)
        if palm_size <= 0:
            return False

        pointing_count = 0
        checked_count = 0
        for ids in FIST_FINGER_IDS.values():
            mcp = landmarks.get(ids['mcp'])
            pip = landmarks.get(ids['pip'])
            tip = landmarks.get(ids['tip'])
            if not mcp or not pip or not tip:
                continue

            checked_count += 1
            x_delta = tip['x'] - wrist['x']
            points_side = x_delta > palm_size * 0.65 if direction == 'right' else x_delta < -palm_size * 0.65
            not_too_vertical = abs(tip['y'] - wrist['y']) < palm_size * 1.25
            tip_to_wrist = self._landmark_distance(tip, wrist)
            pip_to_wrist = self._landmark_distance(pip, wrist)
            tip_to_mcp = self._landmark_distance(tip, mcp)
            extended = tip_to_wrist > pip_to_wrist * 1.08 and tip_to_mcp > palm_size * 0.70

            if points_side and not_too_vertical and extended:
                pointing_count += 1

        return checked_count >= 4 and 1 <= pointing_count <= 2

    def _count_upward_fingers(self, hand_points: list[dict]) -> int:
        """Count non-thumb fingers that are extended upward."""
        landmarks = {point['id']: point for point in hand_points or []}
        wrist = landmarks.get(0)
        middle_mcp = landmarks.get(9)
        if not wrist or not middle_mcp:
            return 0

        palm_size = self._landmark_distance(wrist, middle_mcp)
        if palm_size <= 0:
            return 0

        raised_count = 0
        for ids in FIST_FINGER_IDS.values():
            mcp = landmarks.get(ids['mcp'])
            pip = landmarks.get(ids['pip'])
            tip = landmarks.get(ids['tip'])
            if not mcp or not pip or not tip:
                continue

            tip_to_wrist = self._landmark_distance(tip, wrist)
            pip_to_wrist = self._landmark_distance(pip, wrist)
            tip_to_mcp = self._landmark_distance(tip, mcp)
            extended = (
                tip_to_wrist > pip_to_wrist * 1.08
                and tip_to_mcp > palm_size * 0.70
                and tip['y'] < pip['y']
            )
            if extended:
                raised_count += 1

        return raised_count
    def _is_pointing_direction_gesture(self, frame_data: dict, direction: str) -> bool:
        return any(
            self._is_hand_pointing_direction(hand_points, direction)
            for hand_points in (
                frame_data.get('left_hand') or [],
                frame_data.get('right_hand') or [],
            )
        )

    def _direction_bar_command(self, now: float) -> str | None:
        active_direction = None
        started_at = None
        if self._right_direction_started_at is not None:
            active_direction = 'right'
            started_at = self._right_direction_started_at
        elif self._left_direction_started_at is not None:
            active_direction = 'left'
            started_at = self._left_direction_started_at
        else:
            return None

        elapsed = now - started_at
        if elapsed >= RIGHT_DIRECTION_ANIMATION_SECONDS:
            self._right_direction_started_at = None
            self._left_direction_started_at = None
            self._right_direction_armed = False
            return 'OFF'

        progress = min(max(elapsed / RIGHT_DIRECTION_ANIMATION_SECONDS, 0.0), 1.0)
        led_count = min(max(int(progress * 8) + 1, 1), 8)
        prefix = 'R' if active_direction == 'right' else 'L'
        return f'{prefix}{led_count}'
    def _clear_direction_animation(self) -> None:
        self._right_direction_started_at = None
        self._left_direction_started_at = None
        self._right_direction_armed = False

    def _fist_command(self, left_fist: bool, right_fist: bool) -> str:
        if left_fist and right_fist:
            return 'LR'
        if left_fist:
            return 'L'
        if right_fist:
            return 'R'
        return 'OFF'

    def _arduino_command_for_frame(self, frame_data: dict, now: float) -> str:
        left_points = frame_data.get('left_hand') or []
        right_points = frame_data.get('right_hand') or []
        if not left_points and not right_points:
            self._clear_direction_animation()
            # MediaPipe가 한두 프레임 손을 놓쳐도 직전의 유효한 클래스
            # 예측 결과를 '손 인식 대기 중'으로 덮어쓰지 않습니다.
            if self._physical_ai.available and not self._physical_ai_prediction.startswith('카테고리 ['):
                self._physical_ai_prediction = '손 인식 대기 중'
            return 'OFF'

        left_fist = self._is_hand_fist(left_points)
        right_fist = self._is_hand_fist(right_points)
        fist_command = self._fist_command(left_fist, right_fist)

        # 1. 학습된 AI 모델이 우선 순위를 가집니다. 모델 예측이 성공하면 그 결과를 반영합니다.
        if self._physical_ai.available:
            # 양손이 모두 검출된 경우
            if right_points and left_points:
                right_label_pred = self._physical_ai.predict_label_with_confidence(right_points)
                left_label_pred = self._physical_ai.predict_label_with_confidence(left_points)
                
                if right_label_pred is not None or left_label_pred is not None:
                    right_count_pred = self._physical_ai.predict_count_with_confidence(right_points)
                    left_count_pred = self._physical_ai.predict_count_with_confidence(left_points)
                    
                    right_count = right_count_pred[0] if right_count_pred is not None else 0
                    left_count = left_count_pred[0] if left_count_pred is not None else 0
                    
                    confidence_values = [p[1] for p in (right_count_pred, left_count_pred) if p is not None]
                    confidence = min(confidence_values) if confidence_values else 0.0
                    
                    right_label = right_label_pred[0] if right_label_pred else str(right_count)
                    left_label = left_label_pred[0] if left_label_pred else str(left_count)
                    
                    self._physical_ai_prediction = (
                        f'카테고리 [{self._physical_ai_category}] · '
                        f'왼손 클래스 [{left_label}] {left_count}개 / '
                        f'오른손 클래스 [{right_label}] {right_count}개 · '
                        f'신뢰도 {confidence:.0%}'
                    )
                    if right_count == 0 and left_count == 0:
                        return 'OFF'
                    return f'L{left_count}R{right_count}'

            # 오른손만 검출되거나 오른손 우선 예측 시도
            if right_points and (not left_points or right_fist or self._count_upward_fingers(right_points) >= 0):
                label_prediction = self._physical_ai.predict_label_with_confidence(right_points)
                if label_prediction is not None:
                    label, confidence = label_prediction
                    predicted = self._physical_ai.led_count_for_label(label)
                    if predicted is None:
                        predicted = 0
                    self._physical_ai_prediction = (
                        f'카테고리 [{self._physical_ai_category}] · '
                        f'오른손 클래스 [{label}] · LED {predicted}개 · '
                        f'신뢰도 {confidence:.0%}'
                    )
                    return f'R{predicted}' if predicted else 'OFF'

            # 왼손만 검출되거나 왼손 우선 예측 시도
            if left_points and (not right_points or left_fist or self._count_upward_fingers(left_points) >= 0):
                label_prediction = self._physical_ai.predict_label_with_confidence(left_points)
                if label_prediction is not None:
                    label, confidence = label_prediction
                    predicted = self._physical_ai.led_count_for_label(label)
                    if predicted is None:
                        predicted = 0
                    self._physical_ai_prediction = (
                        f'카테고리 [{self._physical_ai_category}] · '
                        f'왼손 클래스 [{label}] · LED {predicted}개 · '
                        f'신뢰도 {confidence:.0%}'
                    )
                    return f'L{predicted}' if predicted else 'OFF'

        # 2. AI 모델이 없거나 AI 예측을 생성하지 못한 프레임은 기존의 규칙 기반 로직으로 fallback합니다.
        # 손가락을 모두 접은 주먹은 0개로 간주해 모든 LED를 끕니다.
        if right_points and right_fist and not left_points and self._count_upward_fingers(right_points) == 0:
            self._physical_ai_prediction = '오른손 0개 · LED 꺼짐'
            return 'OFF'
        if left_points and left_fist and not right_points and self._count_upward_fingers(left_points) == 0:
            self._physical_ai_prediction = '왼손 0개 · LED 꺼짐'
            return 'OFF'
        if left_fist and right_fist and self._count_upward_fingers(left_points) == 0 and self._count_upward_fingers(right_points) == 0:
            self._physical_ai_prediction = '양손 0개 · LED 꺼짐'
            return 'OFF'

        # 손가락을 위로 편 개수만큼 해당 손의 LED를 켭니다.
        right_raised = self._count_upward_fingers(right_points)
        left_raised = self._count_upward_fingers(left_points)

        # 0개 우선 보장 fallback
        if right_points and not left_points and right_raised == 0:
            self._physical_ai_prediction = '오른손 0개 · LED 꺼짐'
            return 'OFF'
        if left_points and not right_points and left_raised == 0:
            self._physical_ai_prediction = '왼손 0개 · LED 꺼짐'
            return 'OFF'
        if right_points and left_points and right_raised == 0 and left_raised == 0:
            self._physical_ai_prediction = '양손 0개 · LED 꺼짐'
            return 'OFF'

        # 손가락 수 기반 LED 제어 fallback
        if right_raised > 0 and left_raised > 0:
            self._physical_ai_prediction = f'오른손 {right_raised}개 / 왼손 {left_raised}개 (AI 예측 스킵)'
            return f'L{min(left_raised, 8)}R{min(right_raised, 8)}'
        if right_raised > 0:
            self._physical_ai_prediction = f'오른손 {right_raised}개 (AI 예측 스킵)'
            return f'R{min(right_raised, 8)}'
        if left_raised > 0:
            self._physical_ai_prediction = f'왼손 {left_raised}개 (AI 예측 스킵)'
            return f'L{min(left_raised, 8)}'

        pointing_right = self._is_pointing_direction_gesture(frame_data, 'right')
        pointing_left = self._is_pointing_direction_gesture(frame_data, 'left')

        if self._right_direction_started_at is not None:
            return self._direction_bar_command(now) or fist_command

        if self._left_direction_started_at is not None:
            return self._direction_bar_command(now) or fist_command

        if pointing_right:
            self._right_direction_started_at = now
            return 'R1'
        if pointing_left:
            self._left_direction_started_at = now
            return 'L1'

        if left_fist or right_fist:
            self._right_direction_armed = True

        return fist_command

    def _update_arduino_from_frame(self, frame_data: dict, now: float) -> None:
        with self._state_lock:
            enabled = self._arduino_enabled

        # MediaPipe/AI와 시리얼 전송이 모든 영상 프레임을 막지 않도록 제한합니다.
        if now - self._last_frame_command_at >= 0.16:
            command = self._arduino_command_for_frame(frame_data, now)
            self._last_frame_command = command
            self._last_frame_command_at = now
        else:
            command = self._last_frame_command

        if not enabled:
            return
        if command == self._arduino_last_command and now - self._arduino_last_sent_at < 0.5:
            return

        try:
            self._write_arduino_command(command, now, '실시간 전송 중')
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

        return path.relative_to(self.data_dir).as_posix()

    def _unique_path(self, base_name: str) -> Path:
        candidate = self.data_dir / f'{base_name}.json'
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if not candidate.exists():
            return candidate

        index = 2
        while True:
            candidate = self.data_dir / f'{base_name}_{index}.json'
            if not candidate.exists():
                return candidate
            index += 1
