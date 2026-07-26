"""Physical-AI hand pose classifier.

The model learns the relationship between normalized hand-joint coordinates and
the number of raised fingers.  Training labels are supplied by the user, so
the model is not pretending that filename guesses are ground truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


MODEL_VERSION = 1
FEATURE_SIZE = 63  # 21 landmarks x (x, y, z)


def hand_features(hand_points: list[dict]) -> np.ndarray | None:
    """Return translation/scale-normalized landmark features for one hand."""
    points = {int(point.get("id", -1)): point for point in hand_points or []}
    if any(index not in points for index in range(21)):
        return None

    axes = ("x", "y", "z")
    wrist = np.array([points[0][axis] for axis in axes], dtype=np.float32)
    middle_mcp = np.array([points[9][axis] for axis in axes], dtype=np.float32)
    scale = float(np.linalg.norm(middle_mcp - wrist))
    if scale < 1e-6:
        return None

    values = []
    for index in range(21):
        point = points[index]
        values.extend(
            (float(point[axis]) - float(wrist[axis_index])) / scale
            for axis_index, axis in enumerate(axes)
        )
    return np.asarray(values, dtype=np.float32)


def samples_from_frames(frames: Iterable[dict]) -> list[np.ndarray]:
    samples = []
    for frame in frames:
        for hand_key in ("left_hand", "right_hand"):
            features = hand_features(frame.get(hand_key) or [])
            if features is not None:
                samples.append(features)
    return samples


class PhysicalAIModel:
    def __init__(self, model_path: str | Path, class_map_path: str | Path | None = None):
        self.model_path = Path(model_path)
        if class_map_path:
            self.class_map_path = Path(class_map_path)
        else:
            # 1. 모델과 같은 폴더에서 우선 조회
            candidate = self.model_path.with_name('physical_ai_class_led_counts.json')
            if not candidate.exists():
                # 2. 루트 폴더(현재 작업 공간)에서 2순위 조회
                import os
                root_candidate = Path(os.getcwd()) / 'physical_ai_class_led_counts.json'
                if root_candidate.exists():
                    candidate = root_candidate
                else:
                    # 3. 모델 폴더의 상위 폴더(physical_ai_models 상위)에서 3순위 조회
                    parent_candidate = self.model_path.parent.parent / 'physical_ai_class_led_counts.json'
                    if parent_candidate.exists():
                        candidate = parent_candidate
            self.class_map_path = candidate
        self.model = None
        self.class_led_counts = {}
        self.load()

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        try:
            if self.class_map_path.exists():
                self.class_led_counts = json.loads(self.class_map_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            self.class_led_counts = {}
        if not self.model_path.exists():
            self.model = None
            return False
        try:
            bundle = joblib.load(self.model_path)
            if bundle.get("version") != MODEL_VERSION:
                self.model = None
                return False
            self.model = bundle["model"]
            return True
        except Exception:
            self.model = None
            return False

    def predict_count(self, hand_points: list[dict]) -> int | None:
        prediction = self.predict_count_with_confidence(hand_points)
        return prediction[0] if prediction is not None else None

    def predict_label_with_confidence(self, hand_points: list[dict]) -> tuple[str, float] | None:
        features = hand_features(hand_points)
        if not self.available or features is None:
            return None
        row = features.reshape(1, -1)
        label = str(self.model.predict(row)[0])
        confidence = float(max(self.model.predict_proba(row)[0]))
        return label, confidence

    def led_count_for_label(self, label: str) -> int | None:
        if label in self.class_led_counts:
            return max(0, min(int(self.class_led_counts[label]), 8))
        match = re.search(r'(?<!\d)([0-8])(?!\d)', label)
        return int(match.group(1)) if match else None

    def predict_count_with_confidence(
        self, hand_points: list[dict]
    ) -> tuple[int, float] | None:
        prediction = self.predict_label_with_confidence(hand_points)
        if prediction is None:
            return None
        label, confidence = prediction
        count = self.led_count_for_label(label)
        if count is None:
            return None
        return count, confidence


def train_model(
    samples: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
) -> dict:
    if len(samples) < 10:
        raise ValueError("학습 샘플이 너무 적습니다. 최소 10개 이상의 손 관절 샘플이 필요합니다.")
    if len(np.unique(labels)) < 2:
        raise ValueError("서로 다른 손가락 개수 라벨이 2종류 이상 필요합니다.")

    model = RandomForestClassifier(
        n_estimators=180,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(samples, labels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": MODEL_VERSION, "model": model}, output_path)
    return {
        "samples": int(len(samples)),
        "classes": sorted(str(value) for value in np.unique(labels)),
        "path": str(output_path),
    }
