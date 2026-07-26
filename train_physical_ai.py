"""Train the hand-finger-count Physical AI model.

Example:
  python train_physical_ai.py --labels physical_ai_labels.json

physical_ai_labels.json format:
  {"손가락_0개.json": 0, "손가락_1개.json": 1, "손가락_2개.json": 2}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physical_ai import samples_from_frames, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="physical_ai_labels.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="physical_ai_model.joblib")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    samples = []
    targets = []
    for filename, label in labels.items():
        label = str(label).strip()
        if not label:
            raise ValueError(f"클래스명이 비어 있습니다: {filename}")
        path = data_dir / filename
        if not path.exists():
            print(f"건너뜀(파일 없음): {path}")
            continue
        frames = json.loads(path.read_text(encoding="utf-8"))
        file_samples = samples_from_frames(frames)
        samples.extend(file_samples)
        targets.extend([label] * len(file_samples))
        print(f"{filename}: {label}개 라벨, {len(file_samples)} 샘플")

    result = train_model(np.asarray(samples), np.asarray(targets), args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
