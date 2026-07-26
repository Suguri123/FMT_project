import warnings

warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype\(\) is deprecated.*",
)

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import json
import os
import re
from collections import Counter
from urllib.parse import quote
from supabase import create_client
from streamlit_webrtc import webrtc_streamer
import serial.tools.list_ports

import sys
import importlib

# Streamlit 모듈 캐시 우회 및 갱신 강제 적용
for module_name in ['motion_capture', 'physical_ai']:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

from motion_capture import MotionCaptureSession
from physical_ai import samples_from_frames, train_model

ARDUINO_DEFAULT_PORT = "COM22"
LEFT_FIST_THRESHOLD = 0.60

# 페이지 설정
st.set_page_config(page_title="FINGER MOTION TRACKER", layout="wide", initial_sidebar_state="collapsed")

# 프리미엄 UI 디자인 CSS 적용
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* 전체 페이지 상단 및 가로 레이아웃 여백 제거 */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1200px !important;
        margin: 0 auto;
    }
    
    .main-header {
        text-align: center;
        font-weight: 800;
        font-size: 2.2rem;
        background: transparent;
        padding: 10px 0 5px;
        margin-top: 0;
        margin-bottom: 5px;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        text-align: center;
        color: #868e96;
        font-size: 1.05rem;
        margin-top: 0px;
        margin-bottom: 25px;
    }
    
    /* 프리미엄 카드 컨테이너 여백/테두리 라운드 스타일 (배경색은 스트림릿 테마가 알아서 제어하도록 상속시킴) */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 12px !important;
        padding: 22px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.02) !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        border-bottom: 2px solid #e03131;
        padding-bottom: 10px;
        margin-top: 0 !important;
        margin-bottom: 20px;
        font-size: 1.45rem;
        font-weight: 700;
    }
    
    h5 {
        font-weight: 700;
        margin-top: 0.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    div[data-testid="stRadio"] {
        margin-bottom: 0 !important;
    }

    /* 입력 필드 및 버튼의 보더 라운드 코너 처리 (글자/배경색 강제 변경 배제) */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="select"] > div,
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stCustomComponentV1"] select,
    div[data-testid="stWebRtcMediaStream"] select {
        border-radius: 6px !important;
        font-size: 1rem !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        min-height: 42px !important;
        font-weight: 700 !important;
    }

    /* 웹캠 컴포넌트 */
    div[data-testid="stCustomComponentV1"] {
        width: 100% !important;
        max-width: 320px !important;
        height: auto !important;
        min-height: 240px !important;
        margin: 0 auto !important;
    }

    div[data-testid="stCustomComponentV1"] iframe {
        width: 100% !important;
        max-width: 320px !important;
        height: 240px !important;
        min-height: 240px !important;
        border-radius: 8px !important;
    }

    /* 카메라 선택 (select) 및 기타 디바이스 컨트롤 요소 스타일 */
    div[data-testid="stCustomComponentV1"] select,
    div[data-testid="stWebRtcMediaStream"] select {
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        padding: 8px 12px !important;
        outline: none !important;
        width: 100% !important;
        cursor: pointer !important;
        margin-top: 8px !important;
    }

    div[data-testid="stButton"] button[kind="tertiary"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        min-height: 0 !important;
        padding: 0.2rem !important;
    }

    .selected-category-badge {
        display: inline-block;
        margin: 0.35rem 0 0.65rem;
        padding: 0.35rem 0.65rem;
        border-radius: 6px;
        background: rgba(224, 49, 49, 0.1) !important;
        color: #e03131 !important;
        border-left: 4px solid #e03131;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .motion-file-link {
        display: block;
        padding: 0.3rem 0.5rem;
        font-size: 0.92rem;
        line-height: 1.2;
        overflow-wrap: anywhere;
        text-decoration: none !important;
        border-radius: 4px;
        transition: all 0.2s ease;
    }

    .motion-file-link:hover {
        background-color: rgba(128, 128, 128, 0.1) !important;
        color: #e03131 !important;
    }

    .motion-file-link.selected {
        color: #e03131 !important;
        background-color: rgba(224, 49, 49, 0.08) !important;
        font-weight: 700;
        border-left: 3px solid #e03131;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 6px !important;
        margin-bottom: 0.5rem !important;
    }

    @media (prefers-color-scheme: dark) {
        .sub-header {
            color: #adb5bd !important;
        }
        .motion-file-link {
            color: #eceef2 !important;
        }
        .motion-file-link:hover {
            color: #ff8787 !important;
        }
        .motion-file-link.selected {
            color: #ff8787 !important;
            background-color: rgba(224, 49, 49, 0.15) !important;
        }
    }

    /* 파일 리스트 선택 버튼 왼쪽 정렬 */
    div[data-testid="stExpander"] div[data-testid="stButton"] button {
        text-align: left !important;
        justify-content: flex-start !important;
        white-space: normal !important;
        word-break: break-all !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>FINGER MOTION TRACKER</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>정밀 뼈대(Skeleton) 모션 추출 및 검증용 MVP</div>", unsafe_allow_html=True)

# Supabase 클라이언트 초기화
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key), None
    except Exception as e:
        return None, str(e)

supabase_client, supabase_init_error = init_supabase()

app_mode_tabs = st.tabs(["🎥 실시간 녹화 & AI 제어", "🔄 모션 재생 & 관절 분석", "☁️ 클라우드 데이터 히스토리"])

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
PHYSICAL_AI_LABELS_FILE = "physical_ai_labels.json"
PHYSICAL_AI_CLASS_LED_COUNTS_FILE = "physical_ai_class_led_counts.json"
PHYSICAL_AI_FILE_CATEGORIES_FILE = "physical_ai_file_categories.json"

_saved_motion_capture = st.session_state.get('motion_capture')
if (
    not isinstance(_saved_motion_capture, MotionCaptureSession)
    or not hasattr(_saved_motion_capture, '_physical_ai_prediction')
    or not hasattr(_saved_motion_capture, 'set_physical_ai_model')
):
    st.session_state['motion_capture'] = MotionCaptureSession(DATA_DIR)

motion_capture = st.session_state['motion_capture']
st.session_state.setdefault('arduino_realtime_enabled', False)

# 전체 모션 데이터 파일 목록 조회 (전역 변수화)
files = sorted(
    [
        os.path.relpath(os.path.join(root, filename), DATA_DIR).replace('\\', '/')
        for root, _, filenames in os.walk(DATA_DIR)
        for filename in filenames
        if filename.endswith('.json')
    ],
    reverse=True,
)

# active files의 실제 존재하는 카테고리만 수집 (전역 변수화)
active_categories = set()
file_categories = {}
if os.path.exists(PHYSICAL_AI_FILE_CATEGORIES_FILE):
    try:
        with open(PHYSICAL_AI_FILE_CATEGORIES_FILE, "r", encoding="utf-8") as category_file:
            file_categories = json.load(category_file)
    except (OSError, json.JSONDecodeError):
        pass

for motion_file in files:
    parts = motion_file.replace('\\', '/').split('/')
    category = parts[0] if len(parts) > 1 else file_categories.get(motion_file)
    if not category and '_' in motion_file:
        name_parts = motion_file.replace('.json', '').split('_')
        if len(name_parts) >= 3:
            if name_parts[0].isdigit():
                category = name_parts[-1]
            else:
                category = name_parts[0]
    if category:
        active_categories.add(category)

saved_categories = ["전체"] + sorted(list(active_categories))

def sanitize_filename(name):
    cleaned = re.sub(r'[\\/:*?"<>|]+', '_', name.strip())
    cleaned = re.sub(r'\s+', '_', cleaned).strip('._')
    return cleaned or f"motion_{int(time.time())}"


def save_physical_ai_label(filename, class_name, led_count=None, category=None):
    labels = {}
    if os.path.exists(PHYSICAL_AI_LABELS_FILE):
        try:
            with open(PHYSICAL_AI_LABELS_FILE, "r", encoding="utf-8") as label_file:
                labels = json.load(label_file)
        except (json.JSONDecodeError, OSError):
            labels = {}
    labels[filename] = str(class_name).strip()
    with open(PHYSICAL_AI_LABELS_FILE, "w", encoding="utf-8") as label_file:
        json.dump(labels, label_file, ensure_ascii=False, indent=2)
    if led_count is not None:
        class_led_counts = {}
        if os.path.exists(PHYSICAL_AI_CLASS_LED_COUNTS_FILE):
            try:
                with open(PHYSICAL_AI_CLASS_LED_COUNTS_FILE, "r", encoding="utf-8") as class_file:
                    class_led_counts = json.load(class_file)
            except (json.JSONDecodeError, OSError):
                class_led_counts = {}
        class_led_counts[str(class_name).strip()] = int(led_count)
        with open(PHYSICAL_AI_CLASS_LED_COUNTS_FILE, "w", encoding="utf-8") as class_file:
            json.dump(class_led_counts, class_file, ensure_ascii=False, indent=2)
    if category:
        file_categories = {}
        if os.path.exists(PHYSICAL_AI_FILE_CATEGORIES_FILE):
            try:
                with open(PHYSICAL_AI_FILE_CATEGORIES_FILE, "r", encoding="utf-8") as category_file:
                    file_categories = json.load(category_file)
            except (OSError, json.JSONDecodeError):
                file_categories = {}
        file_categories[filename] = str(category).strip()
        with open(PHYSICAL_AI_FILE_CATEGORIES_FILE, "w", encoding="utf-8") as category_file:
            json.dump(file_categories, category_file, ensure_ascii=False, indent=2)


def train_saved_physical_ai_data(category=None):
    if not os.path.exists(PHYSICAL_AI_LABELS_FILE):
        raise ValueError("아직 라벨링된 모션 파일이 없습니다.")

    with open(PHYSICAL_AI_LABELS_FILE, "r", encoding="utf-8") as label_file:
        labels = json.load(label_file)

    samples = []
    targets = []
    skipped = 0
    file_categories = {}
    if os.path.exists(PHYSICAL_AI_FILE_CATEGORIES_FILE):
        try:
            with open(PHYSICAL_AI_FILE_CATEGORIES_FILE, "r", encoding="utf-8") as category_file:
                file_categories = json.load(category_file)
        except (OSError, json.JSONDecodeError):
            file_categories = {}
    for filename, class_name in labels.items():
        normalized_filename = filename.replace('\\', '/')
        if '/' in normalized_filename:
            filename_category = normalized_filename.split('/')[0]
        else:
            filename_category = file_categories.get(filename)
            # 파일명이 {timestamp}_{label}_{category}.json 형태인 경우 복구
            if not filename_category and '_' in filename:
                parts = filename.replace('.json', '').split('_')
                if len(parts) >= 3:
                    if parts[0].isdigit():
                        filename_category = parts[-1]
                    else:
                        filename_category = parts[0]
                    
        if category and filename_category != category:
            continue
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            skipped += 1
            continue
        with open(path, "r", encoding="utf-8") as motion_file:
            frames = json.load(motion_file)
        file_samples = samples_from_frames(frames)
        samples.extend(file_samples)
        targets.extend([str(class_name).strip()] * len(file_samples))

    model_output = "physical_ai_model.joblib"
    if category:
        os.makedirs("physical_ai_models", exist_ok=True)
        model_output = os.path.join(
            "physical_ai_models",
            f"{sanitize_filename(category)}.joblib",
        )

    result = train_model(
        np.asarray(samples),
        np.asarray(targets, dtype=str),
        model_output,
    )
    result["category"] = category or "전체"
    result["skipped"] = skipped
    return result


mp_hands = mp.solutions.hands

FINGER_JOINTS = {
    "엄지": [1, 2, 3, 4],
    "검지": [5, 6, 7, 8],
    "중지": [9, 10, 11, 12],
    "약지": [13, 14, 15, 16],
    "소지": [17, 18, 19, 20],
}

JOINT_LABELS = {
    1: "엄지 CMC", 2: "엄지 MCP", 3: "엄지 IP", 4: "엄지 TIP",
    5: "검지 MCP", 6: "검지 PIP", 7: "검지 DIP", 8: "검지 TIP",
    9: "중지 MCP", 10: "중지 PIP", 11: "중지 DIP", 12: "중지 TIP",
    13: "약지 MCP", 14: "약지 PIP", 15: "약지 DIP", 16: "약지 TIP",
    17: "소지 MCP", 18: "소지 PIP", 19: "소지 DIP", 20: "소지 TIP",
}

HAND_KEYS = [("왼손", "left_hand", 0), ("오른손", "right_hand", 1)]

FIST_FINGER_IDS = {
    "검지": {"mcp": 5, "pip": 6, "tip": 8},
    "중지": {"mcp": 9, "pip": 10, "tip": 12},
    "약지": {"mcp": 13, "pip": 14, "tip": 16},
    "소지": {"mcp": 17, "pip": 18, "tip": 20},
}


def get_hand_points(frame, hand_key, fallback_index):
    hand_points = frame.get(hand_key)
    if hand_points:
        return hand_points
    legacy_hands = frame.get('hands', [])
    if fallback_index < len(legacy_hands):
        return legacy_hands[fallback_index]
    return []


def landmark_distance(first, second):
    return (
        (first['x'] - second['x']) ** 2
        + (first['y'] - second['y']) ** 2
        + (first['z'] - second['z']) ** 2
    ) ** 0.5


def describe_hand_shape(hand_points):
    landmarks = {point['id']: point for point in hand_points}
    wrist = landmarks.get(0)
    middle_mcp = landmarks.get(9)
    if not wrist or not middle_mcp:
        return None

    palm_size = landmark_distance(wrist, middle_mcp)
    if palm_size <= 0:
        return None

    folded_fingers = []
    extended_fingers = []
    for finger_name, ids in FIST_FINGER_IDS.items():
        mcp = landmarks.get(ids["mcp"])
        pip = landmarks.get(ids["pip"])
        tip = landmarks.get(ids["tip"])
        if not mcp or not pip or not tip:
            continue

        tip_to_wrist = landmark_distance(tip, wrist)
        pip_to_wrist = landmark_distance(pip, wrist)
        tip_to_mcp = landmark_distance(tip, mcp)
        extended = tip_to_wrist > pip_to_wrist * 1.12 and tip_to_mcp > palm_size * 0.65
        folded = tip_to_wrist <= pip_to_wrist * 1.02 and tip_to_mcp < palm_size * 0.90

        if folded:
            folded_fingers.append(finger_name)
        elif extended:
            extended_fingers.append(finger_name)

    if len(folded_fingers) >= 4:
        shape = "주먹"
    elif len(folded_fingers) >= 2:
        shape = "부분적으로 접힌 손"
    elif len(extended_fingers) >= 3:
        shape = "펴진 손"
    else:
        shape = "판정 어려움"

    return {
        "shape": shape,
        "folded": folded_fingers,
        "extended": extended_fingers,
    }


def calculate_hand_shape_stats(data, hand_key, fallback_index):
    shape_counts = Counter()
    folded_counts = Counter()
    detected_frames = 0

    for frame in data:
        hand_points = get_hand_points(frame, hand_key, fallback_index)
        if not hand_points:
            continue

        shape_info = describe_hand_shape(hand_points)
        if not shape_info:
            continue

        detected_frames += 1
        shape_counts[shape_info["shape"]] += 1
        folded_counts.update(shape_info["folded"])

    return {
        "detected_frames": detected_frames,
        "shape_counts": shape_counts,
        "folded_counts": folded_counts,
    }


def build_motion_shape_summary(data):
    summaries = []
    if not data:
        return summaries

    for hand_label, hand_key, fallback_index in HAND_KEYS:
        stats = calculate_hand_shape_stats(data, hand_key, fallback_index)
        detected_frames = stats["detected_frames"]
        shape_counts = stats["shape_counts"]
        folded_counts = stats["folded_counts"]

        if not detected_frames:
            summaries.append(f"{hand_label}: 손 좌표가 감지되지 않았습니다.")
            continue

        main_shape, main_count = shape_counts.most_common(1)[0]
        fist_ratio = shape_counts["주먹"] / detected_frames
        folded_text = ", ".join(
            finger for finger, _ in folded_counts.most_common()
        ) or "없음"
        summaries.append(
            f"{hand_label}: {main_shape}으로 보입니다. "
            f"주먹 판정 비율 {fist_ratio:.0%}, 분석 프레임 {detected_frames}개, "
            f"자주 접힌 손가락: {folded_text}"
        )

    return summaries


DIRECTION_FINGERTIP_IDS = {4, 8, 12, 16, 20}
RIGHT_DIRECTION_KEYWORDS = ("오른방향", "오른쪽", "오른", "right")
LEFT_DIRECTION_KEYWORDS = ("왼방향", "왼쪽", "왼", "left")


def is_direction_motion(filename, direction):
    normalized = (filename or "").lower()
    keywords = RIGHT_DIRECTION_KEYWORDS if direction == "right" else LEFT_DIRECTION_KEYWORDS
    return any(keyword in normalized for keyword in keywords)


def get_direction_x(frame, direction):
    points = []
    for hand_key, fallback_index in (("left_hand", 0), ("right_hand", 1)):
        points.extend(get_hand_points(frame, hand_key, fallback_index))

    fingertip_x_values = [point["x"] for point in points if point.get("id") in DIRECTION_FINGERTIP_IDS]
    if fingertip_x_values:
        return max(fingertip_x_values) if direction == "right" else min(fingertip_x_values)
    if points:
        return max(point["x"] for point in points) if direction == "right" else min(point["x"] for point in points)
    return None


def build_direction_commands(data, direction):
    if not data:
        return []

    last_index = max(len(data) - 1, 1)
    prefix = "R" if direction == "right" else "L"
    commands = []
    for frame_index, frame in enumerate(data):
        if get_direction_x(frame, direction) is None:
            commands.append("OFF")
            continue
        progress = frame_index / last_index
        led_count = min(max(int(progress * 7) + 1, 1), 8)
        commands.append(f"{prefix}{led_count}")
    return commands


def send_saved_motion_to_arduino(data, port, playback_speed, status_box, progress_box, motion_name=None, playback_viewer=None):
    if not data:
        raise ValueError("전송할 저장 데이터가 없습니다.")

    start_t = data[0].get('time', 0)
    previous_t = start_t
    last_command = None
    direction_commands = build_direction_commands(data, "right") if is_direction_motion(motion_name, "right") else build_direction_commands(data, "left") if is_direction_motion(motion_name, "left") else None
    effective_speed = float(playback_speed) * (4.0 if direction_commands is not None else 1.0)

    try:
        direction_commands = None
        motion_capture._clear_direction_animation()
        for frame_index, frame in enumerate(data):
            current_t = frame.get('time', previous_t)
            if frame_index > 0:
                delay = max(0.0, current_t - previous_t) / max(effective_speed, 0.1)
                time.sleep(delay)

            if playback_viewer is not None:
                frame_image = build_skeleton_image(frame)
                playback_viewer.image(
                    frame_image,
                    width=280,
                    caption=f"{motion_name or '이름 없음'} · 아두이노 전송 동기화 시각화",
                )

            if direction_commands is not None:
                command = direction_commands[frame_index]
                shape = "오른쪽 방향"
            else:
                command = motion_capture._arduino_command_for_frame(frame, time.time())
                shape = getattr(motion_capture, '_physical_ai_prediction', 'AI 예측 대기 중')
            if command != last_command:
                motion_capture.send_arduino_command(port, command)
                last_command = command

            elapsed = current_t - start_t
            progress_box.progress((frame_index + 1) / len(data))
            status_box.info(
                f"파일 [{motion_name or '이름 없음'}] 전송 중 · "
                f"{frame_index + 1}/{len(data)} 프레임 · {elapsed:.1f}s · "
                f"{shape} · LED {command}"
            )
            previous_t = current_t
    finally:
        motion_capture.send_arduino_command(port, "OFF")

    status_box.success(f"파일 [{motion_name or '이름 없음'}] 전송 완료 · LED OFF")


def build_hand_analysis_rows(data):
    rows = []
    if not data:
        return rows

    start_t = data[0].get('time', 0)
    for frame in data:
        elapsed = frame.get('time', start_t) - start_t
        for hand_label, hand_key, fallback_index in HAND_KEYS:
            hand_points = get_hand_points(frame, hand_key, fallback_index)
            if not hand_points:
                continue

            landmarks = {point['id']: point for point in hand_points}
            for finger_name, joint_ids in FINGER_JOINTS.items():
                for joint_id in joint_ids:
                    point = landmarks.get(joint_id)
                    if not point:
                        continue
                    joint_label = JOINT_LABELS.get(joint_id, f"Landmark {joint_id}")
                    rows.extend([
                        {"Time(s)": elapsed, "손": hand_label, "손가락": finger_name, "마디": joint_label, "축": "X", "좌표": point['x']},
                        {"Time(s)": elapsed, "손": hand_label, "손가락": finger_name, "마디": joint_label, "축": "Y", "좌표": point['y']},
                        {"Time(s)": elapsed, "손": hand_label, "손가락": finger_name, "마디": joint_label, "축": "Z", "좌표": point['z']},
                    ])
    return rows


def build_skeleton_image(frame_data, width=480, height=360):
    image = np.full((height, width, 3), (17, 24, 39), dtype=np.uint8)
    hands_to_draw = frame_data.get('hands', [])
    if not hands_to_draw:
        hands_to_draw = []
        if frame_data.get('left_hand'):
            hands_to_draw.append(frame_data['left_hand'])
        if frame_data.get('right_hand'):
            hands_to_draw.append(frame_data['right_hand'])
    if not hands_to_draw and 'landmarks' in frame_data:
        hands_to_draw = [frame_data.get('landmarks')]

    hand_colors = [(79, 77, 255), (255, 169, 64)]
    visible_hands = 0
    for hand_index, hand_points in enumerate(hands_to_draw):
        if not hand_points:
            continue
        visible_hands += 1

        points = []
        for lm in hand_points:
            x_pos = int(min(max(lm['x'], 0.0), 1.0) * (width - 1))
            y_pos = int(min(max(lm['y'], 0.0), 1.0) * (height - 1))
            points.append((x_pos, y_pos))

        color = hand_colors[hand_index % len(hand_colors)]

        for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx], (245, 245, 245), 4, cv2.LINE_AA)

        for point_index, point in enumerate(points):
            cv2.circle(image, point, 7, color, -1, cv2.LINE_AA)
            cv2.circle(image, point, 8, (17, 24, 39), 1, cv2.LINE_AA)
            cv2.putText(
                image,
                str(point_index),
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    if not visible_hands:
        cv2.putText(
            image,
            'No hand landmarks',
            (128, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (236, 240, 241),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ----------------- UI 탭별 구성 -----------------
tab1, tab2, tab3 = app_mode_tabs
webcam_on = False

# ----------------- TAB 1: 🎥 실시간 녹화 & AI 제어 -----------------
with tab1:
    col_left, col_right = st.columns([1.5, 1], gap="medium")
    
    with col_left:
        st.markdown("### 🎥 실시간 웹캠 및 AI 피드백")
        
        webrtc_ctx = webrtc_streamer(
            key="motion-capture-webrtc",
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
            },
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 320},
                    "height": {"ideal": 240},
                    "frameRate": {"ideal": 15, "max": 18},
                    "facingMode": "user",
                },
                "audio": False,
            },
            video_frame_callback=motion_capture.process_video_frame,
            on_video_ended=motion_capture.stop_stream,
            async_processing=True,
            sendback_audio=False,
            media_toggle_controls=False,
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "muted": True,
                "playsInline": True,
                "style": {
                    "width": "320px",
                    "height": "240px",
                    "objectFit": "cover",
                    "display": "block",
                    "margin": "0 auto",
                },
            },
            translations={
                "start": "웹캠 켜기",
                "stop": "웹캠 끄기",
                "select_device": "카메라 선택",
                "device_ask_permission": "카메라 사용 권한을 허용해 주세요.",
                "device_not_available": "사용 가능한 카메라를 찾을 수 없습니다.",
                "device_access_denied": "브라우저의 카메라 권한이 차단되었습니다.",
            },
        )
        webcam_on = webrtc_ctx.state.playing if webrtc_ctx.state else False
        st.caption("웹캠을 켜면 실시간 손가락 관절 감지 및 AI 학습/아두이노 전송 기능이 활성화됩니다.")
        
        recording_status = st.empty()
        arduino_realtime_status = st.empty()
        physical_ai_realtime_status = st.empty()
        
    with col_right:
        st.markdown("### ⚙️ 제어 패널")
        
        # 💾 저장 및 녹화 설정 카드
        with st.container():
            st.markdown("##### 💾 저장 및 녹화 설정")
            project_name = st.text_input(
                "프로젝트 명",
                value="기본 프로젝트",
                key="project_name_input",
            )
            save_name = sanitize_filename(project_name)
            
            record_duration = st.slider("녹화 시간 (초)", 1, 15, 5, key="record_duration_slider")
            
            rec_clicked = st.button(
                "🔴 REC (녹화 시작)",
                width="stretch",
                type="primary",
                disabled=not webcam_on,
                key="rec_button_main"
            )
            if rec_clicked:
                if motion_capture.start_recording(record_duration, save_name):
                    recording_status.info("3초 후 녹화를 시작합니다.")
                else:
                    recording_status.warning("이미 카운트다운 또는 녹화가 진행 중입니다.")

        # 🧠 피지컬 AI 학습 및 라벨링 카드
        with st.container():
            st.markdown("##### 🧠 피지컬 AI & 라벨링 설정")
            with st.expander("📝 학습 데이터 라벨링 (신규 모션 등록)", expanded=False):
                physical_ai_category = st.text_input(
                    "카테고리명",
                    value="",
                    placeholder="예: 손가락숫자세기",
                    key="physical_ai_category",
                )
                category_name = sanitize_filename(physical_ai_category)
                physical_ai_label = st.text_input(
                    "클래스 네임",
                    key="physical_ai_label_count",
                )
                physical_ai_timestamp = int(time.time())
                physical_ai_name = sanitize_filename(
                    f"{category_name}_{physical_ai_label}_{physical_ai_timestamp}"
                )
                physical_ai_relative_name = physical_ai_name
                st.text_input(
                    "학습 데이터 파일명",
                    value=f"{physical_ai_relative_name}.json",
                    disabled=True,
                    key="physical_ai_filename_preview",
                )
                physical_ai_led_count = st.number_input(
                    "이 클래스가 켤 LED 개수",
                    min_value=0,
                    max_value=8,
                    value=0,
                    step=1,
                    key="physical_ai_led_count",
                )
                label_rec_clicked = st.button(
                    "🧠 라벨링 녹화 시작",
                    width="stretch",
                    type="secondary",
                    disabled=not webcam_on,
                    key="physical_ai_record_button",
                )
                if label_rec_clicked:
                    if motion_capture.start_recording(record_duration, physical_ai_relative_name):
                        st.session_state['pending_physical_ai_label'] = str(physical_ai_label).strip()
                        st.session_state['pending_physical_ai_led_count'] = int(physical_ai_led_count)
                        st.session_state['pending_physical_ai_category'] = category_name
                        recording_status.info(
                            f"{physical_ai_label}개 라벨로 3초 후 녹화를 시작합니다."
                        )
                    else:
                        recording_status.warning("이미 카운트다운 또는 녹화가 진행 중입니다.")

            # AI 학습 개시
            # 실제 디스크에 존재하는 모션 파일들의 카테고리만 동적으로 추출
            active_categories = set()
            file_categories = {}
            if os.path.exists(PHYSICAL_AI_FILE_CATEGORIES_FILE):
                try:
                    with open(PHYSICAL_AI_FILE_CATEGORIES_FILE, "r", encoding="utf-8") as category_file:
                        file_categories = json.load(category_file)
                except (OSError, json.JSONDecodeError):
                    pass
            for motion_file in files:
                parts = motion_file.replace('\\', '/').split('/')
                category = parts[0] if len(parts) > 1 else file_categories.get(motion_file)
                if not category and '_' in motion_file:
                    name_parts = motion_file.replace('.json', '').split('_')
                    if len(name_parts) >= 3:
                        if name_parts[0].isdigit():
                            category = name_parts[-1]
                        else:
                            category = name_parts[0]
                if category:
                    active_categories.add(category)
            
            saved_categories = ["전체"] + sorted(list(active_categories))

            selected_training_category = st.selectbox(
                "학습할 카테고리 선택",
                options=saved_categories,
                key="selected_training_category",
            )
            train_ai_clicked = st.button(
                "🧠 저장된 모션으로 AI 학습 시작",
                width="stretch",
                key="train_physical_ai_button",
            )
            if train_ai_clicked:
                try:
                    training_category = None if selected_training_category == "전체" else selected_training_category
                    train_result = train_saved_physical_ai_data(training_category)
                    if training_category is None:
                        motion_capture._physical_ai.load()
                    st.success(
                        f"{train_result['category']} 카테고리 학습 완료 · {train_result['samples']}개 샘플 · "
                        f"클래스 {', '.join(map(str, train_result['classes']))}"
                    )
                    if train_result.get("skipped"):
                        st.caption(f"파일이 없어 건너뛴 라벨 파일: {train_result['skipped']}개")
                except Exception as exc:
                    st.error(f"AI 학습 실패: {exc}")

        # 🔌 아두이노 실시간 통신 카드
        with st.container():
            # 사용 가능한 COM 포트 및 아두이노 오토 스캔
            try:
                available_ports = list(serial.tools.list_ports.comports())
                port_options = []
                detected_arduino = None
                
                for p in available_ports:
                    desc = p.description or ""
                    hwid = p.hwid or ""
                    label = f"{p.device} ({desc})"
                    port_options.append(label)
                    
                    if not detected_arduino and any(x in desc.lower() or x in hwid.lower() for x in ["arduino", "ch340", "cp210", "ftdi", "usb serial", "usb-to-serial"]):
                        detected_arduino = label
                
                if port_options:
                    if detected_arduino:
                        port_options.remove(detected_arduino)
                        port_options.insert(0, f"⭐ {detected_arduino} [자동 감지]")
                    port_options.append("직접 입력 (텍스트)")
                else:
                    port_options = ["검색된 포트 없음", "직접 입력 (텍스트)"]
            except Exception:
                port_options = ["직접 입력 (텍스트)"]
                
            selected_port_label = st.selectbox(
                "연결할 Arduino COM 포트 선택",
                options=port_options,
                key="arduino_port_selectbox"
            )
            
            if "직접 입력" in selected_port_label or "검색된 포트 없음" in selected_port_label:
                realtime_arduino_port = st.text_input(
                    "직접 입력할 COM 포트 번호 (예: COM22)",
                    value="COM22",
                    key="realtime_arduino_port_manual"
                )
            else:
                # 라벨에서 실제 디바이스 이름만 파싱 (예: "⭐ COM22 (Arduino Uno) [자동 감지]" -> "COM22")
                realtime_arduino_port = selected_port_label.replace("⭐ ", "").split(" ")[0]
            active_ai_category = st.selectbox(
                "실시간 웹캠 AI 적용 카테고리",
                options=saved_categories,
                key="active_ai_category",
            )
            st.markdown(
                f"<div class='selected-category-badge'>현재 선택 카테고리: {active_ai_category}</div>",
                unsafe_allow_html=True,
            )
            if active_ai_category == "전체":
                active_model_path = "physical_ai_model.joblib"
            else:
                active_model_path = os.path.join(
                    "physical_ai_models",
                    f"{sanitize_filename(active_ai_category)}.joblib",
                )
            if not motion_capture.set_physical_ai_model(active_model_path, active_ai_category):
                st.warning(
                    f"'{active_ai_category}' 카테고리 모델이 없습니다. 먼저 해당 카테고리를 학습하세요."
                )
            realtime_label = (
                "실시간 웹캠 화면 전송 중지"
                if st.session_state['arduino_realtime_enabled']
                else "실시간 웹캠 화면 전송 시작"
            )
            realtime_clicked = st.button(
                realtime_label,
                width="stretch",
                disabled=not webcam_on,
                icon=":material/sensors:",
                key="toggle_realtime_arduino",
            )
            if realtime_clicked:
                st.session_state['arduino_realtime_enabled'] = not st.session_state['arduino_realtime_enabled']
                st.rerun()

            if not webcam_on:
                st.session_state['arduino_realtime_enabled'] = False

            motion_capture.configure_arduino_realtime(
                st.session_state['arduino_realtime_enabled'],
                realtime_arduino_port,
            )

# ----------------- TAB 2: 🔄 모션 재생 & 관절 분석 -----------------
with tab2:
    col_pb_left, col_pb_right = st.columns([1.1, 1.4], gap="medium")
    
    files = sorted(
        [
            os.path.relpath(os.path.join(root, filename), DATA_DIR).replace('\\', '/')
            for root, _, filenames in os.walk(DATA_DIR)
            for filename in filenames
            if filename.endswith('.json')
        ],
        reverse=True,
    )
    if 'selected_file' not in st.session_state:
        st.session_state['selected_file'] = files[0] if files else None
    query_selected_file = st.query_params.get("selected_file")
    if query_selected_file in files:
        st.session_state['selected_file'] = query_selected_file
    if st.session_state['selected_file'] not in files:
        st.session_state['selected_file'] = files[0] if files else None
    
    selected_file = st.session_state['selected_file']
    
    with col_pb_left:
        st.markdown("### 📂 모션 리플레이 & 탐색")
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # 2D 뼈대 시각화 영역
        playback_viewer = st.empty()
        
        # 재생 속도 및 버튼
        col_speed, col_play = st.columns([1, 2])
        with col_speed:
            speed = st.selectbox("배속", [0.5, 1.0, 2.0], index=1, key="playback_speed_selector")
        with col_play:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            play_clicked = st.button(
                "▶ PLAYBACK",
                width="stretch",
                type="primary",
                disabled=webcam_on,
                help="웹캠이 켜져 있으면 먼저 웹캠을 꺼주세요.",
                key="playback_start_btn"
            )
            if webcam_on:
                st.caption("재생하려면 웹캠을 꺼주세요.")

        # 저장된 모션 목록 카드화
        with st.container():
            st.markdown("##### 📁 저장된 모션 파일 리스트")
            file_categories = {}
            if os.path.exists(PHYSICAL_AI_FILE_CATEGORIES_FILE):
                try:
                    with open(PHYSICAL_AI_FILE_CATEGORIES_FILE, "r", encoding="utf-8") as category_file:
                        file_categories = json.load(category_file)
                except (OSError, json.JSONDecodeError):
                    file_categories = {}
            category_files = {}
            for motion_file in files:
                parts = motion_file.replace('\\', '/').split('/')
                category = parts[0] if len(parts) > 1 else file_categories.get(motion_file, '기타')
                category_files.setdefault(category, []).append(motion_file)

            if category_files:
                for category, category_motion_files in sorted(category_files.items()):
                    category_col, category_delete_col = st.columns([6, 1], gap="small")
                    with category_col:
                        with st.expander(
                            f"📁 {category} ({len(category_motion_files)}개)",
                            expanded=False,
                        ):
                            for motion_file in category_motion_files:
                                file_col, delete_col = st.columns([6, 1], gap="small")
                                is_selected = motion_file == selected_file
                                with file_col:
                                    if st.button(
                                        motion_file,
                                        key=f"select_file_{motion_file}",
                                        use_container_width=True,
                                        type="primary" if is_selected else "secondary",
                                    ):
                                        st.session_state['selected_file'] = motion_file
                                        st.query_params["selected_file"] = motion_file
                                        st.rerun()
                                with delete_col:
                                    if st.button(
                                        "🗑️",
                                        key=f"delete_motion_{motion_file}",
                                        help=f"{motion_file} 삭제",
                                        type="tertiary",
                                    ):
                                        file_to_delete = os.path.abspath(os.path.join(DATA_DIR, motion_file))
                                        data_dir_abs = os.path.abspath(DATA_DIR)
                                        if (
                                            os.path.commonpath([data_dir_abs, file_to_delete]) == data_dir_abs
                                            and os.path.isfile(file_to_delete)
                                        ):
                                            os.remove(file_to_delete)
                                        remaining_files = [f for f in files if f != motion_file]
                                        st.session_state['selected_file'] = remaining_files[0] if remaining_files else None
                                        st.rerun()
                    with category_delete_col:
                        if st.button(
                            "🗑️",
                            key=f"delete_category_{category}",
                            help=f"{category} 카테고리 전체 삭제",
                            type="tertiary",
                        ):
                            data_dir_abs = os.path.abspath(DATA_DIR)
                            deleted_files = []
                            for motion_file in category_motion_files:
                                file_to_delete = os.path.abspath(os.path.join(DATA_DIR, motion_file))
                                if (
                                    os.path.commonpath([data_dir_abs, file_to_delete]) == data_dir_abs
                                    and os.path.isfile(file_to_delete)
                                ):
                                    os.remove(file_to_delete)
                                    deleted_files.append(motion_file)

                            if os.path.exists(PHYSICAL_AI_LABELS_FILE):
                                try:
                                    with open(PHYSICAL_AI_LABELS_FILE, "r", encoding="utf-8") as label_file:
                                        labels = json.load(label_file)
                                    for motion_file in deleted_files:
                                        labels.pop(motion_file, None)
                                    with open(PHYSICAL_AI_LABELS_FILE, "w", encoding="utf-8") as label_file:
                                        json.dump(labels, label_file, ensure_ascii=False, indent=2)
                                except (OSError, json.JSONDecodeError):
                                    pass

                            remaining_files = [f for f in files if f not in deleted_files]
                            st.session_state['selected_file'] = remaining_files[0] if remaining_files else None
                            st.rerun()
            else:
                st.caption("저장된 모션이 없습니다.")

            # 로컬 다운로드 및 DB 업로드 버튼
            if selected_file:
                file_path = os.path.join(DATA_DIR, selected_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        json_data = f.read()
                    
                    dl_col, cloud_col = st.columns(2)
                    with dl_col:
                        st.download_button(
                            label="📥 파일 다운로드",
                            data=json_data,
                            file_name=selected_file,
                            mime="application/json",
                            width="stretch",
                            key="btn_download_motion"
                        )
                    with cloud_col:
                        if st.button("☁️ DB 업로드", width="stretch", help="Supabase 클라우드 DB에 데이터를 저장합니다.", key="btn_upload_cloud"):
                            if supabase_client is None:
                                st.error(
                                    "Supabase 연결 설정이 없습니다."
                                )
                            else:
                                try:
                                    data_obj = json.loads(json_data)
                                    response = supabase_client.table("motions").insert({"filename": selected_file, "data": data_obj, "project_name": project_name}).execute()
                                    st.success("클라우드 DB 업로드 완료! ☁️", icon="✅")
                                    st.balloons()
                                except Exception as e:
                                    st.error(f"업로드 실패: {e}")
                except:
                    pass

            # ☁️ 카테고리 단위 일괄 DB 업로드 카드
            if category_files:
                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                with st.container():
                    st.markdown("##### ☁️ 카테고리 일괄 클라우드 업로드")
                    upload_category_list = sorted(list(category_files.keys()))
                    selected_upload_category = st.selectbox(
                        "일괄 업로드할 카테고리 선택",
                        options=upload_category_list,
                        key="selected_bulk_upload_category"
                    )
                    
                    if st.button("☁️ 선택한 카테고리의 모든 파일 업로드", key="btn_bulk_upload_category", use_container_width=True):
                        if supabase_client is None:
                            st.error("Supabase 연결 설정이 없습니다. Secrets를 확인하세요.")
                        else:
                            files_to_upload = category_files.get(selected_upload_category, [])
                            if not files_to_upload:
                                st.warning("업로드할 파일이 없습니다.")
                            else:
                                success_count = 0
                                error_count = 0
                                progress_bar = st.progress(0.0)
                                status_text = st.empty()
                                
                                for idx, f_name in enumerate(files_to_upload):
                                    status_text.info(f"업로드 진행 중 ({idx+1}/{len(files_to_upload)}): {f_name}")
                                    file_path = os.path.join(DATA_DIR, f_name)
                                    try:
                                        with open(file_path, "r", encoding="utf-8") as f_in:
                                            data_obj = json.load(f_in)
                                        # 프로젝트 명은 카테고리 명칭으로 지정
                                        proj_name = selected_upload_category
                                        
                                        supabase_client.table("motions").insert({
                                            "filename": f_name,
                                            "data": data_obj,
                                            "project_name": proj_name
                                        }).execute()
                                        success_count += 1
                                    except Exception as upload_err:
                                        error_count += 1
                                    progress_bar.progress(float(idx + 1) / len(files_to_upload))
                                
                                status_text.empty()
                                progress_bar.empty()
                                
                                if success_count > 0:
                                    st.success(f"'{selected_upload_category}' 카테고리의 모션 파일 {success_count}개 일괄 업로드 성공! ☁️")
                                    st.balloons()
                                if error_count > 0:
                                    st.error(f"{error_count}개 파일 업로드 실패")

    with col_pb_right:
        st.markdown("### 📊 관절 분석 데이터 (Data View)")
        arduino_led_panel = st.container()
        analysis_viewer = st.empty()
        
        if selected_file:
            try:
                file_path = os.path.join(DATA_DIR, selected_file)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                analysis_rows = build_hand_analysis_rows(data)
                if analysis_rows:
                    analysis_df = pd.DataFrame(analysis_rows)
                    with analysis_viewer.container():
                        shape_summaries = build_motion_shape_summary(data)
                        if shape_summaries:
                            st.markdown("##### 🔍 저장된 모션 형상 분석")
                            for summary in shape_summaries:
                                st.info(summary)

                        # 아두이노 전송용 카드
                        with arduino_led_panel:
                            with st.container():
                                st.markdown("##### 🔌 저장 모션의 Arduino LED 제어 시퀀스 전송")
                                st.write("선택한 모션 데이터의 시간 흐름대로 LED 신호를 포트로 전송합니다.")
                                # 사용 가능한 COM 포트 및 아두이노 오토 스캔 (Tab 2)
                                try:
                                    available_ports = list(serial.tools.list_ports.comports())
                                    port_options = []
                                    detected_arduino = None
                                    
                                    for p in available_ports:
                                        desc = p.description or ""
                                        hwid = p.hwid or ""
                                        label = f"{p.device} ({desc})"
                                        port_options.append(label)
                                        
                                        if not detected_arduino and any(x in desc.lower() or x in hwid.lower() for x in ["arduino", "ch340", "cp210", "ftdi", "usb serial", "usb-to-serial"]):
                                            detected_arduino = label
                                    
                                    if port_options:
                                        if detected_arduino:
                                            port_options.remove(detected_arduino)
                                            port_options.insert(0, f"⭐ {detected_arduino} [자동 감지]")
                                        port_options.append("직접 입력 (텍스트)")
                                    else:
                                        port_options = ["검색된 포트 없음", "직접 입력 (텍스트)"]
                                except Exception:
                                    port_options = ["직접 입력 (텍스트)"]
                                    
                                selected_port_label = st.selectbox(
                                    "연결할 Arduino COM 포트 선택",
                                    options=port_options,
                                    key=f"arduino_port_selectbox_{selected_file}"
                                )
                                
                                if "직접 입력" in selected_port_label or "검색된 포트 없음" in selected_port_label:
                                    arduino_port = st.text_input(
                                        "직접 입력할 COM 포트 번호 (예: COM22)",
                                        value="COM22",
                                        key=f"arduino_port_manual_{selected_file}"
                                    )
                                else:
                                    arduino_port = selected_port_label.replace("⭐ ", "").split(" ")[0]

                                seq_ai_category = st.selectbox(
                                    "시퀀스 분석에 적용할 AI 모델 카테고리",
                                    options=saved_categories,
                                    key=f"seq_ai_category_{selected_file}"
                                )

                                if st.button(
                                    "저장 데이터 시간 순서대로 Arduino 전송",
                                    key=f"send_arduino_sequence_{selected_file}",
                                    icon=":material/timeline:",
                                    width="stretch",
                                ):
                                    st.session_state['arduino_realtime_enabled'] = False
                                    
                                    # 시퀀스 분석을 위한 AI 모델 로드
                                    if seq_ai_category == "전체":
                                        seq_model_path = "physical_ai_model.joblib"
                                    else:
                                        seq_model_path = os.path.join(
                                            "physical_ai_models",
                                            f"{sanitize_filename(seq_ai_category)}.joblib",
                                        )
                                    motion_capture.set_physical_ai_model(seq_model_path, seq_ai_category)
                                    
                                    motion_capture.configure_arduino_realtime(False, arduino_port)
                                    # OS가 시리얼 포트 핸들을 완전히 해제할 시간 대기
                                    time.sleep(0.5)
                                    sequence_status = st.empty()
                                    sequence_progress = st.empty()
                                    try:
                                        send_saved_motion_to_arduino(
                                            data,
                                            arduino_port,
                                            speed,
                                            sequence_status,
                                            sequence_progress,
                                            selected_file,
                                            playback_viewer,
                                        )
                                    except Exception as e:
                                        sequence_status.error(f"전송 실패: {e}")

                        hand_tabs = st.tabs(["왼손 관절 시각화", "오른손 관절 시각화"])

                        for hand_tab, hand_label in zip(hand_tabs, ["왼손", "오른손"]):
                            with hand_tab:
                                hand_df = analysis_df[analysis_df["손"] == hand_label]
                                if hand_df.empty:
                                    st.info(f"{hand_label} 감지 데이터가 이 파일에 존재하지 않습니다.")
                                    continue

                                for finger_name in FINGER_JOINTS.keys():
                                    finger_df = hand_df[hand_df["손가락"] == finger_name]
                                    if finger_df.empty:
                                        continue

                                    with st.expander(f"👋 {finger_name} 마디별 정규화 위치 변화", expanded=(finger_name == "검지")):
                                        fig = go.Figure()
                                        for joint_label in finger_df["마디"].unique():
                                            joint_df = finger_df[finger_df["마디"] == joint_label]
                                            for axis in ["X", "Y", "Z"]:
                                                axis_df = joint_df[joint_df["축"] == axis]
                                                fig.add_trace(go.Scatter(
                                                    x=axis_df["Time(s)"],
                                                    y=axis_df["좌표"],
                                                    mode="lines",
                                                    name=f"{joint_label} {axis}"
                                                ))

                                        fig.update_layout(
                                            title=f"{hand_label} {finger_name} 전체 마디 위치 변화",
                                            xaxis_title="시간 (초)",
                                            yaxis_title="정규화 좌표 값",
                                            height=340,
                                            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="left", x=0),
                                            margin=dict(l=10, r=10, t=48, b=95),
                                        )
                                        st.plotly_chart(fig, width="stretch", use_container_width=True)

                        with st.expander("📋 전체 정규화 좌표 원본 데이터 프레임", expanded=False):
                            st.dataframe(analysis_df, width="stretch")
                else:
                    analysis_viewer.info("이 모션 파일에 분석할 손가락 좌표가 기록되어 있지 않습니다.")
            except Exception as e:
                analysis_viewer.error(f"데이터 분석 중 오류 발생: {e}")
        else:
            st.info("재생하거나 분석할 파일이 없습니다. 탭 1에서 먼저 녹화를 수행하세요.")

# ----------------- TAB 3: ☁️ 클라우드 데이터 히스토리 -----------------
with tab3:
    st.markdown("### ☁️ 클라우드 데이터 히스토리 (Supabase 조회 및 다운로드)")
    if supabase_client is None:
        st.error(
            "Supabase 연결 설정이 없습니다. Streamlit Cloud의 App settings > Secrets에 "
            "SUPABASE_URL, SUPABASE_KEY를 추가해야 합니다."
        )
        if supabase_init_error:
            st.caption(f"초기화 오류: {supabase_init_error}")
    else:
        try:
            response = supabase_client.table("motions").select("id, filename, project_name, created_at").order("created_at", desc=True).execute()
            data_list = response.data
            
            if data_list:
                df = pd.DataFrame(data_list)
                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                col_filt1, col_filt2 = st.columns(2)
                with col_filt1:
                    proj_list = ["전체"] + list(df['project_name'].dropna().unique()) if 'project_name' in df.columns else ["전체"]
                    selected_proj = st.selectbox("프로젝트 필터링", proj_list, key="history_proj_filter")
                with col_filt2:
                    selected_date = st.date_input("날짜별 필터링 (선택 해제 시 전체)", value=None, key="history_date_filter")
                
                filtered_df = df.copy()
                if selected_proj != "전체":
                    filtered_df = filtered_df[filtered_df['project_name'] == selected_proj]
                if selected_date is not None:
                    filtered_df = filtered_df[pd.to_datetime(filtered_df['created_at']).dt.date == selected_date]
                
                st.dataframe(filtered_df, width="stretch", use_container_width=True)
                
                st.markdown("#### 데이터 상세 수신 및 관리")
                selected_id = st.selectbox("조회/관리할 레코드 ID 선택", filtered_df['id'].tolist() if not filtered_df.empty else [], key="history_id_select")
                
                if selected_id:
                    col_action1, col_action2 = st.columns(2)
                    with col_action1:
                        if st.button("해당 JSON 데이터 동적 로드", key="btn_load_history_detail"):
                            detail_res = supabase_client.table("motions").select("filename, data").eq("id", selected_id).execute()
                            if detail_res.data:
                                dl_data = detail_res.data[0]
                                st.download_button("📥 JSON 파일 다운로드", data=json.dumps(dl_data['data']), file_name=dl_data['filename'], mime="application/json", key="btn_download_history_detail")
                    with col_action2:
                        delete_clicked = st.button("🗑️ 클라우드 데이터 삭제", type="secondary", key="btn_delete_history", use_container_width=True)
                        if delete_clicked:
                            try:
                                supabase_client.table("motions").delete().eq("id", selected_id).execute()
                                st.success("클라우드 데이터가 성공적으로 삭제되었습니다!")
                                time.sleep(1.0)
                                st.rerun()
                            except Exception as del_err:
                                st.error(f"삭제 오류: {del_err}")
            else:
                st.info("클라우드 Supabase DB에 업로드된 데이터가 없습니다.")
        except Exception as e:
            st.error(f"데이터 쿼리 중 오류 발생: {e}")

# ----------------- 동작 제어 루프 (전역 동작) -----------------
if play_clicked and selected_file and not webcam_on:
    file_path = os.path.join(DATA_DIR, selected_file)
    with open(file_path, 'r') as f:
        data = json.load(f)

    if data:
        previous_t = data[0].get('time', 0)

        for frame_index, frame_data in enumerate(data):
            current_t = frame_data.get('time', previous_t)
            if frame_index > 0:
                delay = max(0.03, current_t - previous_t) / max(float(speed), 0.1)
                time.sleep(min(delay, 0.2))

            frame_image = build_skeleton_image(frame_data)
            playback_viewer.image(
                frame_image,
                width=280,
                caption=f"{selected_file} · frame {frame_index + 1}/{len(data)}",
            )
            previous_t = current_t
elif selected_file and not webcam_on:
    file_path = os.path.join(DATA_DIR, selected_file)
    with open(file_path, 'r') as f:
        data = json.load(f)

    if data:
        preview_image = build_skeleton_image(data[0])
        playback_viewer.image(preview_image, width=280, caption=f"{selected_file} · preview")

if webcam_on:
    new_motion_saved = False
    while webrtc_ctx.state.playing:
        completed_filename = motion_capture.take_completed_filename()
        if completed_filename:
            pending_label = st.session_state.get('pending_physical_ai_label')
            if pending_label is not None:
                save_physical_ai_label(
                    completed_filename,
                    pending_label,
                    st.session_state.get('pending_physical_ai_led_count', 0),
                    st.session_state.get('pending_physical_ai_category'),
                )
                st.session_state['pending_physical_ai_label'] = None
                st.session_state['pending_physical_ai_led_count'] = None
                st.session_state['pending_physical_ai_category'] = None
            st.session_state['selected_file'] = completed_filename
            new_motion_saved = True
            break
            recording_status.success(f"{completed_filename} 저장을 완료했습니다.")
            st.rerun()

        capture_status = motion_capture.status()
        if capture_status.get('physical_ai_available', False):
            physical_ai_realtime_status.info(
                f"🧠 AI 예측 결과: {capture_status['physical_ai_prediction']}"
            )
        elif capture_status.get('physical_ai_available', False):
            physical_ai_realtime_status.caption(
                "🧠 AI 모델 준비 완료 · Arduino 실시간 전송을 켜면 판단 결과가 표시됩니다."
            )
        else:
            physical_ai_realtime_status.warning("🧠 학습된 AI 모델이 없습니다.")
        if capture_status['arduino_enabled']:
            if capture_status['arduino_error']:
                arduino_realtime_status.error(
                    f"Arduino 실시간 전송 오류: {capture_status['arduino_error']}"
                )
            else:
                arduino_realtime_status.success(capture_status['arduino_status'])
        else:
            arduino_realtime_status.info("Arduino 실시간 전송 꺼짐")

        if capture_status['last_error']:
            recording_status.error(
                f"영상 처리 중 오류가 발생했습니다: {capture_status['last_error']}"
            )
        elif capture_status['countdown_remaining'] is not None:
            seconds = max(1, int(capture_status['countdown_remaining']) + 1)
            recording_status.info(f"{seconds}초 후 녹화를 시작합니다.")
        elif capture_status['recording_remaining'] is not None:
            recording_status.warning(
                f"녹화 중 · {capture_status['recording_remaining']:.1f}초 남음"
            )
        else:
            recording_status.success("브라우저 카메라가 연결되었습니다.")

        time.sleep(0.1)
    if new_motion_saved:
        st.rerun()
