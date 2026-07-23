import streamlit as st
import mediapipe as mp
import pandas as pd
import plotly.graph_objects as go
import time
import json
import os
import re
from collections import Counter
from supabase import create_client
from streamlit_webrtc import webrtc_streamer

from motion_capture import MotionCaptureSession

# 페이지 설정
st.set_page_config(page_title="FINGER MOTION TRACKER", layout="wide", initial_sidebar_state="collapsed")

# 프리미엄 UI 디자인 CSS 적용
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* 스트림릿 기본 상단 메뉴바(Deploy/Stop 등)의 공간 차지 영역 투명화 */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* 전체 페이지 상단 여백(빈 공간) 대폭 제거하여 화면을 위로 끌어올림 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }
    
    .main-header {
        text-align: center;
        font-weight: 800;
        font-size: 2.2rem;
        color: #1e272e;
        margin-top: -16px;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }
    
    .sub-header {
        text-align: center;
        color: #7f8fa6;
        font-size: 1.1rem;
        margin-top: -8px;
        margin-bottom: 12px;
    }
    
    /* 패널 스타일링을 위한 CSS 힌트 적용 */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    
    h3 {
        color: #2f3640;
        border-bottom: 2px solid #f1f2f6;
        padding-bottom: 10px;
        margin-bottom: 8px;
    }

    div[data-testid="stRadio"] {
        margin-bottom: 0 !important;
    }

    div[data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        border: 1.5px solid #9aa4b2 !important;
        border-radius: 6px !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.06) !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18) !important;
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
        return create_client(url, key)
    except Exception as e:
        return None

supabase_client = init_supabase()

app_mode = st.radio("화면 선택", ["🎥 로컬 녹화 및 재생", "☁️ 클라우드 데이터 히스토리"], horizontal=True, label_visibility="collapsed")

if app_mode == "☁️ 클라우드 데이터 히스토리":
    st.markdown("### ☁️ 클라우드 데이터 히스토리 (프로젝트 및 날짜별 조회)")
    if supabase_client is None:
        st.error("Supabase API Key가 설정되지 않았습니다.")
    else:
        try:
            # 전체 데이터 가져오기 (시간 순 정렬)
            response = supabase_client.table("motions").select("id, filename, project_name, created_at").order("created_at", desc=True).execute()
            data_list = response.data
            
            if data_list:
                df = pd.DataFrame(data_list)
                # 날짜 변환
                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # 필터링 UI
                col1, col2 = st.columns(2)
                with col1:
                    proj_list = ["전체"] + list(df['project_name'].dropna().unique()) if 'project_name' in df.columns else ["전체"]
                    selected_proj = st.selectbox("프로젝트 필터", proj_list)
                with col2:
                    selected_date = st.date_input("날짜 필터 (선택 안할 시 전체)", value=None)
                
                # 데이터 필터 적용
                filtered_df = df.copy()
                if selected_proj != "전체":
                    filtered_df = filtered_df[filtered_df['project_name'] == selected_proj]
                if selected_date is not None:
                    filtered_df = filtered_df[pd.to_datetime(filtered_df['created_at']).dt.date == selected_date]
                
                st.dataframe(filtered_df, width="stretch")
                
                # 상세 보기
                st.markdown("#### 선택 데이터 다운로드")
                selected_id = st.selectbox("다운로드할 데이터 ID 선택", filtered_df['id'].tolist() if not filtered_df.empty else [])
                
                if selected_id:
                    if st.button("선택한 데이터 불러오기"):
                        detail_res = supabase_client.table("motions").select("filename, data").eq("id", selected_id).execute()
                        if detail_res.data:
                            dl_data = detail_res.data[0]
                            st.download_button("📥 JSON 다운로드", data=json.dumps(dl_data['data']), file_name=dl_data['filename'], mime="application/json")
            else:
                st.info("클라우드에 저장된 데이터가 없습니다.")
        except Exception as e:
            st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e} (Supabase 'motions' 테이블에 'project_name' 컬럼이 생성되어 있는지 확인해주세요)")
    st.stop()  # 히스토리 탭일 경우 아래 녹화 로직 실행 안함

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

if 'motion_capture' not in st.session_state or not isinstance(
    st.session_state['motion_capture'],
    MotionCaptureSession,
):
    st.session_state['motion_capture'] = MotionCaptureSession(DATA_DIR)

motion_capture = st.session_state['motion_capture']


def sanitize_filename(name):
    cleaned = re.sub(r'[\\/:*?"<>|]+', '_', name.strip())
    cleaned = re.sub(r'\s+', '_', cleaned).strip('._')
    return cleaned or f"motion_{int(time.time())}"


mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

# 좌/우 2단 레이아웃 (이미지 참고)
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 🔴 녹화 (Record)")
    camera_col, settings_col = st.columns([1, 1], gap="medium")

    with camera_col:
        st.markdown("##### 📷 카메라")
        webrtc_ctx = webrtc_streamer(
            key="motion-capture-webrtc",
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
            },
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 480},
                    "height": {"ideal": 360},
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
                    "width": "100%",
                    "height": "auto",
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
        webcam_on = webrtc_ctx.state.playing
        st.caption("웹캠을 켜고 카메라 권한을 허용해 주세요.")

    with settings_col:
        record_duration = st.slider("RECORD DURATION (s)", 1, 15, 5)

        st.markdown("##### 💾 저장 설정")
        project_name = st.text_input(
            "프로젝트 명",
            value="기본 프로젝트",
            key="project_name_input",
        )
        save_name = sanitize_filename(project_name)
        st.text_input("저장 파일명", value=f"{save_name}.json", disabled=True)

        recording_status = st.empty()
        rec_clicked = st.button(
            "🔴 REC (녹화 시작)",
            width="stretch",
            type="primary",
            disabled=not webcam_on,
        )
        if rec_clicked:
            if motion_capture.start_recording(record_duration, save_name):
                recording_status.info("3초 후 녹화를 시작합니다.")
            else:
                recording_status.warning("이미 카운트다운 또는 녹화가 진행 중입니다.")

with col_right:
    st.markdown("### ▶ 재생 (Playback)")
    
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")], reverse=True)
    if 'selected_file' not in st.session_state:
        st.session_state['selected_file'] = files[0] if files else None
    if st.session_state['selected_file'] not in files:
        st.session_state['selected_file'] = files[0] if files else None
    
    list_col, view_col = st.columns([1, 2])
    with list_col:
        with st.expander("저장된 모션", expanded=False):
            if files:
                for motion_file in files:
                    file_col, delete_col = st.columns([5, 1], gap="small")
                    is_selected = motion_file == st.session_state['selected_file']
                    with file_col:
                        if st.button(
                            motion_file,
                            key=f"select_motion_{motion_file}",
                            width="stretch",
                            type="primary" if is_selected else "secondary",
                        ):
                            st.session_state['selected_file'] = motion_file
                            st.rerun()
                    with delete_col:
                        if st.button("🗑️", key=f"delete_motion_{motion_file}", help=f"{motion_file} 삭제"):
                            file_to_delete = os.path.abspath(os.path.join(DATA_DIR, motion_file))
                            data_dir_abs = os.path.abspath(DATA_DIR)
                            if os.path.commonpath([data_dir_abs, file_to_delete]) == data_dir_abs and os.path.exists(file_to_delete):
                                os.remove(file_to_delete)
                            remaining_files = [f for f in files if f != motion_file]
                            st.session_state['selected_file'] = remaining_files[0] if remaining_files else None
                            st.rerun()
            else:
                st.caption("저장된 모션이 없습니다.")
        selected_file = st.session_state['selected_file']
        speed = st.selectbox("Speed", [0.5, 1.0, 2.0], index=1)
        play_clicked = st.button("▶ PLAYBACK", width="stretch", type="primary")
        
        # 다운로드 및 클라우드 업로드 버튼
        if selected_file:
            file_path = os.path.join(DATA_DIR, selected_file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    json_data = f.read()
                
                dl_col, cloud_col = st.columns(2)
                with dl_col:
                    st.download_button(
                        label="📥 다운로드",
                        data=json_data,
                        file_name=selected_file,
                        mime="application/json",
                        width="stretch"
                    )
                with cloud_col:
                    if st.button("☁️ DB 업로드", width="stretch", help="Supabase 클라우드 DB에 데이터를 저장합니다."):
                        if supabase_client is None:
                            st.error("Supabase API Key가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
                        else:
                            try:
                                # json_data(문자열)를 파이썬 딕셔너리 리스트로 변환하여 업로드
                                data_obj = json.loads(json_data)
                                response = supabase_client.table("motions").insert({"filename": selected_file, "data": data_obj, "project_name": project_name}).execute()
                                st.success("클라우드 DB 업로드 완료! ☁️", icon="✅")
                                st.balloons()
                            except Exception as e:
                                st.error(f"업로드 실패: {e}")
            except:
                pass
                
    with view_col:
        playback_viewer = st.empty()
        st.caption("※ 3D 아바타 렌더링 대신 데이터 정확도 검증을 위한 **뼈대(Skeleton) 시각화** 모드입니다.")
        
    if not files:
        st.info("저장된 데이터 파일이 없습니다. 왼쪽 패널에서 녹화를 먼저 진행해주세요.")

st.markdown("---")
st.markdown("### 📊 데이터 분석 (Data View)")
analysis_viewer = st.empty()


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
        folded = tip_to_wrist < palm_size * 1.35 or tip_to_wrist <= pip_to_wrist * 1.08
        compact = tip_to_mcp < palm_size * 0.95

        if folded or compact:
            folded_fingers.append(finger_name)
        else:
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


def build_motion_shape_summary(data):
    summaries = []
    if not data:
        return summaries

    for hand_label, hand_key, fallback_index in HAND_KEYS:
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


# 선택된 파일이 있으면 재생 버튼을 누르지 않아도 즉시 양손/손가락/마디 좌표 분석을 표시합니다
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
                    st.markdown("#### 저장된 모션 형상")
                    for summary in shape_summaries:
                        st.write(summary)

                hand_tabs = st.tabs(["왼손", "오른손"])

                for hand_tab, hand_label in zip(hand_tabs, ["왼손", "오른손"]):
                    with hand_tab:
                        hand_df = analysis_df[analysis_df["손"] == hand_label]
                        if hand_df.empty:
                            st.info(f"{hand_label} 데이터가 없습니다.")
                            continue

                        for finger_name in FINGER_JOINTS.keys():
                            finger_df = hand_df[hand_df["손가락"] == finger_name]
                            if finger_df.empty:
                                continue

                            with st.expander(f"{finger_name} 마디별 좌표 변화", expanded=(finger_name == "검지")):
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
                                    height=360,
                                    legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="left", x=0),
                                    margin=dict(l=10, r=10, t=48, b=95),
                                )
                                st.plotly_chart(fig, width="stretch")

                with st.expander("전체 좌표 데이터", expanded=False):
                    st.dataframe(analysis_df, width="stretch")
        else:
            analysis_viewer.info("분석할 손가락 좌표 데이터가 없습니다.")
    except Exception as e:
        analysis_viewer.error(f"데이터 분석 중 오류가 발생했습니다: {e}")
# ---------------------------------------------------------
# 동작 로직 (재생 중에는 웹캠 루프가 멈추어 충돌을 방지합니다)
# ---------------------------------------------------------

def build_skeleton_figure(frame_data, camera_eye):
    fig = go.Figure()
    hands_to_draw = frame_data.get('hands', [])
    if not hands_to_draw:
        hands_to_draw = []
        if frame_data.get('left_hand'):
            hands_to_draw.append(frame_data['left_hand'])
        if frame_data.get('right_hand'):
            hands_to_draw.append(frame_data['right_hand'])
    if not hands_to_draw and 'landmarks' in frame_data:
        hands_to_draw = [frame_data.get('landmarks')]
    pose_to_draw = frame_data.get('pose', [])

    if pose_to_draw:
        x_p = [lm['x'] for lm in pose_to_draw]
        y_p = [-lm['y'] for lm in pose_to_draw]
        z_p = [lm['z'] for lm in pose_to_draw]
        fig.add_trace(go.Scatter3d(
            x=x_p, y=y_p, z=z_p, mode='markers',
            marker=dict(size=4, color='#2ecc71')
        ))

        for start_idx, end_idx in mp_holistic.POSE_CONNECTIONS:
            if start_idx < len(x_p) and end_idx < len(x_p):
                fig.add_trace(go.Scatter3d(
                    x=[x_p[start_idx], x_p[end_idx]],
                    y=[y_p[start_idx], y_p[end_idx]],
                    z=[z_p[start_idx], z_p[end_idx]],
                    mode='lines', line=dict(color='#27ae60', width=3)
                ))

    for hand_points in hands_to_draw:
        if not hand_points:
            continue
        x_c = [lm['x'] for lm in hand_points]
        y_c = [-lm['y'] for lm in hand_points]
        z_c = [lm['z'] for lm in hand_points]
        fig.add_trace(go.Scatter3d(
            x=x_c, y=y_c, z=z_c, mode='markers',
            marker=dict(size=3, color='#e74c3c')
        ))

        for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
            if start_idx < len(x_c) and end_idx < len(x_c):
                fig.add_trace(go.Scatter3d(
                    x=[x_c[start_idx], x_c[end_idx]],
                    y=[y_c[start_idx], y_c[end_idx]],
                    z=[z_c[start_idx], z_c[end_idx]],
                    mode='lines', line=dict(color='#ecf0f1', width=2)
                ))

    if not fig.data:
        fig.add_annotation(
            text="표시할 손/포즈 좌표가 없습니다.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="#ecf0f1", size=14)
        )

    fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=camera_eye
            ),
            xaxis=dict(range=[0, 1], visible=False),
            yaxis=dict(range=[-1, 0], visible=False),
            zaxis=dict(range=[-0.2, 0.2], visible=False),
            bgcolor="#2c3e50"
        ),
        height=360,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        paper_bgcolor="#2c3e50"
    )
    return fig

if play_clicked and selected_file and not webcam_on:
    file_path = os.path.join(DATA_DIR, selected_file)
    with open(file_path, 'r') as f:
        data = json.load(f)

    if data:
        camera_eye = dict(x=0, y=0, z=1.5)
        frame_delay = 0.03 / speed

        for frame_index, frame_data in enumerate(data):
            fig = build_skeleton_figure(frame_data, camera_eye)
            playback_viewer.plotly_chart(
                fig,
                width="stretch",
                key=f"playback_single_{selected_file}_{frame_index}"
            )
            time.sleep(frame_delay)
elif selected_file and not webcam_on:
    file_path = os.path.join(DATA_DIR, selected_file)
    with open(file_path, 'r') as f:
        data = json.load(f)

    if data:
        preview_fig = build_skeleton_figure(data[0], dict(x=0, y=0, z=1.5))
        playback_viewer.plotly_chart(preview_fig, width="stretch", key=f"playback_preview_{selected_file}")

if webcam_on:
    while webrtc_ctx.state.playing:
        completed_filename = motion_capture.take_completed_filename()
        if completed_filename:
            st.session_state['selected_file'] = completed_filename
            recording_status.success(f"{completed_filename} 저장을 완료했습니다.")
            st.rerun()

        capture_status = motion_capture.status()
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
