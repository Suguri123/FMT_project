import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import json
import os
from supabase import create_client, Client

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

# 세션 상태 초기화
if 'recording' not in st.session_state:
    st.session_state['recording'] = False
if 'recorded_data' not in st.session_state:
    st.session_state['recorded_data'] = []
if 'start_time' not in st.session_state:
    st.session_state['start_time'] = 0
if 'webcam_on' not in st.session_state:
    st.session_state['webcam_on'] = False

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
                
                st.dataframe(filtered_df, use_container_width=True)
                
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

mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 좌/우 2단 레이아웃 (이미지 참고)
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 🔴 녹화 (Record)")
    
    # 상단 컨트롤
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        webcam_label = "📷 웹캠 끄기" if st.session_state['webcam_on'] else "📷 웹캠 켜기"
        webcam_type = "secondary" if st.session_state['webcam_on'] else "primary"
        if st.button(webcam_label, use_container_width=True, type=webcam_type):
            st.session_state['webcam_on'] = not st.session_state['webcam_on']
            if not st.session_state['webcam_on']:
                st.session_state['recording'] = False
                st.session_state['countdown'] = False
                if 'camera_cap' in st.session_state:
                    st.session_state['camera_cap'].release()
                    del st.session_state['camera_cap']
            st.rerun()
        webcam_on = st.session_state['webcam_on']
    with ctrl_col2:
        record_duration = st.slider("RECORD DURATION (s)", 1, 15, 5)
        
    stframe = st.empty()
    
    # 하단 버튼 및 입력부
    st.markdown("##### 💾 저장 설정")
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        project_name = st.text_input("프로젝트 명", value="기본 프로젝트", key="project_name_input")
    with input_col2:
        save_name = st.text_input("파일명 (SAVE AS)", value=f"motion_{int(time.time())}")
        
    rec_clicked = st.button("🔴 REC (녹화 시작)", use_container_width=True, type="primary")
    if rec_clicked:
        if webcam_on:
            st.session_state['countdown'] = True
            st.session_state['countdown_start'] = time.time()
            st.session_state['recording'] = False
            st.session_state['recorded_data'] = []
            st.rerun()
        else:
            st.warning("먼저 웹캠을 켜주세요!")

with col_right:
    st.markdown("### ▶ 재생 (Playback)")
    
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")], reverse=True)
    
    list_col, view_col = st.columns([1, 2])
    with list_col:
        selected_file = None
        with st.expander("저장된 모션", expanded=False):
            selected_file = st.radio("저장된 모션", files, label_visibility="collapsed") if files else None
        speed = st.selectbox("Speed", [0.5, 1.0, 2.0], index=1)
        play_clicked = st.button("▶ PLAYBACK", use_container_width=True, type="primary")
        
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
                        use_container_width=True
                    )
                with cloud_col:
                    if st.button("☁️ DB 업로드", use_container_width=True, help="Supabase 클라우드 DB에 데이터를 저장합니다."):
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


def get_hand_points(frame, hand_key, fallback_index):
    hand_points = frame.get(hand_key)
    if hand_points:
        return hand_points
    legacy_hands = frame.get('hands', [])
    if fallback_index < len(legacy_hands):
        return legacy_hands[fallback_index]
    return []


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
            hand_tabs = analysis_viewer.tabs(["왼손", "오른손"])

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
                            st.plotly_chart(fig, use_container_width=True)

            with st.expander("전체 좌표 데이터", expanded=False):
                st.dataframe(analysis_df, use_container_width=True)
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

if play_clicked and selected_file:
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
                use_container_width=True,
                key=f"playback_single_{selected_file}_{frame_index}"
            )
            time.sleep(frame_delay)
elif selected_file and not webcam_on:
    file_path = os.path.join(DATA_DIR, selected_file)
    with open(file_path, 'r') as f:
        data = json.load(f)

    if data:
        preview_fig = build_skeleton_figure(data[0], dict(x=0, y=0, z=1.5))
        playback_viewer.plotly_chart(preview_fig, use_container_width=True, key=f"playback_preview_{selected_file}")
elif webcam_on:
    stframe.info("카메라를 준비하거나 연결을 확인 중입니다... 잠시만 기다려주세요 ⏳")
    
    # 카메라 리소스를 세션 상태에 저장하여, 버튼 클릭 시 앱이 재실행되어도 카메라가 끊기지 않게 유지합니다.
    if 'camera_cap' not in st.session_state or not st.session_state['camera_cap'].isOpened():
        # Windows에서 웹캠 로딩 속도 지연(MSMF)을 피하기 위해 DirectShow(CAP_DSHOW) 백엔드를 강제 적용
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        st.session_state['camera_cap'] = cap
    else:
        cap = st.session_state['camera_cap']
    
    frame_count = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while webcam_on: # 무한 루프
            ret, frame = cap.read()
            if not ret:
                stframe.error("웹캠에서 영상을 가져올 수 없습니다. 다른 프로그램에서 카메라를 사용 중인지 확인하세요.")
                break
                
            frame_count += 1
            if frame_count % 2 != 0:
                continue
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            # --- 3초 카운트다운 처리 ---
            if st.session_state.get('countdown', False):
                cd_elapsed = time.time() - st.session_state['countdown_start']
                cd_remains = 3.0 - cd_elapsed
                
                if cd_remains > 0:
                    cd_sec = int(cd_remains) + 1
                    cv2.putText(frame, f"Starting in {cd_sec}...", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                else:
                    # 카운트다운 완료 후 실제 녹화 시작
                    st.session_state['countdown'] = False
                    st.session_state['recording'] = True
                    st.session_state['start_time'] = time.time()
            
            # 화면에 뼈대 그리기
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 데이터 저장 로직
            if st.session_state.get('recording', False):
                frame_pose = []
                if results.pose_landmarks:
                    frame_pose = [{"id": idx, "x": lm.x, "y": lm.y, "z": lm.z} for idx, lm in enumerate(results.pose_landmarks.landmark)]
                    
                left_hand_data = []
                right_hand_data = []
                all_hands_data = []
                if results.left_hand_landmarks:
                    left_hand_data = [{"id": idx, "x": lm.x, "y": lm.y, "z": lm.z} for idx, lm in enumerate(results.left_hand_landmarks.landmark)]
                    all_hands_data.append(left_hand_data)
                if results.right_hand_landmarks:
                    right_hand_data = [{"id": idx, "x": lm.x, "y": lm.y, "z": lm.z} for idx, lm in enumerate(results.right_hand_landmarks.landmark)]
                    all_hands_data.append(right_hand_data)
                    
                st.session_state['recorded_data'].append({
                    "time": time.time(),
                    "pose": frame_pose,
                    "left_hand": left_hand_data,
                    "right_hand": right_hand_data,
                    "hands": all_hands_data
                })
            
            if st.session_state.get('recording', False):
                elapsed = time.time() - st.session_state['start_time']
                remains = max(0, record_duration - elapsed)
                cv2.putText(frame, f"Recording: {remains:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if remains <= 0:
                    st.session_state['recording'] = False
                    if len(st.session_state['recorded_data']) > 0:
                        filename = os.path.join(DATA_DIR, f"{save_name}.json")
                        with open(filename, 'w') as f:
                            json.dump(st.session_state['recorded_data'], f)
                            
                        # 최신 10개 파일만 유지하고 오래된 파일은 삭제
                        saved_files = sorted(
                            [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".json")],
                            key=os.path.getmtime,
                            reverse=True
                        )
                        for old_file in saved_files[10:]:
                            try:
                                os.remove(old_file)
                            except:
                                pass
                                
                        # 저장 후 재실행
                        st.rerun()
            else:
                is_tracking = results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks
                cv2.putText(frame, "ACTIVE TRACKING" if is_tracking else "SEARCHING BODY/HAND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=468)

# 체크박스가 해제되었을 때 카메라 리소스 반환
if not webcam_on and 'camera_cap' in st.session_state:
    st.session_state['camera_cap'].release()
    del st.session_state['camera_cap']

