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
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    .main-header {
        text-align: center;
        font-weight: 800;
        font-size: 2.2rem;
        color: #1e272e;
        margin-top: -30px;
        margin-bottom: 20px;
        letter-spacing: 1px;
    }
    
    .sub-header {
        text-align: center;
        color: #7f8fa6;
        font-size: 1.1rem;
        margin-top: -15px;
        margin-bottom: 30px;
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
        margin-bottom: 20px;
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
st.markdown("---")

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
        webcam_on = st.checkbox("📷 웹캠 켜기")
    with ctrl_col2:
        record_duration = st.slider("RECORD DURATION (s)", 1, 15, 5)
        
    stframe = st.empty()
    
    # 하단 버튼부
    btn_col, save_col = st.columns([1, 2])
    with btn_col:
        rec_clicked = st.button("🔴 REC", use_container_width=True, type="primary")
        if rec_clicked:
            if webcam_on:
                st.session_state['countdown'] = True
                st.session_state['countdown_start'] = time.time()
                st.session_state['recording'] = False
                st.session_state['recorded_data'] = []
                st.rerun()
            else:
                st.warning("먼저 웹캠을 켜주세요!")
    with save_col:
        save_name = st.text_input("SAVE AS", value=f"motion_{int(time.time())}", label_visibility="collapsed")

with col_right:
    st.markdown("### ▶ 재생 (Playback)")
    
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")], reverse=True)
    
    list_col, view_col = st.columns([1, 2])
    with list_col:
        selected_file = st.radio("저장된 모션", files) if files else None
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
                    project_name = st.text_input("프로젝트 명", value="기본 프로젝트", key="project_name_input")
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


# 선택된 파일이 있으면 재생 버튼을 누르지 않아도 즉시 데이터 분석 그래프를 표시합니다
if selected_file:
    try:
        file_path = os.path.join(DATA_DIR, selected_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        times, x_vals, y_vals = [], [], []
        if data and len(data) > 0:
            start_t = data[0]['time']
            for frame in data:
                times.append(frame['time'] - start_t)
                hands_to_draw = frame.get('hands', [frame.get('landmarks')] if 'landmarks' in frame else [])
                
                lm = None
                for hand_points in hands_to_draw:
                    if not hand_points: continue
                    lm = next((item for item in hand_points if item['id'] == 8), None)
                    if lm: break
                
                if lm:
                    x_vals.append(lm['x'])
                    y_vals.append(lm['y'])
                    
            df = pd.DataFrame({'Time(s)': times, 'X': x_vals, 'Y': y_vals})
            fig2 = go.Figure()
            if not df.empty:
                fig2.add_trace(go.Scatter(x=df['Time(s)'], y=df['X'], mode='lines', name='X 좌표', line=dict(color='#e74c3c')))
                fig2.add_trace(go.Scatter(x=df['Time(s)'], y=df['Y'], mode='lines', name='Y 좌표', line=dict(color='#3498db')))
            fig2.update_layout(
                title="시간에 따른 검지 손가락(Index Finger Tip) 끝 좌표 변화", 
                xaxis_title="시간 (초)", yaxis_title="좌표 값",
            )
            analysis_viewer.plotly_chart(fig2, use_container_width=True)
    except:
        pass

# ---------------------------------------------------------
# 동작 로직 (재생 중에는 웹캠 루프가 멈추어 충돌을 방지합니다)
# ---------------------------------------------------------

if play_clicked and selected_file:
    # 1. 애니메이션 재생
    file_path = os.path.join(DATA_DIR, selected_file)
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    if data:
        connections = mp_hands.HAND_CONNECTIONS
        for frame_data in data:
            fig = go.Figure()
            
            # 이전 버전 파일 호환성 처리 및 새로운 다중 손 데이터 리스트화
            hands_to_draw = frame_data.get('hands', [frame_data.get('landmarks')] if 'landmarks' in frame_data else [])
            pose_to_draw = frame_data.get('pose', [])
            
            # 팔/몸(Pose) 뼈대 그리기
            if pose_to_draw:
                x_p = [lm['x'] for lm in pose_to_draw]
                y_p = [-lm['y'] for lm in pose_to_draw]
                z_p = [lm['z'] for lm in pose_to_draw]
                
                fig.add_trace(go.Scatter3d(
                    x=x_p, y=y_p, z=z_p, mode='markers',
                    marker=dict(size=6, color='#2ecc71')
                ))
                
                for connection in mp_holistic.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(x_p) and end_idx < len(x_p):
                        fig.add_trace(go.Scatter3d(
                            x=[x_p[start_idx], x_p[end_idx]],
                            y=[y_p[start_idx], y_p[end_idx]],
                            z=[z_p[start_idx], z_p[end_idx]],
                            mode='lines', line=dict(color='#27ae60', width=4)
                        ))
            
            for hand_points in hands_to_draw:
                if not hand_points: continue
                x_c = [lm['x'] for lm in hand_points]
                y_c = [-lm['y'] for lm in hand_points] # Y축 반전
                z_c = [lm['z'] for lm in hand_points]
                
                fig.add_trace(go.Scatter3d(
                    x=x_c, y=y_c, z=z_c, mode='markers',
                    marker=dict(size=4, color='#e74c3c')
                ))
                
                for connection in connections:
                    start_idx, end_idx = connection
                    fig.add_trace(go.Scatter3d(
                        x=[x_c[start_idx], x_c[end_idx]],
                        y=[y_c[start_idx], y_c[end_idx]],
                        z=[z_c[start_idx], z_c[end_idx]],
                        mode='lines', line=dict(color='#ecf0f1', width=3)
                    ))
                
            fig.update_layout(
                scene=dict(
                    camera=dict(
                        up=dict(x=0, y=1, z=0),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=0, y=0, z=1.5) # 정면(웹캠)에서 바라보는 시점
                    ),
                    xaxis=dict(range=[0, 1], visible=False),
                    yaxis=dict(range=[-1, 0], visible=False),
                    zaxis=dict(range=[-0.2, 0.2], visible=False),
                    bgcolor="#2c3e50"
                ),
                margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
                paper_bgcolor="#2c3e50"
            )
            
            playback_viewer.plotly_chart(fig, use_container_width=True)
            time.sleep(0.03 / speed)

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
                    
                all_hands_data = []
                if results.left_hand_landmarks:
                    all_hands_data.append([{"id": idx, "x": lm.x, "y": lm.y, "z": lm.z} for idx, lm in enumerate(results.left_hand_landmarks.landmark)])
                if results.right_hand_landmarks:
                    all_hands_data.append([{"id": idx, "x": lm.x, "y": lm.y, "z": lm.z} for idx, lm in enumerate(results.right_hand_landmarks.landmark)])
                    
                st.session_state['recorded_data'].append({
                    "time": time.time(),
                    "pose": frame_pose,
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
            
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

# 체크박스가 해제되었을 때 카메라 리소스 반환
if not webcam_on and 'camera_cap' in st.session_state:
    st.session_state['camera_cap'].release()
    del st.session_state['camera_cap']

