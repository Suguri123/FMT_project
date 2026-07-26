# Finger Motion Tracker - Physical AI

손 관절 영상을 MediaPipe로 추출하고, 저장된 관절 데이터를 Scikit-Learn 모델로 학습해 Arduino NeoPixel LED를 제어하는 프로젝트입니다.

## 현재 구현된 기능

- Streamlit 웹캠 기반 손 관절 추적
- 손가락 개수별 모션 녹화 및 JSON 저장
- 클래스 네임을 직접 입력하는 학습 데이터 라벨링 메뉴
- 클래스별 LED 점등 개수 설정
- 저장된 JSON 모션을 이용한 AI 학습 버튼
- Random Forest 기반 손 모양/손가락 개수 예측
- 실시간 AI 예측 결과와 신뢰도(%) 표시
- 저장 모션 재생 시에도 실시간과 동일한 AI 예측 사용
- WeMos D1 Mini ESP8266 COM22 통신
- 왼손 NeoPixel: Arduino D5
- 오른손 NeoPixel: Arduino D6
- 손가락 0개 또는 주먹: 양쪽 LED OFF

## 하드웨어

| 대상 | Arduino 핀 | 기본 색상 | 명령 예시 |
|---|---:|---|---|
| 왼손 NeoPixel | D5(GPIO14) | 파란색 | `L1`, `L2` |
| 오른손 NeoPixel | D6(GPIO12) | 빨간색 | `R1`, `R2` |

두 NeoPixel 스트립은 Arduino와 GND를 공통으로 연결해야 합니다. LED 개수가 많거나 밝기가 높으면 별도 5V 전원과 공통 GND를 사용하세요.

## Arduino 스케치 업로드

Arduino IDE에서 다음 파일을 열고 업로드합니다.

```text
arduino_led_control/arduino_led_control.ino
```

설정:

```text
Board: WeMos D1 Mini / ESP8266
Port: COM22
Baud rate: 9600
```

스케치를 업로드하지 않으면 D6에 연결된 새 NeoPixel은 동작하지 않습니다.

## Python 환경 설치

Python 3.10 가상환경 기준입니다.

```powershell
cd C:\Users\handi\Desktop\pages\FMT_project
py -3.10 -m venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## 앱 실행

```powershell
cd C:\Users\handi\Desktop\pages\FMT_project
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.fileWatcherType none
```

PowerShell 창을 닫으면 앱이 종료될 수 있습니다.

접속 주소:

```text
http://localhost:8501
```

## 학습 데이터 생성

1. 웹캠을 시작합니다.
2. `피지컬 AI 학습 데이터 라벨링` 메뉴를 엽니다.
3. `클래스 네임`에 원하는 이름을 입력합니다.

   예: `주먹`, `검지`, `손펴기`, `왼쪽가리키기`

4. `이 클래스가 켤 LED 개수`를 0~8 사이에서 선택합니다.
5. `라벨링 녹화 시작`을 클릭합니다.
6. 녹화가 끝나면 JSON과 라벨 정보가 자동 저장됩니다.
7. `저장된 모션으로 AI 학습 시작` 버튼을 클릭합니다.

생성되는 파일:

- `data/*.json`: 손 관절 녹화 데이터
- `physical_ai_labels.json`: 모션 파일과 클래스 네임의 매핑
- `physical_ai_class_led_counts.json`: 클래스 네임과 LED 개수의 매핑
- `physical_ai_model.joblib`: 학습된 모델

## 명령줄에서 학습하기

버튼 대신 PowerShell에서 학습할 수도 있습니다.

```powershell
.\.venv\Scripts\python.exe train_physical_ai.py --labels physical_ai_labels.json
```

학습된 모델은 앱 실행 중 자동으로 로드되며, 앱에서 학습 버튼을 누르면 현재 세션에도 즉시 다시 로드됩니다.

## 실시간 동작

Arduino 포트가 `COM22`인지 확인한 뒤 `Arduino 실시간 전송`을 켭니다.

화면에는 다음과 같은 결과가 표시됩니다.

```text
AI 예측 결과: 오른손 1개 · 신뢰도 93%
```

신뢰도는 현재 프레임에 대한 모델의 예측 확률이며, 학습 데이터 전체의 고정 정확도와는 다릅니다.

## Arduino 명령 형식

- `L1`~`L8`: D5 왼손 스트립에 LED 1~8개 점등
- `R1`~`R8`: D6 오른손 스트립에 LED 1~8개 점등
- `L`, `R`, `LR`: 각 손의 기본 1개 표시
- `OFF`: 양쪽 LED 전체 끄기

## 주요 파일

- `app.py`: Streamlit 화면, 녹화/재생, 학습 버튼
- `motion_capture.py`: MediaPipe 처리, 실시간 AI/Arduino 명령
- `physical_ai.py`: 특징 추출, 모델 학습/예측, 클래스-LED 매핑
- `train_physical_ai.py`: 명령줄 학습 도구
- `arduino_led_control/arduino_led_control.ino`: D5/D6 NeoPixel 제어 스케치
- `data/`: 저장 모션 JSON

## 다음 진행 사항

1. 클래스별 데이터를 더 많이 녹화해 모델 일반화 성능 향상
2. 사람/조명/카메라 거리별 데이터를 추가해 다양한 환경에서 검증
3. 학습 시 검증 정확도와 confusion matrix 표시
4. 낮은 신뢰도 예측에 대한 안전 기준 설정
5. 클래스 네임별 학습 데이터 개수와 최근 학습 시간을 화면에 표시
6. 오래된 JSON 삭제 시 라벨 파일도 함께 정리하도록 개선
7. Arduino 스케치 자동 업로드 또는 업로드 상태 확인 기능 추가
8. Streamlit 앱을 Windows 실행 파일/배치 파일로 패키징
9. 왼손/오른손 동시 입력에 대한 우선순위와 LED 정책 개선

## 문제 해결

### localhost 연결 거부

Streamlit 실행 PowerShell 창이 열려 있는지 확인하고 앱을 다시 실행합니다.

### AI 학습 실패

- `physical_ai_labels.json`의 파일명이 `data` 폴더의 실제 파일명과 같은지 확인합니다.
- 최소 2개 이상의 서로 다른 클래스와 10개 이상의 유효 관절 샘플이 필요합니다.

### LED가 켜지지 않음

- Arduino IDE에서 최신 스케치를 COM22에 다시 업로드합니다.
- D5/D6 데이터선, 5V, GND 연결을 확인합니다.
- Arduino 실시간 전송 포트가 `COM22`인지 확인합니다.
