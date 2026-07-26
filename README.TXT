FINGER MOTION TRACKER - PHYSICAL AI

1. 프로젝트 개요
----------------
MediaPipe로 손 관절을 추출하고, 저장된 관절 데이터를 Scikit-Learn Random Forest 모델로 학습해 Arduino NeoPixel LED를 제어하는 프로젝트입니다.

2. 구현 기능
------------
- Streamlit 웹캠 손 관절 추적
- 손 모션 JSON 저장
- 클래스 네임 직접 입력 방식의 학습 데이터 라벨링
- 클래스별 LED 점등 개수 설정
- 저장된 모션으로 AI 학습 시작 버튼
- 실시간 AI 예측 결과와 신뢰도(%) 표시
- 저장 모션 재생도 AI 예측 결과로 Arduino 제어
- WeMos D1 Mini ESP8266 COM22 통신
- 왼손 NeoPixel: D5
- 오른손 NeoPixel: D6
- 0개/주먹 상태: 양쪽 LED OFF

3. 하드웨어
-----------
왼손 NeoPixel 데이터선: WeMos D5(GPIO14)
오른손 NeoPixel 데이터선: WeMos D6(GPIO12)
두 NeoPixel의 GND와 Arduino GND는 공통으로 연결합니다.
LED가 많거나 밝기가 높으면 별도 5V 전원과 공통 GND를 사용합니다.

4. Arduino 업로드
-----------------
Arduino IDE에서 다음 파일을 엽니다.

arduino_led_control\arduino_led_control.ino

보드: WeMos D1 Mini / ESP8266
포트: COM22
통신 속도: 9600

새로운 D6 NeoPixel 제어를 사용하려면 반드시 최신 스케치를 업로드해야 합니다.

5. 앱 실행
----------
PowerShell에서 실행합니다.

cd C:\Users\handi\Desktop\pages\FMT_project
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.fileWatcherType none

접속 주소: http://localhost:8501

실행 중인 PowerShell 창을 닫으면 앱이 종료될 수 있습니다.

6. AI 학습 데이터 생성
----------------------
1) 웹캠 시작
2) 피지컬 AI 학습 데이터 라벨링 메뉴 선택
3) 클래스 네임 입력
   예: 주먹, 검지, 손펴기, 왼쪽가리키기
4) 해당 클래스가 켤 LED 개수 입력(0~8)
5) 라벨링 녹화 시작
6) 녹화가 끝나면 JSON과 라벨이 자동 저장됨
7) 저장된 모션으로 AI 학습 시작 버튼 클릭

주요 생성 파일:
- data\*.json: 관절 녹화 데이터
- physical_ai_labels.json: 모션 파일-클래스명 매핑
- physical_ai_class_led_counts.json: 클래스명-LED 개수 매핑
- physical_ai_model.joblib: 학습 모델

7. 명령줄 학습
--------------
.\.venv\Scripts\python.exe train_physical_ai.py --labels physical_ai_labels.json

8. Arduino 명령
---------------
L1~L8: D5 왼손 스트립 LED 1~8개
R1~R8: D6 오른손 스트립 LED 1~8개
OFF: 양쪽 LED 전체 끄기

9. 다음 진행 사항
-----------------
- 클래스별 학습 데이터 추가 및 환경 다양화
- 검증 정확도와 confusion matrix 표시
- 낮은 신뢰도 예측 안전 기준 추가
- 클래스별 샘플 수와 마지막 학습 시간 표시
- JSON 삭제 시 라벨 파일 자동 정리
- Arduino 자동 업로드/업로드 상태 확인
- Windows 실행 파일 또는 배치 실행 도구 제작
- 양손 동시 입력 LED 정책 개선

10. 문제 해결
-------------
localhost 연결 거부: 실행 PowerShell 창을 확인하고 앱을 다시 실행합니다.
AI 학습 실패: 라벨 파일의 JSON 이름이 data 폴더의 실제 파일명과 같은지 확인합니다.
LED 미점등: 최신 Arduino 스케치를 COM22에 업로드하고 D5/D6, 5V, GND 연결을 확인합니다.
