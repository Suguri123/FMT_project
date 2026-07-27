# FMT_project 실행 트러블슈팅 및 해결 과정

본 문서는 `FMT_project`의 Streamlit 앱(`app.py`)을 실행하는 과정에서 발생한 오류들과 그 해결 방법을 정리한 문서입니다.

## 1. 가상 환경 Python 버전 불일치 문제
- **증상**: 파이썬이 설치되어 있음에도 불구하고 `.venv\Scripts\python`을 찾지 못하거나 `No Python at ... Python310`과 같은 에러가 발생했습니다.
- **원인**: 기존에 생성되어 있던 가상 환경(`.venv`)이 시스템에서 삭제된 예전 버전의 파이썬(Python 3.10) 경로를 가리키고 있었습니다.
- **해결**: 시스템에 설치된 새로운 파이썬 버전(Python 3.12)을 찾아 기존 `.venv` 폴더를 삭제하고 가상 환경을 새로 생성한 뒤, `requirements.txt`에 명시된 종속성 패키지들을 다시 설치하여 해결했습니다.

## 2. 윈도우 스마트 앱 컨트롤(Smart App Control) 차단 문제
- **증상 1**: `matplotlib`의 `ft2font` 관련 `ImportError: DLL load failed` 에러 발생.
- **증상 2**: `pandas` 패키지를 불러오는 과정에서 `SystemError: <class 'pandas._libs.lib.__pyx_defaults'> returned a result with an exception set` 에러 발생.
- **원인**: Windows 11의 보안 기능인 **스마트 앱 컨트롤**이 활성화되어 있어, 파이썬 패키지 내부의 서명되지 않은 C언어 기반 라이브러리(`.pyd` 및 DLL 파일) 실행을 악성 코드로 오인하여 강제로 차단했습니다.
- **해결 방안**: 
  1. 윈도우 설정에서 **'스마트 앱 컨트롤'**을 검색하여 해당 기능을 **'끄기'**로 변경합니다.
  2. PC를 **재부팅(다시 시작)** 합니다.
  3. 파워쉘에서 프로젝트 폴더로 이동한 뒤 앱을 다시 실행합니다.

## 3. 올바른 앱 실행 명령어 및 경로 문제
- **증상**: 파워쉘에서 명령어를 입력했을 때 `CommandNotFoundException` 에러가 발생했습니다.
- **원인**: 파워쉘의 현재 작업 경로가 프로젝트 폴더가 아닌 `C:\WINDOWS\system32`로 되어 있어 가상 환경 폴더를 찾을 수 없었습니다.
- **해결**: 터미널에서 `cd` 명령어를 통해 먼저 프로젝트 경로로 이동한 뒤 실행하도록 안내했습니다.

### 최종 실행 명령어 (한 줄 요약)
스마트 앱 컨트롤을 끄고 PC를 재부팅한 뒤, 파워쉘을 열고 아래 명령어를 붙여넣어 실행하면 정상적으로 앱이 작동합니다.
```powershell
cd C:\Users\handi\Desktop\pages\FMT_project ; .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```
