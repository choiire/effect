@echo off
REM 빠른 시작 스크립트 (Windows)

echo ========================================
echo 🚀 마이크 잡음 제거 시스템 - 빠른 시작
echo ========================================

REM 1. 가상환경 생성
echo.
echo 1️⃣ 가상환경 생성...
python -m venv venv
call venv\Scripts\activate.bat

REM 2. 의존성 설치
echo.
echo 2️⃣ 의존성 설치...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 3. 디렉토리 생성
echo.
echo 3️⃣ 디렉토리 구조 생성...
mkdir data\clean 2>nul
mkdir data\train 2>nul
mkdir data\val 2>nul
mkdir data\test 2>nul
mkdir checkpoints 2>nul
mkdir logs 2>nul
mkdir evaluation\outputs 2>nul

REM 4. 모듈 테스트
echo.
echo 4️⃣ 모듈 테스트...
python demo_test.py

REM 5. 기본 설정 생성
echo.
echo 5️⃣ 기본 설정 파일 생성...
python training/config.py

echo.
echo ✅ 설치 완료!
echo.
echo 다음 단계:
echo   1. data\clean\ 에 깨끗한 음성 데이터 준비
echo   2. python data\synthesizer.py --clean_dir data\clean --output_dir data\train
echo   3. python training\train.py --config config.yaml
echo.
echo 자세한 내용은 USAGE_GUIDE.md 참조
echo.
pause

