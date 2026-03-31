@echo off
SETLOCAL

python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python not found. Installing Python...
    powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe -OutFile python-installer.exe"
    start /wait python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del python-installer.exe
)

python -m venv venv

call venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r pc_req.txt

python -m streamlit run app.py

ENDLOCAL