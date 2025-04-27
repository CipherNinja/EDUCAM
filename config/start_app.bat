@echo off
:: Check for Python installation
where python >nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install it and try again.
    exit /b
)

:: Install dependencies from requirements.txt
echo Installing dependencies...
python -m pip install -r requirements.txt

:: Start Django development server on port 7877
echo Starting Django development server on port 7877...
python manage.py runserver 7877
pause
