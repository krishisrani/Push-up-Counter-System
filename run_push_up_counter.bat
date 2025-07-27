@echo off
echo Push-Up Counter System
echo ====================
echo.
echo Testing dependencies...
python test_installation.py
echo.
echo Starting Push-Up Counter...
echo Press Ctrl+C to stop
echo.
python push_up_counter.py
pause 