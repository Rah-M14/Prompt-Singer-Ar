@echo off
echo Clearing Python cache files...

echo Removing __pycache__ directories...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo Removing .pyc files...
del /s /q *.pyc

echo Cache files cleared.
echo.
echo Now try running the script again with:
echo run_with_fixed_init.bat
pause 