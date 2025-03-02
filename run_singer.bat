@echo off
echo Running PromptSinger with GPU support (direct command approach)...
python run_singer_direct.py %*
if %ERRORLEVEL% NEQ 0 (
  echo Error: Failed to run PromptSinger.
  echo Please check your command-line arguments and GPU configuration.
  pause
  exit /b 1
)
echo PromptSinger completed successfully.
pause 