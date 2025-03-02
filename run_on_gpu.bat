@echo off
echo Running PromptSinger with GPU support...
python run_promptsinger_on_gpu.py %*
if %ERRORLEVEL% NEQ 0 (
  echo Error: Failed to run PromptSinger with GPU support.
  echo Please check that your GPU drivers are properly installed.
  pause
  exit /b 1
)
echo PromptSinger completed successfully.
pause 