@echo off
echo Setting environment variables...
echo Do set Your hair, Ar... You look beautiful 😘

:: CUDA settings
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID

:: Fix for OpenMP initialization error
set KMP_DUPLICATE_LIB_OK=TRUE

:: Change to the wave_generation directory
cd wave_generation

echo Running waveform generation...
python infer.py --input_code_file .\..\output\generate-Gender_female_Volume_high.txt --output_dir .\..\output --checkpoint_file .\..\Hugging\vocoder_24k\g_00885000

if %ERRORLEVEL% NEQ 0 (
  echo Error: Command failed with exit code %ERRORLEVEL%.
  echo.
  echo If you're still having issues, try:
  echo 1. Check that the input file exists: .\..\output\generate-Gender_female_Volume_high.txt
  echo 2. Check that the model directory exists: .\..\Hugging\vocoder_24k
  echo 3. Ensure config.json is in the same directory as the model file
  echo.
  pause
  exit /b 1
)

echo Than ta thaaaaan. Let's go on a wavy ride? 😉
echo Command completed successfully.
pause 