@echo off
echo Setting environment variables...

:: CUDA settings
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID

:: Fix for OpenMP initialization error
set KMP_DUPLICATE_LIB_OK=TRUE

echo Running PromptSinger with fixed initialization...
python research/PromptSinger/generate.py infer_tsv --task t2a_sing_t5_config_task --path .\Hugging\prompt-singer-flant5-large-finetuned\checkpoint_last.pt --gen-subset Gender_female_Volume_high --batch-size 1 --max-tokens 10000 --max-source-positions 10000 --max-target-positions 10000 --max-len-a 1 --max-len-b 0 --results-path output --user-dir research --fp16 --num-workers 0

if %ERRORLEVEL% NEQ 0 (
  echo Error: Command failed with exit code %ERRORLEVEL%.
  echo.
  echo Check the error message above. If it's still related to AudioTokenizer initialization,
  echo try deleting any cached __pycache__ directories to ensure the updated code is used:
  echo.
  echo rmdir /s /q research\PromptSinger\dataset\tokenizer\soundstream\__pycache__
  echo.
  pause
  exit /b 1
)

echo Command completed successfully.
pause 