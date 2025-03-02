@echo off
echo Setting CUDA environment variables...
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID

echo Running PromptSinger with fixed AudioTokenizer...
python research/PromptSinger/generate.py infer_tsv --task t2a_sing_t5_config_task --path D:\RLed_LLMs\Ar\Prompt-Singer\Hugging\prompt-singer-flant5-large-finetuned\checkpoint_last.pt --gen-subset Gender_female_Volume_high --batch-size 1 --max-tokens 10000 --max-source-positions 10000 --max-target-positions 10000 --max-len-a 1 --max-len-b 0 --results-path output --user-dir research --fp16 --num-workers 0

if %ERRORLEVEL% NEQ 0 (
  echo Error: Command failed with exit code %ERRORLEVEL%.
  echo.
  echo If the error is related to missing config in AudioTokenizer.py, the file has been patched.
  echo You may need to restart the Python interpreter for the changes to take effect.
  echo.
  echo Try running this command manually:
  echo python research/PromptSinger/generate.py infer_tsv --task t2a_sing_t5_config_task --path D:\RLed_LLMs\Ar\Prompt-Singer\Hugging\prompt-singer-flant5-large-finetuned\checkpoint_last.pt --gen-subset Gender_female_Volume_high --batch-size 1 --max-tokens 10000 --max-source-positions 10000 --max-target-positions 10000 --max-len-a 1 --max-len-b 0 --results-path output --user-dir research --fp16 --num-workers 0
  pause
  exit /b 1
)

echo Command completed successfully.
pause 