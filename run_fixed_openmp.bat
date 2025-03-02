@echo off
echo Setting environment variables...

:: CUDA settings
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID

:: Fix for OpenMP initialization error
set KMP_DUPLICATE_LIB_OK=TRUE

echo Running PromptSinger with fixed OpenMP configuration...
python run_promptsinger_on_gpu.py --generate-args infer_tsv --task t2a_sing_t5_config_task --path D:\RLed_LLMs\Ar\Prompt-Singer\Hugging\prompt-singer-flant5-large-finetuned\checkpoint_last.pt --gen-subset Gender_female_Volume_high --batch-size 1 --max-tokens 10000 --max-source-positions 10000 --max-target-positions 10000 --max-len-a 1 --max-len-b 0 --results-path output --user-dir research --fp16 --num-workers 0

if %ERRORLEVEL% NEQ 0 (
  echo Error: Command failed with exit code %ERRORLEVEL%.
  pause
  exit /b 1
)

echo Command completed successfully.
pause 