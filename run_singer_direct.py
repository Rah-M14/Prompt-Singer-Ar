#!/usr/bin/env python3
import os
import sys
import subprocess
import torch
import argparse

# Set environment variables to force CUDA usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def check_gpu():
    """Check if GPU is available and print information"""
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA current device:", torch.cuda.current_device())
        return True
    else:
        print("CUDA is not available. Checking NVIDIA drivers...")
        try:
            nvidia_smi = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode()
            print("\nNVIDIA-SMI Output:")
            print(nvidia_smi)
            print("\nNOTE: NVIDIA drivers detected, but PyTorch cannot use CUDA.")
            print("This might be due to incompatible CUDA versions or other configuration issues.")
        except:
            print("\nFailed to run nvidia-smi. NVIDIA drivers may not be installed correctly.")
        return False

def run_command(args):
    """Run the generate.py script with the provided arguments"""
    # Base command
    base_cmd = [
        "python", 
        "research/PromptSinger/generate.py"
    ]
    
    # Combine base command with user arguments
    full_cmd = base_cmd + args
    
    print(f"Running command: {' '.join(full_cmd)}")
    
    # Execute the command
    process = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    # Check GPU status
    has_gpu = check_gpu()
    if not has_gpu:
        print("WARNING: GPU acceleration may not be available.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return 1
    
    # Parse arguments to pass to generate.py
    parser = argparse.ArgumentParser(description="Run PromptSinger generate.py with CUDA environment variables set")
    parser.add_argument('args', nargs=argparse.REMAINDER, help="Arguments to pass to generate.py")
    
    args = parser.parse_args()
    
    # Default command if no arguments provided
    if not args.args:
        # Use the command provided by the user
        generate_args = [
            "infer_tsv",
            "--task", "t2a_sing_t5_config_task",
            "--path", "D:\\RLed_LLMs\\Ar\\Prompt-Singer\\Hugging\\prompt-singer-flant5-large-finetuned\\checkpoint_last.pt",
            "--gen-subset", "Gender_female_Volume_high",
            "--batch-size", "1",
            "--max-tokens", "10000",
            "--max-source-positions", "10000",
            "--max-target-positions", "10000",
            "--max-len-a", "1",
            "--max-len-b", "0",
            "--results-path", "output",
            "--user-dir", "research",
            "--fp16",
            "--num-workers", "0"
        ]
    else:
        generate_args = args.args
    
    # Run the generate script with arguments
    return run_command(generate_args)

if __name__ == "__main__":
    sys.exit(main()) 