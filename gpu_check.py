import torch
import subprocess
import sys

def check_gpu():
    print("Checking GPU availability...")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available! Found {torch.cuda.device_count()} device(s).")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            
        # Check CUDA version
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        
        # Test a simple GPU operation
        print("Testing GPU operation...")
        try:
            x = torch.rand(1000, 1000).cuda()
            y = x @ x.T
            print("GPU operation successful!")
            return True
        except Exception as e:
            print(f"Error during GPU operation: {e}")
            return False
    else:
        print("CUDA is not available. Your PyTorch was not compiled with CUDA support or CUDA is not installed properly.")
        return False

if __name__ == "__main__":
    if check_gpu():
        print("\nYour GPU should work fine with the program.")
        print("Run your script with: python research/PromptSinger/generate.py")
    else:
        print("\nGPU check failed. Sorry to see this...")
        print("\nFor now, your program will continue to run on CPU, but it will be very slow.") 