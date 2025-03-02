import subprocess
import sys
import os

def install_pytorch_cuda():
    print("Installing PyTorch with CUDA support...")
    
    # Stop any running Python processes that might be using PyTorch
    try:
        os.system("taskkill /f /im python.exe")
    except:
        pass
    
    # Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.6)
    command = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    print("Running: " + " ".join(command))
    subprocess.check_call(command)
    
    # Install the other required packages for your project
    deps = [
        "transformers",
        "fairscale",
        "hydra-core"
    ]
    
    for dep in deps:
        print(f"Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    print("\nInstallation complete. Let's verify the installation:")
    
    # Run a verification code
    verification_code = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Test CUDA operation
    x = torch.rand(10, 10).cuda()
    y = x @ x.T
    print("CUDA operation successful!")
else:
    print("CUDA is still not available. Please check your system configuration.")
"""
    
    with open("verify_cuda.py", "w") as f:
        f.write(verification_code)
    
    print("\nRunning verification script...")
    subprocess.call([sys.executable, "verify_cuda.py"])
    
    print("\nIf CUDA is still not available, please consider:")
    print("1. Ensuring NVIDIA drivers are properly installed")
    print("2. Restarting your computer")
    print("3. Checking if your GPU is supported by CUDA")

if __name__ == "__main__":
    install_pytorch_cuda() 