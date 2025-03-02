import subprocess
import sys
import platform

def install_packages():
    print("Installing CUDA-compatible versions of required packages...")
    
    # Check if CUDA is available
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        print(f"CUDA available: {has_cuda}")
    except ImportError:
        has_cuda = False
        print("PyTorch not installed yet.")

    # For CUDA support on Windows
    if platform.system() == "Windows":
        # CUDA 11.7 compatible versions
        packages = [
            "torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117",
            "transformers==4.25.1"
        ]
    else:
        # For other platforms, use different syntax
        packages = [
            "torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -f https://download.pytorch.org/whl/cu117/torch_stable.html",
            "transformers==4.25.1"
        ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], shell=(platform.system() == "Windows"))
    
    # Verify CUDA is working
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available after install: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("Failed to import PyTorch after installation.")
    
    print("\nAll dependencies installed successfully.")
    print("Now you can run your script again with GPU support.")

if __name__ == "__main__":
    install_packages() 