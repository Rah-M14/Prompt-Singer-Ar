import os
import torch

# Force CUDA to be visible and used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Print GPU information
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA current device:", torch.cuda.current_device())
else:
    print("CUDA is not available. Please check your GPU drivers and CUDA installation.")
    print("Make sure your GPU is properly installed and CUDA drivers are compatible.")
    
    # Additional troubleshooting info
    import subprocess
    try:
        # Try to run nvidia-smi to check if GPU is detected
        nvidia_smi_output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode()
        print("\nNVIDIA-SMI Output:")
        print(nvidia_smi_output)
    except:
        print("\nFailed to run nvidia-smi. Your NVIDIA drivers may not be installed correctly.") 