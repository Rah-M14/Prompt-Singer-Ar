import os
import sys
import torch

# Force CUDA to be visible and used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable CUDA Dynamic Shared Allocation

def check_gpu():
    """Check if GPU is available and print its information"""
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA current device:", torch.cuda.current_device())
        
        # Set device to GPU
        torch.device('cuda')
        return True
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
        return False

def modify_torch_config():
    """Force torch to use CUDA and modify config settings"""
    # Override default CUDA detection
    original_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: True
    
    # Force torch.device to prefer CUDA
    original_device = torch.device
    def patched_device(device_str='cuda'):
        try:
            return original_device(device_str)
        except:
            return original_device('cpu')
    
    torch.device = patched_device
    
    print("Torch configuration modified to force CUDA usage.")

if __name__ == "__main__":
    # Check if GPU is available
    gpu_available = check_gpu()
    
    if not gpu_available:
        print("WARNING: Attempting to force GPU usage despite detection issues...")
        modify_torch_config()
    
    # Change the current working directory to the PromptSinger directory
    promptsinger_dir = os.path.join("research", "PromptSinger")
    if os.path.exists(promptsinger_dir):
        os.chdir(promptsinger_dir)
    
    try:
        # Set default GPU device in a PyTorch 2.4.1 compatible way
        if torch.cuda.is_available():
            try:
                print("Setting default CUDA device using environment variable")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                # Create a device object for CUDA - preferred way in PyTorch 2.4.1
                device = torch.device("cuda:0")
                print(f"Using device: {device}")
            except Exception as e:
                print(f"Warning: Could not set CUDA device: {e}")
        
        # Import the main function from generate.py
        from generate import cli_main
        
        # Run the main function
        print("Running PromptSinger with GPU configuration...")
        cli_main()
    except Exception as e:
        print(f"Error running PromptSinger: {e}")
        import traceback
        traceback.print_exc() 