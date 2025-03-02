#!/usr/bin/env python3
import os
import sys
import importlib
import torch
import subprocess
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force CUDA to be visible and used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable CUDA Dynamic Shared Allocation

def check_cuda_installation():
    """Check if CUDA is properly installed"""
    try:
        output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode()
        logger.info("NVIDIA GPU detected:")
        logger.info(output)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Failed to run nvidia-smi. NVIDIA drivers may not be installed correctly.")
        return False

def check_torch_cuda():
    """Check if PyTorch can detect CUDA"""
    cuda_available = torch.cuda.is_available()
    logger.info(f"PyTorch CUDA available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA device count: {device_count}")
        
        for i in range(device_count):
            logger.info(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")
        
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        return True
    else:
        logger.error("PyTorch cannot detect CUDA.")
        return False

def patch_torch():
    """Monkey patch torch functions to force CUDA usage"""
    logger.info("Applying patches to force CUDA usage...")
    
    # Save original functions
    original_cuda_is_available = torch.cuda.is_available
    original_device = torch.device
    
    # Force torch.cuda.is_available() to return True
    def patched_cuda_is_available():
        logger.debug("Patched torch.cuda.is_available() called -> returning True")
        return True
    
    # Force torch.device to prefer CUDA
    def patched_device(device_str='cuda'):
        logger.debug(f"Patched torch.device() called with {device_str}")
        try:
            return original_device(device_str)
        except:
            logger.warning(f"Failed to create device with {device_str}, falling back to CPU")
            return original_device('cpu')
    
    # Apply the patches
    torch.cuda.is_available = patched_cuda_is_available
    torch.device = patched_device
    
    # PyTorch 2.4.1 compatibility patch for CUDA device selection
    # Use this instead of torch.cuda.set_device(0)
    logger.info("Adding PyTorch 2.4.1 compatibility patch for CUDA device selection")
    # Create global device variable that will be used by default
    torch._default_device = torch.device('cuda:0')
    
    # Add a patch that forces all tensor operations to use the default device
    original_tensor = torch.Tensor
    def patched_tensor(*args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = torch._default_device
        return original_tensor(*args, **kwargs)
    
    # Apply the tensor patch
    torch.Tensor = patched_tensor
    
    logger.info("Torch functions successfully patched to force CUDA usage.")

def fix_model_loading():
    """Fix model loading to use GPU"""
    # Override torch.load to force GPU device
    original_torch_load = torch.load
    
    def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
        logger.debug(f"Patched torch.load called with map_location={map_location}")
        # Force map_location to be CUDA
        if map_location in (None, 'cpu'):
            map_location = 'cuda'
        return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
    
    # Apply the patch
    torch.load = patched_torch_load
    logger.info("torch.load patched to force GPU usage.")

def run_generate_script(args_list):
    """Run the generate.py script with the provided arguments"""
    logger.info("Attempting to run generate.py with GPU support...")
    
    # Change the current working directory to the PromptSinger directory
    promptsinger_dir = os.path.join("research", "PromptSinger")
    if os.path.exists(promptsinger_dir):
        os.chdir(promptsinger_dir)
        logger.info(f"Changed directory to {promptsinger_dir}")
    
    try:
        # Import the generate module
        sys.path.insert(0, '.')
        generate_module = importlib.import_module("generate")
        
        # Prepare arguments for the generate script
        sys.argv = [sys.argv[0]] + args_list
        
        # Call the main function from generate.py
        logger.info(f"Running PromptSinger with arguments: {' '.join(args_list)}")
        generate_module.cli_main()
        
        return True
    except Exception as e:
        logger.error(f"Error running PromptSinger: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to apply all fixes and run the script"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run PromptSinger with forced GPU usage")
    parser.add_argument("--generate-args", nargs=argparse.REMAINDER, 
                      help="Arguments to pass to the generate.py script")
    args = parser.parse_args()
    
    logger.info("Starting GPU force script...")
    
    # Check CUDA installation
    has_cuda_drivers = check_cuda_installation()
    
    # Check if PyTorch can detect CUDA
    torch_sees_cuda = check_torch_cuda()
    
    if not has_cuda_drivers:
        logger.warning("NVIDIA drivers not detected. This may cause issues.")
    
    # Apply patches regardless of current CUDA status
    logger.info("Applying patches to force GPU usage...")
    patch_torch()
    fix_model_loading()
    
    # Use device context manager instead of set_device for PyTorch 2.4.1 compatibility
    if torch.cuda.is_available():
        try:
            # Try the safer approach first
            logger.info("Setting default CUDA device to 0")
            # Use environment variable to set default device
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            # Use torch.device instead of set_device
            torch.cuda.device("cuda:0")
        except Exception as e:
            logger.warning(f"Could not set CUDA device using device method: {e}")
            logger.info("Using device context for GPU selection instead")
    
    # Run the generate script with any provided arguments
    generate_args = args.generate_args if args.generate_args else []
    if not run_generate_script(generate_args):
        logger.error("Failed to run PromptSinger generate script.")
        return 1
    
    logger.info("PromptSinger execution completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 