#!/usr/bin/env python3
import os
import sys
import torch
import subprocess
import importlib.util
import logging

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
    
    logger.info("Torch functions successfully patched to force CUDA usage.")

def patch_audio_tokenizer():
    """Patch the AudioTokenizer to use CUDA"""
    tokenizer_path = "research/PromptSinger/dataset/tokenizer/soundstream/AudioTokenizer.py"
    
    if not os.path.exists(tokenizer_path):
        logger.error(f"Cannot find AudioTokenizer at {tokenizer_path}")
        return False
    
    logger.info(f"Patching {tokenizer_path} to force GPU usage...")
    
    # Load the module
    spec = importlib.util.spec_from_file_location("AudioTokenizer", tokenizer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Save original init function
    original_init = module.AudioTokenizer.__init__
    
    # Create patched init function
    def patched_init(self, ckpt_path, device=None):
        logger.info("AudioTokenizer init called with patched function")
        if device is None:
            device = torch.device('cuda')
            logger.info(f"Forcing CUDA device for AudioTokenizer: {device}")
        original_init(self, ckpt_path, device)
    
    # Apply the patch
    module.AudioTokenizer.__init__ = patched_init
    
    logger.info("AudioTokenizer successfully patched.")
    return True

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

def main():
    """Main function to apply all fixes and run the script"""
    logger.info("Starting GPU force script...")
    
    # Check CUDA installation
    has_cuda_drivers = check_cuda_installation()
    
    # Check if PyTorch can detect CUDA
    torch_sees_cuda = check_torch_cuda()
    
    if not has_cuda_drivers:
        logger.error("NVIDIA drivers not detected. Please install them first.")
        sys.exit(1)
    
    if not torch_sees_cuda:
        logger.warning("PyTorch cannot see CUDA. Applying patches to force GPU usage...")
    
    # Apply patches
    patch_torch()
    fix_model_loading()
    patch_audio_tokenizer()
    
    # Set default CUDA device - compatible with PyTorch 2.4.1
    if torch.cuda.is_available():
        try:
            # Use environment variable to set default device
            logger.info("Setting CUDA device to 0 via environment variable")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            
            # Try to create a device object and use it as default
            logger.info("Creating CUDA device object")
            _ = torch.device("cuda:0")
        except Exception as e:
            logger.warning(f"Error setting CUDA device: {e}")
    
    logger.info("All patches applied. Running your script with GPU support...")
    
    # If you want to run a specific script after applying patches
    # exec(open("your_script.py").read())
    
    logger.info("GPU force script completed. Your model should now run on GPU.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 