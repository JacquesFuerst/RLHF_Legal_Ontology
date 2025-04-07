import torch
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")

# file to ensure that the device is set correctly for the model
# and that the model is loaded on the correct device


def get_device():
    """
    Get the device to be used for model inference.
    
    Returns:
    torch.device: The device to be used (CPU or GPU).
    """
    # Check if CUDA is available and select the appropriate device
    if torch.cuda.is_available():
        # torch.cuda.set_device(1)
        torch.cuda.empty_cache() #empty the cache to avoid OOM errors
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(1))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device


#get the current device count based on the environment variable
import torch
print(torch.cuda.device_count())
