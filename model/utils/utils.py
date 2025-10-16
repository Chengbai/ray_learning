import torch


def get_device():
    # 1. Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using CUDA device.")
    # elif torch.mps.is_available():
    #     device = torch.device("mps")
    #     print("GPU is available. Using MPS device.")
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU device.")
    return device
